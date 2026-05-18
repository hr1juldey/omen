#!/usr/bin/env python3
"""Prove multi-layer conv2d backward works on GPU — no conv_transpose, no cuDNN.

Uses pure nabla im2col + matmul for forward. Nabla's built-in autodiff
handles backward through each individual op (pad, slice, reshape, matmul).
No custom Operation, no conv_transpose, no cuDNN.

This is the CRITICAL test: native nb.conv2d crashes with cudnnCreate when
2+ layers use backward. This approach uses im2col+matmul instead.

Progressive scale-up: 16x16 → 32x32 → 64x64 with RAM cleanup between phases.
"""

import gc
import os
import sys
import time

import numpy as np
import nabla as nb
from max.driver import CPU, Accelerator, accelerator_count
from nabla.nn.optim import adamw_init
from omen.kernels.activations import silu_gpu, square

LIMIT_MB = 20 * 1024


def _rss():
    try:
        text = open(f"/proc/{os.getpid()}/status").read()
        ln = next(ln for ln in text.splitlines() if ln.startswith("VmRSS:"))
        return int(ln.split()[1]) // 1024
    except Exception:
        return 0


def guard(label=""):
    rss = _rss()
    if rss > LIMIT_MB:
        print(f"KILL: RSS={rss}MB > {LIMIT_MB}MB {label}")
        sys.exit(99)
    print(f"  [guard] RSS={rss}MB")
    return rss


DEV = None


def dev():
    global DEV
    if DEV is None:
        DEV = Accelerator() if accelerator_count() > 0 else CPU()
    return DEV


def to_dev(arr):
    return nb.ops.transfer_to(nb.Tensor.from_dlpack(arr.astype(np.float32)), dev())


def to_cpu(tensor):
    return nb.ops.transfer_to(tensor, CPU()).to_numpy()


def cleanup():
    gc.collect()


# ── im2col conv2d (pure nabla, no cuDNN) ─────────────────────
def _conv_out_size(in_size, kernel, stride, pad):
    return (in_size + 2 * pad - kernel) // stride + 1


def conv2d_im2col(x, filt, stride=1, padding=0, bias=None):
    """Conv2d via im2col + matmul. Nabla auto-diffs through each op.

    Forward: pad(x) → extract patches (slice) → matmul(patches, filt_flat)
    Backward: automatic via nabla's per-op VJPs (no conv_transpose!)
    """
    if isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride

    if isinstance(padding, int):
        ph, pw = padding, padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            ph, pw = padding
        elif len(padding) == 4:
            ph, pw = padding[0], padding[2]
        else:
            ph, pw = 0, 0
    else:
        ph, pw = 0, 0

    kh, kw, c_in, c_out = (int(d) for d in filt.shape)
    b, h, w, _ = (int(d) for d in x.shape)

    if ph > 0 or pw > 0:
        x = nb.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    h_out = _conv_out_size(h, kh, sh, ph)
    w_out = _conv_out_size(w, kw, sw, pw)

    patches = []
    for ki in range(kh):
        for kj in range(kw):
            patch = x[:, ki:ki + h_out, kj:kj + w_out, :]
            patch = nb.reshape(patch, (b * h_out * w_out, c_in))
            patches.append(patch)

    patches = nb.concatenate(patches, axis=1)
    filt_flat = nb.reshape(filt, (kh * kw * c_in, c_out))
    out_flat = nb.matmul(patches, filt_flat)
    out = nb.reshape(out_flat, (b, h_out, w_out, c_out))

    if bias is not None:
        out = out + bias
    return out


# ── Phase runners ────────────────────────────────────────────
def phase1():
    """16x16, 4→8, 1 conv2d forward+backward."""
    print("\n=== Phase 1: 16x16, 4→8, 1 conv2d ===")
    guard("start")

    x = to_dev(np.random.randn(1, 16, 16, 4).astype(np.float32) * 0.1)
    w = to_dev(np.random.randn(3, 3, 4, 8).astype(np.float32) * 0.01)

    # Forward
    y = conv2d_im2col(x, w, stride=1, padding=1)
    y_np = to_cpu(y)
    print(f"  Forward: (1,16,16,4) → {y_np.shape}")
    assert y_np.shape == (1, 16, 16, 8), f"Bad shape: {y_np.shape}"

    # Compare against native
    y_native = nb.conv2d(x, w, padding=(1, 1, 1, 1))
    diff = np.max(np.abs(y_np - to_cpu(y_native)))
    print(f"  vs native nb.conv2d: max_diff={diff:.6f}")

    # Backward
    def loss(p, x_in):
        return nb.mean(square(conv2d_im2col(x_in, p["w"], stride=1, padding=1)))

    lv, grads = nb.value_and_grad(loss, argnums=0)({"w": w}, x)
    nb.realize_all(lv, grads["w"])
    gw = to_cpu(grads["w"])
    print(f"  loss={float(to_cpu(lv)):.6f} grad_finite={np.all(np.isfinite(gw))}")
    assert np.all(np.isfinite(gw)), "NaN grad!"
    print("  Phase 1 PASSED")
    del x, w, y, y_native, lv, grads, gw
    cleanup()
    guard("end")


def phase2():
    """16x16, 4→8→16, 2 conv2d backward — THE CRITICAL TEST."""
    print("\n=== Phase 2: 16x16, 4→8→16, 2 conv2d (CRITICAL) ===")
    guard("start")

    x = to_dev(np.random.randn(1, 16, 16, 4).astype(np.float32) * 0.1)
    w0 = to_dev(np.random.randn(3, 3, 4, 8).astype(np.float32) * 0.01)
    w1 = to_dev(np.random.randn(3, 3, 8, 16).astype(np.float32) * 0.01)

    def loss(p, x_in):
        h = conv2d_im2col(x_in, p["w0"], stride=1, padding=1)
        h = conv2d_im2col(h, p["w1"], stride=1, padding=1)
        return nb.mean(square(h))

    lv, grads = nb.value_and_grad(loss, argnums=0)(
        {"w0": w0, "w1": w1}, x
    )
    nb.realize_all(lv, grads["w0"], grads["w1"])
    loss_f = float(to_cpu(lv))
    g0 = to_cpu(grads["w0"])
    g1 = to_cpu(grads["w1"])
    print(f"  loss={loss_f:.6f}")
    print(f"  g0: shape={g0.shape} finite={np.all(np.isfinite(g0))}")
    print(f"  g1: shape={g1.shape} finite={np.all(np.isfinite(g1))}")
    assert np.all(np.isfinite(g0)), "NaN g0!"
    assert np.all(np.isfinite(g1)), "NaN g1!"
    print("  Phase 2 PASSED — multi-layer conv2d backward works WITHOUT cuDNN!")
    del x, w0, w1, lv, grads, g0, g1
    cleanup()
    guard("end")


def phase3():
    """32x32, 4→16→32, 2 conv2d backward."""
    print("\n=== Phase 3: 32x32, 4→16→32, 2 conv2d ===")
    guard("start")

    x = to_dev(np.random.randn(1, 32, 32, 4).astype(np.float32) * 0.1)
    w0 = to_dev(np.random.randn(3, 3, 4, 16).astype(np.float32) * 0.01)
    w1 = to_dev(np.random.randn(3, 3, 16, 32).astype(np.float32) * 0.01)

    def loss(p, x_in):
        h = conv2d_im2col(x_in, p["w0"], stride=1, padding=1)
        h = conv2d_im2col(h, p["w1"], stride=1, padding=1)
        return nb.mean(square(h))

    lv, grads = nb.value_and_grad(loss, argnums=0)(
        {"w0": w0, "w1": w1}, x
    )
    nb.realize_all(lv, grads["w0"], grads["w1"])
    loss_f = float(to_cpu(lv))
    g0 = to_cpu(grads["w0"])
    g1 = to_cpu(grads["w1"])
    print(f"  loss={loss_f:.6f} g0_finite={np.all(np.isfinite(g0))} g1_finite={np.all(np.isfinite(g1))}")
    assert np.all(np.isfinite(g0)) and np.all(np.isfinite(g1)), "NaN!"
    print("  Phase 3 PASSED")
    del x, w0, w1, lv, grads, g0, g1
    cleanup()
    guard("end")


def phase4():
    """64x64, 4→16→32, 2 conv2d backward."""
    print("\n=== Phase 4: 64x64, 4→16→32, 2 conv2d ===")
    guard("start")

    x = to_dev(np.random.randn(1, 64, 64, 4).astype(np.float32) * 0.1)
    w0 = to_dev(np.random.randn(3, 3, 4, 16).astype(np.float32) * 0.01)
    w1 = to_dev(np.random.randn(3, 3, 16, 32).astype(np.float32) * 0.01)

    def loss(p, x_in):
        h = conv2d_im2col(x_in, p["w0"], stride=1, padding=1)
        h = conv2d_im2col(h, p["w1"], stride=1, padding=1)
        return nb.mean(square(h))

    t0 = time.time()
    lv, grads = nb.value_and_grad(loss, argnums=0)(
        {"w0": w0, "w1": w1}, x
    )
    nb.realize_all(lv, grads["w0"], grads["w1"])
    dt = time.time() - t0
    loss_f = float(to_cpu(lv))
    print(f"  loss={loss_f:.6f} first_compile={dt:.1f}s")
    g0 = to_cpu(grads["w0"])
    g1 = to_cpu(grads["w1"])
    assert np.all(np.isfinite(g0)) and np.all(np.isfinite(g1)), "NaN!"
    print("  Phase 4 PASSED")
    del x, w0, w1, lv, grads, g0, g1
    cleanup()
    guard("end")


def phase5():
    """64x64, 2+ conv2d + 10-step AdamW training loop."""
    print("\n=== Phase 5: 64x64 training loop (2 conv2d + AdamW) ===")
    guard("start")

    STEPS = 10
    LR = 1e-3
    RES = 64

    p_np = {
        "w0": np.random.randn(3, 3, 4, 16).astype(np.float32) * 0.01,
        "b0": np.zeros(16, dtype=np.float32),
        "w1": np.random.randn(3, 3, 16, 32).astype(np.float32) * 0.01,
        "b1": np.zeros(32, dtype=np.float32),
        "dw": np.random.randn(32, RES * RES * 3).astype(np.float32) * 0.01,
        "db": np.zeros(RES * RES * 3, dtype=np.float32),
    }
    n_p = sum(v.size for v in p_np.values())
    print(f"  {len(p_np)} tensors, {n_p:,} params")

    opt_m = {k: np.zeros_like(v) for k, v in p_np.items()}
    opt_v = {k: np.zeros_like(v) for k, v in p_np.items()}

    noisy_np = np.random.randn(1, RES, RES, 4).astype(np.float32) * 0.1
    gt_np = np.random.randn(1, RES, RES, 3).astype(np.float32) * 0.1

    def loss_fn(p, noisy, gt):
        h = silu_gpu(conv2d_im2col(noisy, p["w0"], stride=1, padding=1, bias=p["b0"]))
        h = silu_gpu(conv2d_im2col(h, p["w1"], stride=1, padding=1, bias=p["b1"]))
        pool = nb.mean(h, axis=(1, 2))
        pred = pool @ p["dw"] + p["db"]
        pred_img = nb.reshape(pred, (1, RES, RES, 3))
        return nb.mean(square(pred_img - gt))

    losses = []
    for step in range(1, STEPS + 1):
        params = {k: to_dev(v) for k, v in p_np.items()}
        noisy = to_dev(noisy_np)
        gt = to_dev(gt_np)

        t0 = time.time()
        lv, grads = nb.value_and_grad(loss_fn, argnums=0)(params, noisy, gt)
        for k in grads:
            nb.realize_all(grads[k])
        nb.realize_all(lv)
        loss_f = float(to_cpu(lv))

        g_np = {k: to_cpu(v) for k, v in grads.items()}

        # AdamW on CPU (numpy)
        b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.01
        for k in p_np:
            opt_m[k] = b1 * opt_m[k] + (1 - b1) * g_np[k]
            opt_v[k] = b2 * opt_v[k] + (1 - b2) * g_np[k] ** 2
            mh = opt_m[k] / (1 - b1 ** (step + 1))
            vh = opt_v[k] / (1 - b2 ** (step + 1))
            p_np[k] -= LR * (mh / (np.sqrt(vh) + eps) + wd * p_np[k])

        dt = time.time() - t0
        losses.append(loss_f)
        guard(f"step {step}")
        print(f"  Step {step:2d}: loss={loss_f:.6f} ({dt*1000:.0f}ms)")

        del params, noisy, gt, lv, grads, g_np
        cleanup()

    all_ok = all(np.isfinite(l) for l in losses)
    print(f"\n  Losses: {[f'{l:.4f}' for l in losses]}")
    print(f"  All finite: {all_ok}")
    assert all_ok, "NaN in losses!"
    print("  Phase 5 PASSED")


def main():
    if accelerator_count() == 0:
        print("No GPU — aborting")
        return

    print(f"=== Custom Conv2d Backward Test (im2col + matmul) ===")
    print(f"Device: {dev()}")
    print(f"No conv_transpose, no cuDNN — pure nabla autodiff")
    guard("start")

    phase1()
    phase2()
    phase3()
    phase4()
    phase5()

    print("\n=== ALL PASSED — Multi-layer conv2d backward works on GPU without cuDNN ===")
    guard("final")


if __name__ == "__main__":
    main()
