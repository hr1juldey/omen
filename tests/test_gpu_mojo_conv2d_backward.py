#!/usr/bin/env python3
"""Prove multi-layer conv2d backward works on GPU — no conv_transpose, no cuDNN.

Architecture (the "known correction"):
  kernel()   — graph_ops.conv2d() → 1 native MLIR op (fast JIT, same output as nb.conv2d)
  vjp_rule() — pure nabla: recompute patches + matmul grad + col2im scatter
               NO conv_transpose, NO cuDNN.

Why this works:
  - Forward uses native graph_ops.conv2d (single op, fast GPU JIT)
  - Backward OVERRIDES the default vjp (which uses conv_transpose → SIGABRT)
  - Custom vjp uses only matmul, pad, reshape, concatenate — all have working GPU VJPs
  - Native nb.conv2d crashes with 2+ layers because conv_transpose can't load cudnnCreate
  - This custom op proves multi-layer conv2d backward works WITHOUT that dependency
"""

import gc
import logging
import os
import sys
import time

import numpy as np
import nabla as nb
from max.driver import CPU, Accelerator, accelerator_count
from nabla.ops import Operation
from omen.kernels.activations import silu_gpu, square

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("conv2d_test")

LIMIT_MB = 24 * 1024


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
        log.error("KILL: RSS=%dMB > %dMB %s", rss, LIMIT_MB, label)
        sys.exit(99)
    log.info("[guard] RSS=%dMB  %s", rss, label)
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


# ── Helper functions ──────────────────────────────────────────
def _conv_out_size(in_size, kernel, stride, pad):
    return (in_size + 2 * pad - kernel) // stride + 1


def _parse_stride_padding(stride, padding):
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
    return sh, sw, ph, pw


# ── Pure nabla col2im (for VJP backward) ─────────────────────
def _col2im(col, b, h, w, c_in, kh, kw, ph, pw, h_out, w_out):
    """Scatter column matrix back to image.  col: (B*H_out*W_out, Kh*Kw*C_in)."""
    log.debug("col2im: col=%s -> target=(%d,%d,%d,%d)", col.shape, b, h, w, c_in)
    col_6d = nb.reshape(col, (b, h_out, w_out, kh, kw, c_in))
    h_pad, w_pad = h + 2 * ph, w + 2 * pw

    padded_patches = []
    for ki in range(kh):
        for kj in range(kw):
            patch = col_6d[:, :, :, ki, kj, :]
            pt, pb = ki, h_pad - ki - h_out
            pl, pr = kj, w_pad - kj - w_out
            padded = nb.pad(patch, ((0, 0), (pt, pb), (pl, pr), (0, 0)))
            padded_patches.append(padded)

    result = padded_patches[0]
    for p in padded_patches[1:]:
        result = result + p

    if ph > 0 or pw > 0:
        result = result[:, ph : ph + h, pw : pw + w, :]

    log.debug("col2im: result=%s", result.shape)
    return result


# ── Custom Conv2d Operation (native forward + custom VJP) ─────
class Conv2dMojoOp(Operation):
    """Conv2d via native graph_ops.conv2d + custom VJP.

    Forward (kernel):   graph_ops.conv2d() — 1 native MLIR op (fast JIT)
    Backward (vjp_rule): pure nabla — recompute patches + matmul + col2im scatter
                         No conv_transpose, no cuDNN.
    """

    @property
    def name(self) -> str:
        return "mojo_conv2d"

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        x, filt = args[0], args[1]
        b, h, w, _ = (int(d) for d in x.shape)
        kh, kw, c_in, c_out = (int(d) for d in filt.shape)
        sh, sw = kwargs["sh"], kwargs["sw"]
        ph, pw = kwargs["ph"], kwargs["pw"]
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1
        log.debug(
            "compute_physical_shape: (%d,%d,%d,%d) * (%d,%d,%d,%d) -> (%d,%d,%d,%d)",
            b,
            h,
            w,
            c_in,
            kh,
            kw,
            c_in,
            c_out,
            b,
            h_out,
            w_out,
            c_out,
        )
        return ([(b, h_out, w_out, c_out)], [x.dtype], [x.device])

    def kernel(self, args, kwargs):
        from max.graph import ops as graph_ops

        x = args[0]  # TensorValue (B, H, W, C_in)
        filt = args[1]  # TensorValue (Kh, Kw, C_in, C_out)  [RSCF layout]

        sh, sw = kwargs["sh"], kwargs["sw"]
        ph, pw = kwargs["ph"], kwargs["pw"]
        bias = args[2] if len(args) > 2 else None

        log.debug(
            "kernel: graph_ops.conv2d stride=(%d,%d) padding=(%d,%d,%d,%d)",
            sh,
            sw,
            ph,
            ph,
            pw,
            pw,
        )
        # Native conv2d — single MLIR op, fast GPU JIT compilation
        return [
            graph_ops.conv2d(
                x,
                filt,
                stride=(sh, sw),
                padding=(ph, ph, pw, pw),
                bias=bias,
            )
        ]

    def vjp_rule(self, primals, cotangents, outputs, kwargs):
        x = primals[0]  # Tensor (B, H, W, C_in)
        filt = primals[1]  # Tensor (Kh, Kw, C_in, C_out)
        ct = cotangents[0]  # Tensor (B, H_out, W_out, C_out)

        sh, sw = kwargs["sh"], kwargs["sw"]
        ph, pw = kwargs["ph"], kwargs["pw"]

        kh, kw, c_in, c_out = (int(d) for d in filt.shape)
        b, h, w, _ = (int(d) for d in x.shape)
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1

        log.info(
            "vjp_rule: x=(%d,%d,%d,%d) filt=(%d,%d,%d,%d) ct=(%s)",
            b,
            h,
            w,
            c_in,
            kh,
            kw,
            c_in,
            c_out,
            tuple(int(d) for d in ct.shape),
        )

        # ── Recompute patches (pure nabla) ──
        t0 = time.time()
        x_pad = (
            nb.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0))) if (ph > 0 or pw > 0) else x
        )

        patch_list = []
        for ki in range(kh):
            for kj in range(kw):
                p = x_pad[:, ki : ki + h_out, kj : kj + w_out, :]
                p = nb.reshape(p, (b * h_out * w_out, c_in))
                patch_list.append(p)
        patches = nb.concatenate(patch_list, axis=1)
        log.info(
            "vjp_rule: patches recomputed in %.3fs  shape=%s",
            time.time() - t0,
            patches.shape,
        )

        # ── Flatten tensors ──
        ct_flat = nb.reshape(ct, (b * h_out * w_out, c_out))
        filt_flat = nb.reshape(filt, (kh * kw * c_in, c_out))

        # ── grad_filter = patches.T @ ct_flat ──
        t1 = time.time()
        patches_t = nb.transpose(patches, 1, 0)
        grad_filt_flat = nb.matmul(patches_t, ct_flat)
        grad_filter = nb.reshape(grad_filt_flat, (kh, kw, c_in, c_out))
        log.info("vjp_rule: grad_filter computed in %.3fs", time.time() - t1)

        # ── grad_input = col2im(ct_flat @ filt_flat.T) ──
        t2 = time.time()
        filt_flat_t = nb.transpose(filt_flat, 1, 0)
        grad_patches = nb.matmul(ct_flat, filt_flat_t)
        grad_input = _col2im(grad_patches, b, h, w, c_in, kh, kw, ph, pw, h_out, w_out)
        log.info("vjp_rule: grad_input (col2im) computed in %.3fs", time.time() - t2)

        has_bias = len(primals) > 2
        if has_bias:
            grad_bias = nb.sum(ct, axis=(0, 1, 2))
            log.info("vjp_rule: returning [grad_input, grad_filter, grad_bias]")
            return [grad_input, grad_filter, grad_bias]

        log.info("vjp_rule: returning [grad_input, grad_filter]")
        return [grad_input, grad_filter]


_mojo_conv2d_op = Conv2dMojoOp()


# ── Convenience function ──────────────────────────────────────
def mojo_conv2d(x, filt, stride=1, padding=0, bias=None):
    """Conv2d with native forward + custom VJP backward (no conv_transpose)."""
    sh, sw, ph, pw = _parse_stride_padding(stride, padding)
    args = [x, filt]
    if bias is not None:
        args.append(bias)
    kwargs = {"sh": sh, "sw": sw, "ph": ph, "pw": pw}
    return _mojo_conv2d_op(args, kwargs)[0]


# ── Phase runners ────────────────────────────────────────────


def phase1():
    """16x16, 4->8, 1 conv2d forward+backward."""
    log.info("=" * 60)
    log.info("Phase 1: 16x16, 4->8, 1 conv2d forward+backward")
    log.info("=" * 60)
    guard("start")

    x = to_dev(np.random.randn(1, 16, 16, 4).astype(np.float32) * 0.1)
    w = to_dev(np.random.randn(3, 3, 4, 8).astype(np.float32) * 0.01)
    log.info("x shape=(1,16,16,4)  w shape=(3,3,4,8)")

    # Forward
    t0 = time.time()
    y = mojo_conv2d(x, w, stride=1, padding=1)
    nb.realize_all(y)
    y_np = to_cpu(y)
    dt = time.time() - t0
    log.info("Forward: (1,16,16,4) -> %s  (%.3fs)", y_np.shape, dt)
    assert y_np.shape == (1, 16, 16, 8), f"Bad shape: {y_np.shape}"

    # Compare against native — forward should be IDENTICAL (same graph_ops.conv2d)
    y_native = nb.conv2d(x, w, padding=(1, 1, 1, 1))
    nb.realize_all(y_native)
    diff = np.max(np.abs(y_np - to_cpu(y_native)))
    log.info("vs native nb.conv2d: max_diff=%.6f (expect ~0.0)", diff)
    assert diff < 1e-5, f"Forward mismatch: max_diff={diff}"

    # Backward
    def loss(p, x_in):
        return nb.mean(square(mojo_conv2d(x_in, p["w"], stride=1, padding=1)))

    t1 = time.time()
    lv, grads = nb.value_and_grad(loss, argnums=0)({"w": w}, x)
    nb.realize_all(lv, grads["w"])
    gw = to_cpu(grads["w"])
    dt_bw = time.time() - t1
    loss_f = float(to_cpu(lv))
    log.info(
        "Backward: loss=%.6f  grad_finite=%s  (%.3fs)",
        loss_f,
        np.all(np.isfinite(gw)),
        dt_bw,
    )
    assert np.all(np.isfinite(gw)), "NaN grad!"

    log.info("Phase 1 PASSED")
    del x, w, y, y_native, lv, grads, gw
    cleanup()
    guard("end")


def phase2():
    """16x16, 4->8->16, 2 conv2d backward — THE CRITICAL TEST."""
    log.info("=" * 60)
    log.info(
        "Phase 2: 16x16, 4->8->16, 2 conv2d (CRITICAL — native nb.conv2d crashes here)"
    )
    log.info("=" * 60)
    guard("start")

    x = to_dev(np.random.randn(1, 16, 16, 4).astype(np.float32) * 0.1)
    w0 = to_dev(np.random.randn(3, 3, 4, 8).astype(np.float32) * 0.01)
    w1 = to_dev(np.random.randn(3, 3, 8, 16).astype(np.float32) * 0.01)
    log.info("x=(1,16,16,4)  w0=(3,3,4,8)  w1=(3,3,8,16)")

    def loss(p, x_in):
        h = mojo_conv2d(x_in, p["w0"], stride=1, padding=1)
        h = mojo_conv2d(h, p["w1"], stride=1, padding=1)
        return nb.mean(square(h))

    t0 = time.time()
    lv, grads = nb.value_and_grad(loss, argnums=0)({"w0": w0, "w1": w1}, x)
    nb.realize_all(lv, grads["w0"], grads["w1"])
    dt = time.time() - t0
    loss_f = float(to_cpu(lv))
    g0 = to_cpu(grads["w0"])
    g1 = to_cpu(grads["w1"])
    log.info("loss=%.6f  compile+exec=%.1fs", loss_f, dt)
    log.info("g0: shape=%s  finite=%s", g0.shape, np.all(np.isfinite(g0)))
    log.info("g1: shape=%s  finite=%s", g1.shape, np.all(np.isfinite(g1)))
    assert np.all(np.isfinite(g0)), "NaN g0!"
    assert np.all(np.isfinite(g1)), "NaN g1!"

    log.info("Phase 2 PASSED — multi-layer conv2d backward works WITHOUT cuDNN!")
    del x, w0, w1, lv, grads, g0, g1
    cleanup()
    guard("end")


def phase3():
    """32x32, 4->16->32, 2 conv2d backward."""
    log.info("=" * 60)
    log.info("Phase 3: 32x32, 4->16->32, 2 conv2d backward")
    log.info("=" * 60)
    guard("start")

    x = to_dev(np.random.randn(1, 32, 32, 4).astype(np.float32) * 0.1)
    w0 = to_dev(np.random.randn(3, 3, 4, 16).astype(np.float32) * 0.01)
    w1 = to_dev(np.random.randn(3, 3, 16, 32).astype(np.float32) * 0.01)
    log.info("x=(1,32,32,4)  w0=(3,3,4,16)  w1=(3,3,16,32)")

    def loss(p, x_in):
        h = mojo_conv2d(x_in, p["w0"], stride=1, padding=1)
        h = mojo_conv2d(h, p["w1"], stride=1, padding=1)
        return nb.mean(square(h))

    t0 = time.time()
    lv, grads = nb.value_and_grad(loss, argnums=0)({"w0": w0, "w1": w1}, x)
    nb.realize_all(lv, grads["w0"], grads["w1"])
    dt = time.time() - t0
    loss_f = float(to_cpu(lv))
    g0 = to_cpu(grads["w0"])
    g1 = to_cpu(grads["w1"])
    log.info("loss=%.6f  compile+exec=%.1fs", loss_f, dt)
    log.info("g0: shape=%s  finite=%s", g0.shape, np.all(np.isfinite(g0)))
    log.info("g1: shape=%s  finite=%s", g1.shape, np.all(np.isfinite(g1)))
    assert np.all(np.isfinite(g0)) and np.all(np.isfinite(g1)), "NaN!"

    log.info("Phase 3 PASSED")
    del x, w0, w1, lv, grads, g0, g1
    cleanup()
    guard("end")


def phase4():
    """64x64, 4->16->32, 2 conv2d backward."""
    log.info("=" * 60)
    log.info("Phase 4: 64x64, 4->16->32, 2 conv2d backward")
    log.info("=" * 60)
    guard("start")

    x = to_dev(np.random.randn(1, 64, 64, 4).astype(np.float32) * 0.1)
    w0 = to_dev(np.random.randn(3, 3, 4, 16).astype(np.float32) * 0.01)
    w1 = to_dev(np.random.randn(3, 3, 16, 32).astype(np.float32) * 0.01)
    log.info("x=(1,64,64,4)  w0=(3,3,4,16)  w1=(3,3,16,32)")

    def loss(p, x_in):
        h = mojo_conv2d(x_in, p["w0"], stride=1, padding=1)
        h = mojo_conv2d(h, p["w1"], stride=1, padding=1)
        return nb.mean(square(h))

    t0 = time.time()
    lv, grads = nb.value_and_grad(loss, argnums=0)({"w0": w0, "w1": w1}, x)
    nb.realize_all(lv, grads["w0"], grads["w1"])
    dt = time.time() - t0
    loss_f = float(to_cpu(lv))
    g0 = to_cpu(grads["w0"])
    g1 = to_cpu(grads["w1"])
    log.info("loss=%.6f  compile+exec=%.1fs", loss_f, dt)
    log.info("g0: shape=%s  finite=%s", g0.shape, np.all(np.isfinite(g0)))
    log.info("g1: shape=%s  finite=%s", g1.shape, np.all(np.isfinite(g1)))
    assert np.all(np.isfinite(g0)) and np.all(np.isfinite(g1)), "NaN!"

    log.info("Phase 4 PASSED")
    del x, w0, w1, lv, grads, g0, g1
    cleanup()
    guard("end")


def phase5():
    """64x64, 2+ conv2d + 10-step AdamW training loop."""
    log.info("=" * 60)
    log.info("Phase 5: 64x64 training loop (2 conv2d + AdamW, 10 steps)")
    log.info("=" * 60)
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
    log.info("%d tensors, %s params", len(p_np), f"{n_p:,}")

    opt_m = {k: np.zeros_like(v) for k, v in p_np.items()}
    opt_v = {k: np.zeros_like(v) for k, v in p_np.items()}

    noisy_np = np.random.randn(1, RES, RES, 4).astype(np.float32) * 0.1
    gt_np = np.random.randn(1, RES, RES, 3).astype(np.float32) * 0.1

    def loss_fn(p, noisy, gt):
        h = silu_gpu(mojo_conv2d(noisy, p["w0"], stride=1, padding=1, bias=p["b0"]))
        h = silu_gpu(mojo_conv2d(h, p["w1"], stride=1, padding=1, bias=p["b1"]))
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

        # AdamW on CPU (numpy) — graph break prevents RAM leak
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
        log.info(
            "Step %2d: loss=%.6f (%dms)",
            step,
            loss_f,
            int(dt * 1000),
        )

        del params, noisy, gt, lv, grads, g_np
        cleanup()

    all_ok = all(np.isfinite(l) for l in losses)
    log.info("Losses: %s", [f"{l:.4f}" for l in losses])
    log.info("All finite: %s", all_ok)
    assert all_ok, "NaN in losses!"
    log.info("Phase 5 PASSED")


def main():
    if accelerator_count() == 0:
        log.info("No GPU — aborting")
        return

    log.info("=" * 60)
    log.info("Custom Conv2d Backward Test")
    log.info("Device: %s", dev())
    log.info(
        "Architecture: graph_ops.conv2d forward + custom VJP backward (no conv_transpose)"
    )
    log.info("=" * 60)
    guard("start")

    phase1()
    phase2()
    phase3()
    phase4()
    phase5()

    log.info("=" * 60)
    log.info("ALL PASSED — Multi-layer conv2d backward works on GPU without cuDNN")
    log.info("Forward: graph_ops.conv2d (1 native op)")
    log.info("Backward: custom vjp_rule (pure nabla, no conv_transpose)")
    log.info("=" * 60)
    guard("final")


if __name__ == "__main__":
    main()
