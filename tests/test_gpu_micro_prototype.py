#!/usr/bin/env python3
"""Progressive GPU prototype — scale up from single ops to small model.

Each test is independent and logs VRAM. Stop at any point.
Runs while CPU training is going in tmux (separate process, separate VRAM).
"""

import time

import numpy as np

import nabla as nb
from max.driver import CPU, Accelerator, accelerator_count

# GPU-safe activations (exp-based, no CPU scalar in VJP)
from omen.kernels.activations import sigmoid_gpu, silu_gpu
from omen.kernels.conv2d import conv2d_safe


def _vram_mb():
    """VRAM used in MB via torch."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0


def _gpu():
    if accelerator_count() == 0:
        raise RuntimeError("No GPU detected")
    return Accelerator()


def _t(arr, device=None):
    """numpy -> nabla tensor on device."""
    t = nb.Tensor.from_dlpack(arr.astype(np.float32))
    if device:
        return nb.ops.transfer_to(t, device)
    return t


def _to(tensor, device):
    return nb.ops.transfer_to(tensor, device)


# ── Test 01: Device detection ──────────────────────────────────
def test_01_device():
    gpu = _gpu()
    print(f"[01] GPU: {gpu}, accelerator_count={accelerator_count()}")


# ── Test 02: Tensor transfer ───────────────────────────────────
def test_02_transfer():
    gpu = _gpu()
    x = _t(np.random.randn(4, 4), gpu)
    print(f"[02] Tensor on {x.device}, shape={x.shape}")
    assert "gpu" in str(x.device).lower() or "accelerator" in str(x.device).lower()


# ── Test 03: GPU matmul ────────────────────────────────────────
def test_03_matmul():
    gpu = _gpu()
    a = _t(np.random.randn(64, 64), gpu)
    b = _t(np.random.randn(64, 64), gpu)
    c = a @ b
    nb.realize_all(c)
    result = _to(c, CPU()).to_numpy()
    print(f"[03] GPU matmul result shape={result.shape}, sum={result.sum():.4f}")


# ── Test 04: GPU forward+backward (tiny MLP) ───────────────────
def test_04_fwd_bwd():
    gpu = _gpu()
    w = _t(np.random.randn(32, 16) * 0.01, gpu)
    x = _t(np.random.randn(4, 32), gpu)

    def loss_fn(w, x):
        h = silu_gpu(x @ w)
        return (h * h).sum()

    val, grads = nb.value_and_grad(loss_fn, argnums=0)(w, x)
    nb.realize_all(val, grads)
    v = _to(val, CPU()).to_numpy()
    g = _to(grads, CPU()).to_numpy()
    has_nan = np.isnan(g).any()
    print(f"[04] GPU fwd+bwd: loss={v:.4f}, grad_nan={has_nan}, vram={_vram_mb():.0f}MB")


# ── Test 05: GPU conv2d forward+backward ───────────────────────
def test_05_conv2d():
    gpu = _gpu()
    x = _t(np.random.randn(1, 8, 8, 3), gpu)
    filt = _t(np.random.randn(3, 3, 3, 16) * 0.01, gpu)

    def loss_fn(filt, x):
        out = conv2d_safe(x, filt, padding=1)
        return (out * out).sum()

    val, grads = nb.value_and_grad(loss_fn, argnums=0)(filt, x)
    nb.realize_all(val, grads)
    v = _to(val, CPU()).to_numpy()
    g = _to(grads, CPU()).to_numpy()
    has_nan = np.isnan(g).any()
    print(f"[05] GPU conv2d fwd+bwd: loss={v:.4f}, grad_nan={has_nan}, vram={_vram_mb():.0f}MB")


# ── Test 06: @nb.compile on GPU (tiny) ─────────────────────────
def test_06_compile_tiny():
    gpu = _gpu()
    w = _t(np.random.randn(16, 8) * 0.01, gpu)
    x = _t(np.random.randn(2, 16), gpu)

    @nb.compile
    def compiled_fn(w, x):
        return silu_gpu(x @ w).sum()

    t0 = time.time()
    result = compiled_fn(w, x)
    nb.realize_all(result)
    dt = time.time() - t0
    v = _to(result, CPU()).to_numpy()
    print(f"[06] @nb.compile GPU (tiny): val={v:.4f}, compile={dt:.1f}s, vram={_vram_mb():.0f}MB")

    # Cached run
    w2 = _t(np.random.randn(16, 8) * 0.01, gpu)
    x2 = _t(np.random.randn(2, 16), gpu)
    t0 = time.time()
    result2 = compiled_fn(w2, x2)
    nb.realize_all(result2)
    dt2 = time.time() - t0
    print(f"[06] Cached run: {dt2*1000:.1f}ms")


# ── Test 07: @nb.compile GPU with value_and_grad ──────────────
def test_07_compile_vjp():
    gpu = _gpu()
    w = _t(np.random.randn(32, 16) * 0.01, gpu)
    x = _t(np.random.randn(4, 32), gpu)

    @nb.compile
    def compiled_loss(w, x):
        h = silu_gpu(x @ w)
        return (h * h).sum()

    t0 = time.time()
    val = compiled_loss(w, x)
    nb.realize_all(val)
    dt = time.time() - t0
    v = _to(val, CPU()).to_numpy()
    print(f"[07] @nb.compile GPU fwd: val={v:.4f}, compile={dt:.1f}s, vram={_vram_mb():.0f}MB")

    # value_and_grad inside compile
    @nb.compile
    def compiled_fwd_bwd(w, x):
        return nb.value_and_grad(lambda w, x: silu_gpu(x @ w).sum(), argnums=0)(w, x)

    t0 = time.time()
    val, grads = compiled_fwd_bwd(w, x)
    nb.realize_all(val, grads)
    dt = time.time() - t0
    g = _to(grads, CPU()).to_numpy()
    has_nan = np.isnan(g).any()
    print(f"[07] @nb.compile GPU fwd+bwd: val={_to(val, CPU()).to_numpy():.4f}, "
          f"grad_nan={has_nan}, compile={dt:.1f}s, vram={_vram_mb():.0f}MB")


# ── Test 08: Scale up — 256x256 conv model ─────────────────────
def test_08_scale_256():
    gpu = _gpu()
    # Tiny encoder: conv 3→8→16, pool, linear 16→4
    f1 = _t(np.random.randn(3, 3, 3, 8) * 0.01, gpu)
    f2 = _t(np.random.randn(3, 3, 8, 16) * 0.01, gpu)
    pw = _t(np.random.randn(16, 4) * 0.01, gpu)
    pb = _t(np.zeros((1, 4)), gpu)
    x = _t(np.random.randn(1, 64, 64, 3), gpu)

    def model(params, x):
        f1, f2, pw, pb = params
        h = silu_gpu(conv2d_safe(x, f1, stride=2, padding=1))
        h = silu_gpu(conv2d_safe(h, f2, stride=2, padding=1))
        h = h.mean(axis=(1, 2))
        return (h @ pw + pb).sum()

    params = (f1, f2, pw, pb)
    t0 = time.time()
    val, grads = nb.value_and_grad(model, argnums=0)(params, x)
    nb.realize_all(val, grads)
    dt = time.time() - t0
    v = _to(val, CPU()).to_numpy()
    g0 = _to(grads[0], CPU()).to_numpy()
    has_nan = np.isnan(g0).any()
    print(f"[08] 64x64 tiny model fwd+bwd: val={v:.4f}, grad_nan={has_nan}, "
          f"time={dt:.1f}s, vram={_vram_mb():.0f}MB")


# ── Test 09: @nb.compile on scaled model ───────────────────────
def test_09_compile_scaled():
    gpu = _gpu()
    f1 = _t(np.random.randn(3, 3, 3, 8) * 0.01, gpu)
    f2 = _t(np.random.randn(3, 3, 8, 16) * 0.01, gpu)
    pw = _t(np.random.randn(16, 4) * 0.01, gpu)
    pb = _t(np.zeros((1, 4)), gpu)
    x = _t(np.random.randn(1, 64, 64, 3), gpu)

    @nb.compile
    def compiled_model(params, x):
        f1, f2, pw, pb = params
        h = silu_gpu(conv2d_safe(x, f1, stride=2, padding=1))
        h = silu_gpu(conv2d_safe(h, f2, stride=2, padding=1))
        h = h.mean(axis=(1, 2))
        return nb.value_and_grad(
            lambda params, x: (params[2] @ params[3] + params[3]).sum(),
            argnums=0,
        )(params, x)

    print(f"[09] Compiling 64x64 model on GPU (may take 30-120s)...")
    t0 = time.time()
    params = (f1, f2, pw, pb)
    val, grads = compiled_model(params, x)
    nb.realize_all(val, grads)
    dt = time.time() - t0
    print(f"[09] Compiled: {dt:.1f}s, vram={_vram_mb():.0f}MB")

    # Cached
    x2 = _t(np.random.randn(1, 64, 64, 3), gpu)
    t0 = time.time()
    val2, _ = compiled_model(params, x2)
    nb.realize_all(val2)
    dt2 = time.time() - t0
    print(f"[09] Cached run: {dt2*1000:.1f}ms, vram={_vram_mb():.0f}MB")


# ── Runner ─────────────────────────────────────────────────────
TESTS = [
    ("Device detection", test_01_device),
    ("Tensor transfer", test_02_transfer),
    ("GPU matmul", test_03_matmul),
    ("GPU fwd+bwd (tiny MLP)", test_04_fwd_bwd),
    ("GPU conv2d fwd+bwd", test_05_conv2d),
    ("@nb.compile tiny", test_06_compile_tiny),
    ("@nb.compile vjp", test_07_compile_vjp),
    ("Scale 64x64 model", test_08_scale_256),
    ("@nb.compile scaled", test_09_compile_scaled),
]


def main():
    if accelerator_count() == 0:
        print("No GPU — aborting")
        return

    print(f"GPU micro-prototype: {len(TESTS)} tests")
    print(f"Initial VRAM: {_vram_mb():.0f}MB")
    print("=" * 60)

    for i, (name, fn) in enumerate(TESTS):
        print(f"\n--- Test {i+1:02d}: {name} ---")
        try:
            t0 = time.time()
            fn()
            dt = time.time() - t0
            print(f"    PASS ({dt:.1f}s) VRAM={_vram_mb():.0f}MB")
        except Exception as e:
            print(f"    FAIL: {e}")
            # Don't continue if basic ops fail
            if i < 3:
                print("Basic GPU ops failed — stopping")
                break
            print("Continuing to next test...")
        time.sleep(2)  # Let VRAM settle between tests

    print(f"\nFinal VRAM: {_vram_mb():.0f}MB")


if __name__ == "__main__":
    main()
