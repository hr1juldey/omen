#!/usr/bin/env python3
"""Progressive GPU prototype — RAM-safe with 6GB process guard.

Each test checks RSS < 6GB before running, cleans up after.
Aborts if process RSS > 6GB or system RAM < 4GB available.
"""

import gc
import os
import sys
import time

import numpy as np

import nabla as nb
from max.driver import CPU, Accelerator, accelerator_count

from omen.kernels.activations import silu_gpu
from omen.kernels.conv2d import conv2d_safe

LIMIT_MB = 8 * 1024  # 8GB process RSS limit


def _rss():
    """Process RSS in MB. Returns 0 on error."""
    try:
        text = open(f"/proc/{os.getpid()}/status").read()
        match = next(ln for ln in text.splitlines() if ln.startswith("VmRSS:"))
        return int(match.split()[1]) // 1024
    except Exception:
        return 0


def _avail():
    """System available RAM in MB. Returns 0 on error."""
    try:
        text = open("/proc/meminfo").read()
        match = next(ln for ln in text.splitlines() if ln.startswith("MemAvailable:"))
        return int(match.split()[1]) // 1024
    except Exception:
        return 0


def _vram():
    """GPU VRAM allocated in MB. Returns 0 on error."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.memory_allocated() // 1024 // 1024
    except Exception:
        return 0


def guard(label=""):
    rss = _rss()
    avail = _avail()
    if rss > LIMIT_MB:
        print(f"KILL: RSS={rss}MB > {LIMIT_MB}MB {label}")
        sys.exit(99)
    if avail < 4000:
        print(f"KILL: sys_avail={avail}MB < 4000MB {label}")
        sys.exit(98)
    print(f"    [guard] RSS={rss}MB avail={avail}MB VRAM={_vram()}MB")
    return rss


GPU = None  # Lazy init


def gpu():
    global GPU
    if GPU is None:
        GPU = Accelerator()
    return GPU


def t(arr):
    return nb.ops.transfer_to(nb.Tensor.from_dlpack(arr.astype(np.float32)), gpu())


def cpu(tensor):
    return nb.ops.transfer_to(tensor, CPU()).to_numpy()


def clean():
    gc.collect()


# ── Model (flat, no nesting) ──────────────────────────────────
def tiny_conv_loss(params, x):
    f1, f2, pw, pb = params
    h = silu_gpu(conv2d_safe(x, f1, stride=2, padding=1))
    h = silu_gpu(conv2d_safe(h, f2, stride=2, padding=1))
    h = h.mean(axis=(1, 2))
    return (h @ pw + pb).sum()


@nb.compile
def compiled_conv_vjp(params, x):
    return nb.value_and_grad(tiny_conv_loss, argnums=0)(params, x)


@nb.compile
def compiled_silu_sum(w, x):
    return silu_gpu(x @ w).sum()


@nb.compile
def compiled_silu_vjp(w, x):
    return nb.value_and_grad(lambda w, x: silu_gpu(x @ w).sum(), argnums=0)(w, x)


# ── Tests (flat, one function each) ───────────────────────────
def test_01_device():
    print(f"    GPU={gpu()}, count={accelerator_count()}")


def test_02_transfer():
    x = t(np.random.randn(4, 4))
    print(f"    device={x.device}, sum={cpu(x).sum():.4f}")
    del x
    clean()


def test_03_matmul():
    a = t(np.random.randn(64, 64))
    b = t(np.random.randn(64, 64))
    c = a @ b
    nb.realize_all(c)
    print(f"    sum={cpu(c).sum():.4f}")
    del a, b, c
    clean()


def test_04_fwd_bwd():
    w = t(np.random.randn(32, 16) * 0.01)
    x = t(np.random.randn(4, 32))
    val, grads = nb.value_and_grad(
        lambda w, x: silu_gpu(x @ w).sum(), argnums=0
    )(w, x)
    nb.realize_all(val, grads)
    nan = np.isnan(cpu(grads)).any()
    print(f"    loss={cpu(val):.4f}, grad_nan={nan}")
    del w, x, val, grads
    clean()


def test_05_conv2d():
    x = t(np.random.randn(1, 8, 8, 3))
    f = t(np.random.randn(3, 3, 3, 16) * 0.01)
    val, grads = nb.value_and_grad(
        lambda f, x: conv2d_safe(x, f, padding=1).sum(), argnums=0
    )(f, x)
    nb.realize_all(val, grads)
    nan = np.isnan(cpu(grads)).any()
    print(f"    loss={cpu(val):.4f}, grad_nan={nan}")
    del x, f, val, grads
    clean()


def test_06_compile_tiny():
    w = t(np.random.randn(16, 8) * 0.01)
    x = t(np.random.randn(2, 16))
    t0 = time.time()
    r = compiled_silu_sum(w, x)
    nb.realize_all(r)
    dt = time.time() - t0
    print(f"    compile={dt:.1f}s, val={cpu(r):.4f}")
    # Cached
    w2 = t(np.random.randn(16, 8) * 0.01)
    x2 = t(np.random.randn(2, 16))
    t0 = time.time()
    r2 = compiled_silu_sum(w2, x2)
    nb.realize_all(r2)
    print(f"    cached={((time.time()-t0)*1000):.1f}ms")
    del w, x, w2, x2, r, r2
    clean()


def test_07_compile_vjp():
    w = t(np.random.randn(32, 16) * 0.01)
    x = t(np.random.randn(4, 32))
    t0 = time.time()
    val, grads = compiled_silu_vjp(w, x)
    nb.realize_all(val, grads)
    dt = time.time() - t0
    nan = np.isnan(cpu(grads)).any()
    print(f"    compile={dt:.1f}s, loss={cpu(val):.4f}, grad_nan={nan}")
    del w, x, val, grads
    clean()


def _make_scale_test(res, compiled=False):
    """Factory: returns a test function for given resolution."""
    def test():
        f1 = t(np.random.randn(3, 3, 3, 8) * 0.01)
        f2 = t(np.random.randn(3, 3, 8, 16) * 0.01)
        pw = t(np.random.randn(16, 4) * 0.01)
        pb = t(np.zeros((1, 4)))
        x = t(np.random.randn(1, res, res, 3))
        params = (f1, f2, pw, pb)
        t0 = time.time()
        if compiled:
            val, grads = compiled_conv_vjp(params, x)
        else:
            val, grads = nb.value_and_grad(tiny_conv_loss, argnums=0)(params, x)
        nb.realize_all(val, grads)
        dt = time.time() - t0
        nan = np.isnan(cpu(grads[0])).any()
        tag = "compiled" if compiled else "eager"
        print(f"    {tag} {res}x{res}: loss={cpu(val):.4f}, nan={nan}, {dt:.1f}s")
        del f1, f2, pw, pb, x, params, val, grads
        clean()
    return test


# ── Build test list (flat, no nesting in runner) ──────────────
TESTS = [
    ("01 Device", test_01_device),
    ("02 Transfer", test_02_transfer),
    ("03 Matmul", test_03_matmul),
    ("04 Fwd+Bwd", test_04_fwd_bwd),
    ("05 Conv2d", test_05_conv2d),
    ("06 Compile tiny", test_06_compile_tiny),
    ("07 Compile vjp", test_07_compile_vjp),
]
# Scale up only if RAM allows
for res in [8, 16, 32, 64]:
    TESTS.append((f"Eager {res}x{res}", _make_scale_test(res)))


def main():
    if accelerator_count() == 0:
        print("No GPU")
        return
    print(f"GPU prototype: {len(TESTS)} tests, RSS limit={LIMIT_MB}MB")
    print(f"Initial: RSS={_rss()}MB avail={_avail()}MB")
    print("=" * 60)

    for i, (name, fn) in enumerate(TESTS):
        rss = guard(f"test {i+1}")
        print(f"\n--- {name} [RSS={rss}MB] ---")
        try:
            t0 = time.time()
            fn()
            dt = time.time() - t0
            rss_after = guard("done")
            print(f"    PASS ({dt:.1f}s)")
            if rss_after > LIMIT_MB:
                print("    OVER LIMIT — STOPPING")
                break
        except SystemExit:
            raise
        except Exception as e:
            print(f"    FAIL: {e}")
            if i < 3:
                break

    print(f"\nFinal: RSS={_rss()}MB avail={_avail()}MB VRAM={_vram()}MB")


if __name__ == "__main__":
    main()
