"""Systematic diagnosis: find which Mojo activation combination triggers SIGSEGV.

Tests all combinations at 256x256 with 16D tensors, pure-nabla baseline first.
"""
import sys
import time
import traceback

import nabla as nb
import numpy as np
from nabla.ops import conv2d

from omen.kernels.activations import sigmoid_gpu, silu_gpu, square
from omen.kernels.activations_gpu import sigmoid_mojo, silu_mojo

D = 16  # channel dim
RES = 16  # start small, scale up after combos pass
np.random.seed(42)


def _w(*shape):
    return nb.Tensor.from_dlpack(np.random.randn(*shape).astype(np.float32) * 0.01)


def _b(n):
    return nb.Tensor.from_dlpack(np.zeros(n).astype(np.float32))


def run_test(name, loss_fn, args, argnums=0, timeout_label=""):
    """Run value_and_grad on loss_fn, report pass/fail/crash."""
    t0 = time.time()
    try:
        val, grads = nb.value_and_grad(loss_fn, argnums=argnums)(*args)
        dt = time.time() - t0
        v = float(val.to_numpy())
        print(f"  PASS  {name:40s}  val={v:.4f}  time={dt:.1f}s")
        return True
    except Exception as e:
        dt = time.time() - t0
        print(f"  FAIL  {name:40s}  err={e}  time={dt:.1f}s")
        traceback.print_exc()
        return False


print("=" * 70)
print(f"Mojo Activation Combination Diagnosis — RES={RES}, D={D}")
print("=" * 70)

# ── Test 0: Pure-nabla baseline ────────────────────────────────
print("\n[0] Pure-nabla baseline (silu_gpu + sigmoid_gpu)")

rgba = _w(1, RES, RES, 4)
scene_f = _w(1, 18)
gt = _w(1, D)
f1, b1 = _w(3, 3, 4, D), _b(D)
f2, b2 = _w(3, 3, D, D), _b(D)
pw, pb = _w(D, D), _b(D)
gw, gb = _w(D, D), _b(D)
se_w, se_b = _w(18, D), _b(D)


def loss_pure(rgba, scene_f, gt):
    x = silu_gpu(conv2d(rgba, f1, stride=2, padding=1, bias=b1))
    x = silu_gpu(conv2d(x, f2, stride=2, padding=1, bias=b2))
    rl = x.mean(axis=(1, 2)) @ pw + pb
    sl = silu_gpu(scene_f @ se_w + se_b)
    g = sigmoid_gpu(rl @ gw + gb)
    fused = rl + g * sl
    return nb.mean(square(fused - gt))


run_test("pure-nabla full model", loss_pure, [rgba, scene_f, gt])

# ── Test 1: silu_mojo only ─────────────────────────────────────
print("\n[1] silu_mojo only (sigmoid stays pure-nabla)")


def loss_silu_mojo(rgba, scene_f, gt):
    x = silu_mojo(conv2d(rgba, f1, stride=2, padding=1, bias=b1))
    x = silu_mojo(conv2d(x, f2, stride=2, padding=1, bias=b2))
    rl = x.mean(axis=(1, 2)) @ pw + pb
    sl = silu_mojo(scene_f @ se_w + se_b)
    g = sigmoid_gpu(rl @ gw + gb)
    fused = rl + g * sl
    return nb.mean(square(fused - gt))


run_test("silu_mojo only", loss_silu_mojo, [rgba, scene_f, gt])

# ── Test 2: sigmoid_mojo only ──────────────────────────────────
print("\n[2] sigmoid_mojo only (silu stays pure-nabla)")


def loss_sig_mojo(rgba, scene_f, gt):
    x = silu_gpu(conv2d(rgba, f1, stride=2, padding=1, bias=b1))
    x = silu_gpu(conv2d(x, f2, stride=2, padding=1, bias=b2))
    rl = x.mean(axis=(1, 2)) @ pw + pb
    sl = silu_gpu(scene_f @ se_w + se_b)
    g = sigmoid_mojo(rl @ gw + gb)
    fused = rl + g * sl
    return nb.mean(square(fused - gt))


run_test("sigmoid_mojo only", loss_sig_mojo, [rgba, scene_f, gt])

# ── Test 3: Both Mojo ──────────────────────────────────────────
print("\n[3] Both silu_mojo + sigmoid_mojo")


def loss_both_mojo(rgba, scene_f, gt):
    x = silu_mojo(conv2d(rgba, f1, stride=2, padding=1, bias=b1))
    x = silu_mojo(conv2d(x, f2, stride=2, padding=1, bias=b2))
    rl = x.mean(axis=(1, 2)) @ pw + pb
    sl = silu_mojo(scene_f @ se_w + se_b)
    g = sigmoid_mojo(rl @ gw + gb)
    fused = rl + g * sl
    return nb.mean(square(fused - gt))


run_test("both Mojo", loss_both_mojo, [rgba, scene_f, gt])

# ── Test 4: 3 conv2d layers with silu_mojo ─────────────────────
print("\n[4] 3 conv2d layers (mirrors render_encoder)")
f3, b3 = _w(3, 3, D, D), _b(D)
pw3, pb3 = _w(D, D), _b(D)


def loss_3conv(rgba, gt):
    x = silu_mojo(conv2d(rgba, f1, stride=2, padding=1, bias=b1))
    x = silu_mojo(conv2d(x, f2, stride=2, padding=1, bias=b2))
    x = silu_mojo(conv2d(x, f3, stride=2, padding=1, bias=b3))
    rl = x.mean(axis=(1, 2)) @ pw3 + pb3
    return nb.mean(square(rl - gt))


run_test("3 conv2d + silu_mojo", loss_3conv, [rgba, gt])

# ── Test 5: Grad w.r.t. params (not input) ─────────────────────
print("\n[5] Grad w.r.t. params (argnums=0 on params dict)")


def loss_params(p, rgba, scene_f, gt):
    x = silu_mojo(conv2d(rgba, p["f1"], stride=2, padding=1, bias=p["b1"]))
    x = silu_mojo(conv2d(x, p["f2"], stride=2, padding=1, bias=p["b2"]))
    rl = x.mean(axis=(1, 2)) @ p["pw"] + p["pb"]
    sl = silu_mojo(scene_f @ p["se_w"] + p["se_b"])
    g = sigmoid_mojo(rl @ p["gw"] + p["gb"])
    fused = rl + g * sl
    return nb.mean(square(fused - gt))


p = {
    "f1": _w(3, 3, 4, D), "b1": _b(D),
    "f2": _w(3, 3, D, D), "b2": _b(D),
    "pw": _w(D, D), "pb": _b(D),
    "se_w": _w(18, D), "se_b": _b(D),
    "gw": _w(D, D), "gb": _b(D),
}

run_test("both Mojo + grad w.r.t. dict", loss_params, [p, rgba, scene_f, gt])

# ── Test 6: 256x256 resolution ─────────────────────────────────
print("\n[6] 256x256 resolution (both Mojo)")
rgba256 = _w(1, 256, 256, 4)
scene_f256 = _w(1, 18)
gt256 = _w(1, D)


def loss_256(p, rgba, scene_f, gt):
    x = silu_mojo(conv2d(rgba, p["f1"], stride=2, padding=1, bias=p["b1"]))
    x = silu_mojo(conv2d(x, p["f2"], stride=2, padding=1, bias=p["b2"]))
    rl = x.mean(axis=(1, 2)) @ p["pw"] + p["pb"]
    sl = silu_mojo(scene_f @ p["se_w"] + p["se_b"])
    g = sigmoid_mojo(rl @ p["gw"] + p["gb"])
    fused = rl + g * sl
    return nb.mean(square(fused - gt))


run_test("256x256 both Mojo", loss_256, [p, rgba256, scene_f256, gt256])

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE — check which tests PASS/FAIL above")
print("=" * 70)