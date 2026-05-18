"""Test Mojo kernels on GPU-transferred tensors — find SIGSEGV root cause."""
import time
import numpy as np
import nabla as nb
from nabla.ops import conv2d

from omen.kernels.activations import sigmoid_gpu, silu_gpu, square
from omen.kernels.activations_gpu import sigmoid_mojo, silu_mojo

dev = nb.Accelerator()
CPU = nb.CPU


def to_dev(arr):
    return nb.ops.transfer_to(nb.Tensor.from_dlpack(arr.astype(np.float32)), dev)


def to_cpu(tensor):
    return nb.ops.transfer_to(tensor, CPU()).to_numpy()


print("=" * 60)
print("Mojo kernels on GPU-transferred tensors")
print("=" * 60)

np.random.seed(42)

# ── Test 1: silu_mojo on GPU tensor ───────────────────────────
print("\n[1] silu_mojo on GPU tensor")
x_cpu = np.random.randn(1, 64, 64, 4).astype(np.float32)
x_gpu = to_dev(x_cpu)
print(f"  device: {x_gpu.device}")
try:
    out = silu_mojo(x_gpu)
    print(f"  forward OK: {out.shape} device={out.device}")
    val = float(to_cpu(out.mean()))
    print(f"  mean={val:.4f}")
except Exception as e:
    print(f"  FAIL: {e}")

# ── Test 2: value_and_grad with GPU params + GPU data ─────────
print("\n[2] value_and_grad with GPU params + Mojo silu")

def _he(s):
    return np.random.randn(*s).astype(np.float32) * 0.02


def _z(n):
    return np.zeros(n, dtype=np.float32)


p_np = {
    "f1": _he((3, 3, 4, 16)), "b1": _z(16),
    "f2": _he((3, 3, 16, 32)), "b2": _z(32),
    "pw": _he((64, 256)), "pb": _z(256),
}
noisy_np = np.random.rand(1, 64, 64, 4).astype(np.float32)
gt_np = np.random.randn(1, 256).astype(np.float32) * 0.01


def loss_mojo_gpu(p, noisy, gt):
    x = silu_mojo(conv2d(noisy, p["f1"], stride=2, padding=1, bias=p["b1"]))
    x = silu_mojo(conv2d(x, p["f2"], stride=2, padding=1, bias=p["b2"]))
    rl = x.mean(axis=(1, 2)) @ p["pw"] + p["pb"]
    return nb.mean(square(rl - gt))


p_gpu = {k: to_dev(v) for k, v in p_np.items()}
noisy_gpu = to_dev(noisy_np)
gt_gpu = to_dev(gt_np)

print(f"  params on GPU: {p_gpu['f1'].device}")
try:
    t0 = time.time()
    val, grads = nb.value_and_grad(loss_mojo_gpu, argnums=0)(p_gpu, noisy_gpu, gt_gpu)
    dt = time.time() - t0
    print(f"  PASS: val={float(to_cpu(val)):.6f} time={dt:.1f}s")
except Exception as e:
    print(f"  FAIL: {e}")

# ── Test 3: Full Stage B model on GPU with Mojo ───────────────
print("\n[3] Full Stage B on GPU with Mojo (random data)")

p2_np = {
    "re_f1": _he((3, 3, 4, 16)), "re_b1": _z(16),
    "re_f2": _he((3, 3, 16, 32)), "re_b2": _z(32),
    "re_f3": _he((3, 3, 32, 64)), "re_b3": _z(64),
    "re_pw": _he((64, 256)), "re_pb": _z(256),
    "se_w1": _he((18, 64)), "se_b1": _z(64),
    "se_pw": _he((64, 256)), "se_pb": _z(256),
    "ca_gw": _he((256, 256)), "ca_gb": _z(256),
}
scene_np = np.random.rand(1, 18).astype(np.float32)


def loss_stage_b_mojo(p, noisy, scene_f, gt):
    x = silu_mojo(conv2d(noisy, p["re_f1"], stride=2, padding=1, bias=p["re_b1"]))
    x = silu_mojo(conv2d(x, p["re_f2"], stride=2, padding=1, bias=p["re_b2"]))
    x = silu_mojo(conv2d(x, p["re_f3"], stride=2, padding=1, bias=p["re_b3"]))
    rl = x.mean(axis=(1, 2)) @ p["re_pw"] + p["re_pb"]
    h = silu_mojo(scene_f @ p["se_w1"] + p["se_b1"])
    sl = h @ p["se_pw"] + p["se_pb"]
    g = sigmoid_mojo(rl @ p["ca_gw"] + p["ca_gb"])
    fused = rl + g * sl
    return nb.mean(square(fused - gt))


p2_gpu = {k: to_dev(v) for k, v in p2_np.items()}
scene_gpu = to_dev(scene_np)

try:
    t0 = time.time()
    val, grads = nb.value_and_grad(loss_stage_b_mojo, argnums=0)(
        p2_gpu, noisy_gpu, scene_gpu, gt_gpu
    )
    dt = time.time() - t0
    print(f"  PASS: val={float(to_cpu(val)):.6f} time={dt:.1f}s")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  FAIL: {e}")

# ── Test 4: Same but with Mitsuba-rendered data ────────────────
print("\n[4] Full Stage B on GPU with Mojo + MITSUBA data")
try:
    import mitsuba as mi

    mi.set_variant("scalar_rgb")
    from omen.scenes import build_shaderball

    noisy_rgba, gt_rgb, scene_feat = build_shaderball(
        res=64, gt_spp=64, noisy_spp=2
    )
    print(f"  Mitsuba: noisy={noisy_rgba.shape} scene={scene_feat.shape}")
    noisy_m_gpu = to_dev(noisy_rgba)
    scene_m_gpu = to_dev(scene_feat)
    gt_m_gpu = to_dev(np.random.randn(1, 256).astype(np.float32) * 0.01)

    val, grads = nb.value_and_grad(loss_stage_b_mojo, argnums=0)(
        p2_gpu, noisy_m_gpu, scene_m_gpu, gt_m_gpu
    )
    print(f"  PASS: val={float(to_cpu(val)):.6f}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  FAIL: {e}")

# ── Test 5: Pure-nabla with Mitsuba data ──────────────────────
print("\n[5] Pure-nabla with Mitsuba data (sanity check)")


def loss_stage_b_pure(p, noisy, scene_f, gt):
    x = silu_gpu(conv2d(noisy, p["re_f1"], stride=2, padding=1, bias=p["re_b1"]))
    x = silu_gpu(conv2d(x, p["re_f2"], stride=2, padding=1, bias=p["re_b2"]))
    x = silu_gpu(conv2d(x, p["re_f3"], stride=2, padding=1, bias=p["re_b3"]))
    rl = x.mean(axis=(1, 2)) @ p["re_pw"] + p["re_pb"]
    h = silu_gpu(scene_f @ p["se_w1"] + p["se_b1"])
    sl = h @ p["se_pw"] + p["se_pb"]
    g = sigmoid_gpu(rl @ p["ca_gw"] + p["ca_gb"])
    fused = rl + g * sl
    return nb.mean(square(fused - gt))


try:
    p3_gpu = {k: to_dev(v) for k, v in p2_np.items()}
    val, grads = nb.value_and_grad(loss_stage_b_pure, argnums=0)(
        p3_gpu, noisy_m_gpu, scene_m_gpu, gt_m_gpu
    )
    print(f"  PASS: val={float(to_cpu(val)):.6f}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  FAIL: {e}")

print("\n" + "=" * 60)
print("DONE")
