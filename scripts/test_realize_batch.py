"""Quick test: can we batch-realize non-conv2d gradients in one call?"""

import sys
sys.path.insert(0, "src")

import numpy as np
import nabla as nb
import mitsuba as mi

mi.set_variant("scalar_rgb")

from omen.config import OmenConfig
from omen.model.jepa import OmenJEPA
from omen.scenes import build_cornell_box
from omen.training.trainer.loss import compute_training_loss


def _t(a, batch=False):
    a = np.asarray(a, dtype=np.float32)
    if batch:
        a = a[np.newaxis]
    return nb.Tensor.from_dlpack(a)


# CONV2D FILTER NAMES (bare tensors, not in named_parameters)
CONV2D_BLOCKERS = {"decoder.e1", "decoder.e2", "decoder.e3", "decoder.e4"}

config = OmenConfig.v1_dense()
model = OmenJEPA(config=config)

scene, sg_raw = build_cornell_box(resolution=(32, 32))
sg = {
    "geometry": {"vertices": _t(sg_raw["geometry"]["vertices"], batch=True)},
    "materials": {"params": _t(sg_raw["materials"]["params"], batch=True)},
    "lights": {"params": _t(sg_raw["lights"]["params"], batch=True)},
}
gt_np = np.array(mi.render(scene, spp=4, seed=0))[:, :, :3]
noisy_np = np.array(mi.render(scene, spp=2, seed=42))[:, :, :3]
H, W, _ = gt_np.shape
gt = _t(np.concatenate([gt_np, np.ones((H, W, 1), np.float32)], axis=-1)[np.newaxis])
noisy = _t(
    np.concatenate([noisy_np, np.ones((H, W, 1), np.float32)], axis=-1)[np.newaxis]
)

print("Running value_and_grad...", flush=True)
params = model.state_dict()
total_loss, grads = nb.value_and_grad(compute_training_loss, argnums=0)(
    params, model, noisy, gt, sg, config
)
print(f"Got {len(grads)} gradient tensors", flush=True)

# Split into safe (non-conv2d) and blocked (conv2d) grads
safe_grads = {n: g for n, g in grads.items() if n not in CONV2D_BLOCKERS}
blocked_grads = {n: g for n, g in grads.items() if n in CONV2D_BLOCKERS}
print(f"Safe: {len(safe_grads)}, Blocked: {len(blocked_grads)}", flush=True)

# TEST 1: realize all safe grads at once via realize_all
print("\nTEST 1: nb.realize_all on safe grads...", flush=True)
safe_tensors = [g for g in safe_grads.values() if nb.is_tensor(g) and not g.real]
print(f"  Realizing {len(safe_tensors)} tensors...", flush=True)
try:
    nb.realize_all(*safe_tensors[:5])  # try first 5
    print("  SUCCESS: first 5 realized together", flush=True)
except Exception as e:
    print(f"  FAIL: {str(e)[:100]}", flush=True)

# TEST 2: realize safe grads individually (timing)
print("\nTEST 2: individual realization timing (first 5)...", flush=True)
import time
for name in list(safe_grads.keys())[:5]:
    t0 = time.time()
    try:
        val = safe_grads[name].to_numpy()
        dt = time.time() - t0
        print(f"  {name}: {dt:.1f}s shape={val.shape}", flush=True)
    except Exception as e:
        dt = time.time() - t0
        print(f"  {name}: FAIL {dt:.1f}s {str(e)[:60]}", flush=True)

# TEST 3: realize blocked grads individually
print("\nTEST 3: blocked conv2d grads...", flush=True)
for name in blocked_grads:
    t0 = time.time()
    try:
        val = grads[name].to_numpy()
        dt = time.time() - t0
        print(f"  {name}: OK {dt:.1f}s", flush=True)
    except Exception as e:
        dt = time.time() - t0
        print(f"  {name}: FAIL {dt:.1f}s", flush=True)

print("\nDone.", flush=True)
