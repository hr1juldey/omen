"""End-to-end training smoke test — 3 steps with real Cornell Box data.

Run via: pixi run python scripts/test_e2e_training.py
"""

import sys
import time

sys.path.insert(0, "src")

import numpy as np
import nabla as nb
import mitsuba as mi

mi.set_variant("scalar_rgb")

from omen.config import OmenConfig
from omen.model.jepa import OmenJEPA
from omen.training.trainer.core import OmenTrainer
from omen.scenes import build_cornell_box


def _t(a, batch=False):
    a = np.asarray(a, dtype=np.float32)
    if batch:
        a = a[np.newaxis]
    return nb.Tensor.from_dlpack(a)


def main():
    t_start = time.time()

    print("=" * 60, flush=True)
    print("Omen E2E Training Smoke Test (Cornell Box)", flush=True)
    print("=" * 60, flush=True)

    # 1. Create model + trainer
    print("\n[1/5] Creating model + trainer...", flush=True)
    config = OmenConfig.v1_dense()
    model = OmenJEPA(config=config)
    trainer = OmenTrainer(model, config=config)
    n_params = len(list(model.named_parameters()))
    print(f"  Model: {n_params} named parameters", flush=True)
    print(f"  Components: {list(trainer._components.keys())}", flush=True)

    # 2. Build real scene data
    print("\n[2/5] Building Cornell Box scene...", flush=True)
    scene, sg_raw = build_cornell_box(resolution=(32, 32))
    sg = {
        "geometry": {"vertices": _t(sg_raw["geometry"]["vertices"], batch=True)},
        "materials": {"params": _t(sg_raw["materials"]["params"], batch=True)},
        "lights": {"params": _t(sg_raw["lights"]["params"], batch=True)},
    }

    # 3. Render GT + noisy pair
    print("\n[3/5] Rendering GT@4spp + noisy@2spp...", flush=True)
    gt_np = np.array(mi.render(scene, spp=4, seed=0))[:, :, :3]
    noisy_np = np.array(mi.render(scene, spp=2, seed=42))[:, :, :3]
    H, W, _ = gt_np.shape
    gt = _t(np.concatenate([gt_np, np.ones((H, W, 1), np.float32)], axis=-1)[np.newaxis])
    noisy = _t(
        np.concatenate([noisy_np, np.ones((H, W, 1), np.float32)], axis=-1)[np.newaxis]
    )
    print(f"  Render size: {H}x{W}", flush=True)

    # 4. Run 3 training steps
    print("\n[4/5] Running 3 training steps...", flush=True)
    losses = []
    for i in range(3):
        t0 = time.time()
        metrics = trainer.train_step(noisy, gt, sg)
        dt = time.time() - t0
        losses.append(metrics["total_loss"])
        print(
            f"  Step {i+1}: loss={metrics['total_loss']:.4f} "
            f"iter={metrics['iteration']} time={dt:.1f}s",
            flush=True,
        )

    # 5. Validate
    print("\n[5/5] Validating...", flush=True)
    unique = len(set(f"{x:.4f}" for x in losses))
    all_finite = all(np.isfinite(l) for l in losses)

    print(f"  Losses: {[f'{x:.4f}' for x in losses]}", flush=True)
    print(f"  Unique values: {unique}/3", flush=True)
    print(f"  All finite: {all_finite}", flush=True)

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s", flush=True)

    if unique > 1 and all_finite:
        print("\n*** ALL CHECKS PASSED ***", flush=True)
        return 0
    else:
        print("\n*** CHECKS FAILED ***", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
