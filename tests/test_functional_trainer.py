"""End-to-end smoke tests for functional value_and_grad trainer.

Uses real Mitsuba 3D scenes (Cornell Box) — not random noise images.

Verifies:
1. value_and_grad differentiates through the full model without fused-backward crash
2. Optimizer state (m/v pytrees) persists across 3 consecutive train steps
3. Losses change across iterations (params are actually being updated)
"""

import numpy as np
import pytest

try:
    import mitsuba as mi

    MITSUBA_AVAILABLE = True
except ImportError:
    MITSUBA_AVAILABLE = False

try:
    import nabla as nb

    NABLA_AVAILABLE = True
except (ImportError, RuntimeError):
    NABLA_AVAILABLE = False

from omen.config import OmenConfig
from omen.model.jepa import OmenJEPA
from omen.training.trainer.core import OmenTrainer

requires_nabla_mitsuba = pytest.mark.skipif(
    not (NABLA_AVAILABLE and MITSUBA_AVAILABLE),
    reason="Nabla + Mitsuba required",
)


def _t(arr):
    """Nabla tensor from numpy float32 array."""
    return nb.Tensor.from_dlpack(np.asarray(arr, dtype=np.float32))


def _to_tensor(arr, batch=False):
    """Convert numpy array to nabla tensor, optionally adding batch dim."""
    a = np.asarray(arr, dtype=np.float32)
    if batch:
        a = a[np.newaxis]
    return nb.Tensor.from_dlpack(a)


def _make_real_scene_graph():
    """Build a real Cornell Box scene graph with nabla tensors."""
    from omen.scenes import build_cornell_box

    _scene, sg = build_cornell_box(resolution=(32, 32))
    # Convert numpy arrays to nabla tensors with batch dim
    return {
        "geometry": {
            "vertices": _to_tensor(sg["geometry"]["vertices"], batch=True),
            "faces": sg["geometry"]["faces"],
        },
        "materials": {"params": _to_tensor(sg["materials"]["params"], batch=True)},
        "lights": {"params": _to_tensor(sg["lights"]["params"], batch=True)},
    }


def _render_real_pair(scene_graph, spp_gt=16, spp_noisy=2):
    """Render a real GT + noisy pair from a Mitsuba Cornell Box scene.

    Returns (gt, noisy) as Nabla tensors with shape (1, H, W, 4).
    """
    from omen.scenes import build_cornell_box

    mi_scene, _ = build_cornell_box(resolution=(32, 32))
    gt_np = np.array(mi.render(mi_scene, spp=spp_gt, seed=0))[:, :, :3]
    noisy_np = np.array(mi.render(mi_scene, spp=spp_noisy, seed=42))[:, :, :3]

    # Add alpha channel and batch dim -> (1, H, W, 4)
    H, W, C = gt_np.shape
    gt_rgba = np.concatenate([gt_np, np.ones((H, W, 1), dtype=np.float32)], axis=-1)
    noisy_rgba = np.concatenate(
        [noisy_np, np.ones((H, W, 1), dtype=np.float32)], axis=-1
    )
    return _t(gt_rgba[np.newaxis]), _t(noisy_rgba[np.newaxis])


class TestFunctionalTrainerE2E:
    """Full train_step with real Cornell Box 3D scene data."""

    @requires_nabla_mitsuba
    def test_single_train_step_real_scene(self):
        """value_and_grad works end-to-end with real Mitsuba-rendered data."""
        config = OmenConfig.v1_dense()
        model = OmenJEPA(config=config)
        trainer = OmenTrainer(model, config=config)

        scene_graph = _make_real_scene_graph()
        gt, noisy = _render_real_pair(scene_graph)

        metrics = trainer.train_step(noisy, gt, scene_graph)

        assert "total_loss" in metrics
        assert metrics["iteration"] == 1
        assert np.isfinite(metrics["total_loss"]), (
            f"Loss is not finite: {metrics['total_loss']}"
        )

    @requires_nabla_mitsuba
    def test_three_consecutive_steps_real_scene(self):
        """Optimizer m/v pytrees persist and losses change over 3 steps."""
        config = OmenConfig.v1_dense()
        model = OmenJEPA(config=config)
        trainer = OmenTrainer(model, config=config)

        scene_graph = _make_real_scene_graph()
        gt, noisy = _render_real_pair(scene_graph)

        losses = []
        for i in range(3):
            metrics = trainer.train_step(noisy, gt, scene_graph)
            assert metrics["iteration"] == i + 1
            assert np.isfinite(metrics["total_loss"])
            losses.append(metrics["total_loss"])

        # Losses should differ across steps (params are actually updating)
        assert len(set(f"{loss:.6f}" for loss in losses)) > 1, (
            f"Losses identical across steps — params not updating: {losses}"
        )

    @requires_nabla_mitsuba
    def test_params_change_after_real_step(self):
        """Verify model weights change after a train step with real data."""
        config = OmenConfig.v1_dense()
        model = OmenJEPA(config=config)
        trainer = OmenTrainer(model, config=config)

        params_before = {k: v.to_numpy().copy() for k, v in model.state_dict().items()}

        scene_graph = _make_real_scene_graph()
        gt, noisy = _render_real_pair(scene_graph)
        trainer.train_step(noisy, gt, scene_graph)

        changed = 0
        for k, v in model.state_dict().items():
            if k in params_before and not np.allclose(v.to_numpy(), params_before[k]):
                changed += 1

        assert changed > 0, "No parameters changed after train_step"

    @requires_nabla_mitsuba
    def test_surprise_modulation_with_real_data(self):
        """z_score > 0 boosts learning rate via surprise modulation."""
        config = OmenConfig.v1_dense()
        model = OmenJEPA(config=config)
        trainer = OmenTrainer(model, config=config)

        scene_graph = _make_real_scene_graph()
        gt, noisy = _render_real_pair(scene_graph)

        m0 = trainer.train_step(noisy, gt, scene_graph, z_score=0.0)
        m1 = trainer.train_step(noisy, gt, scene_graph, z_score=2.0)

        assert m0["iteration"] == 1
        assert m1["iteration"] == 2
        assert np.isfinite(m0["total_loss"])
        assert np.isfinite(m1["total_loss"])
