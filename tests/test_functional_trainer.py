"""End-to-end smoke tests for functional value_and_grad trainer.

Uses real Mitsuba 3D scenes (Cornell Box).

Verifies:
1. value_and_grad differentiates through the full model — forward loss realizes
2. Gradient tensors are produced with correct shapes (lazy)
3. Per-component AdamW updates are applied without error

Known limitation: multi-step training is blocked by a MAX compiler bug
(cannot infer ``num_groups`` when realizing the fused backward graph through
multiple conv2d ops).  Single-step training works correctly.  The test
``test_three_steps_losses_differ`` is marked xfail until the compiler fix.
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


def _t(arr, batch=False):
    """Nabla tensor from numpy float32 array."""
    a = np.asarray(arr, dtype=np.float32)
    if batch:
        a = a[np.newaxis]
    return nb.Tensor.from_dlpack(a)


def _make_real_scene_graph():
    """Build a real Cornell Box scene graph with nabla tensors."""
    from omen.scenes import build_cornell_box

    _scene, sg = build_cornell_box(resolution=(32, 32))
    return {
        "geometry": {
            "vertices": _t(sg["geometry"]["vertices"], batch=True),
        },
        "materials": {"params": _t(sg["materials"]["params"], batch=True)},
        "lights": {"params": _t(sg["lights"]["params"], batch=True)},
    }


def _render_real_pair():
    """Render a real GT + noisy pair from a Mitsuba Cornell Box scene."""
    from omen.scenes import build_cornell_box

    mi_scene, _ = build_cornell_box(resolution=(32, 32))
    gt_np = np.array(mi.render(mi_scene, spp=4, seed=0))[:, :, :3]
    noisy_np = np.array(mi.render(mi_scene, spp=2, seed=42))[:, :, :3]
    H, W, _ = gt_np.shape
    gt_rgba = np.concatenate([gt_np, np.ones((H, W, 1), dtype=np.float32)], axis=-1)
    noisy_rgba = np.concatenate(
        [noisy_np, np.ones((H, W, 1), dtype=np.float32)], axis=-1
    )
    return _t(gt_rgba[np.newaxis]), _t(noisy_rgba[np.newaxis])


class TestFunctionalTrainerE2E:
    """Full train_step with real Cornell Box 3D scene data."""

    @requires_nabla_mitsuba
    def test_forward_loss_realizes_from_real_scene(self):
        """Forward pass through full model produces a finite, realizable loss."""
        config = OmenConfig.v1_dense()
        model = OmenJEPA(config=config)
        trainer = OmenTrainer(model, config=config)

        scene_graph = _make_real_scene_graph()
        gt, noisy = _render_real_pair()

        metrics = trainer.train_step(noisy, gt, scene_graph)

        assert "total_loss" in metrics
        assert metrics["iteration"] == 1
        assert np.isfinite(metrics["total_loss"])

    @requires_nabla_mitsuba
    def test_value_and_grad_produces_gradients(self):
        """value_and_grad returns lazy gradient tensors with correct count."""
        config = OmenConfig.v1_dense()
        model = OmenJEPA(config=config)
        from omen.training.trainer.loss import compute_training_loss

        scene_graph = _make_real_scene_graph()
        gt, noisy = _render_real_pair()

        params = model.state_dict()
        total_loss, grads = nb.value_and_grad(compute_training_loss, argnums=0)(
            params, model, noisy, gt, scene_graph, config
        )

        assert len(grads) == len(params), (
            f"Expected {len(params)} grads, got {len(grads)}"
        )
        assert all(nb.is_tensor(g) for g in grads.values())

    @requires_nabla_mitsuba
    @pytest.mark.xfail(
        reason="MAX compiler cannot infer num_groups for fused backward "
        "through multiple conv2d ops. Single-step training works."
    )
    def test_three_steps_losses_differ(self):
        """Optimizer updates params across 3 steps — losses should change."""
        config = OmenConfig.v1_dense()
        model = OmenJEPA(config=config)
        trainer = OmenTrainer(model, config=config)

        scene_graph = _make_real_scene_graph()
        gt, noisy = _render_real_pair()

        losses = []
        for i in range(3):
            metrics = trainer.train_step(noisy, gt, scene_graph)
            assert metrics["iteration"] == i + 1
            assert np.isfinite(metrics["total_loss"])
            losses.append(metrics["total_loss"])

        unique = len(set(f"{loss:.4f}" for loss in losses))
        assert unique > 1, f"Losses identical across 3 steps: {losses}"

    @requires_nabla_mitsuba
    @pytest.mark.xfail(
        reason="MAX compiler cannot infer num_groups for fused backward "
        "through multiple conv2d ops. Single-step training works."
    )
    def test_surprise_modulation_with_real_data(self):
        """z_score > 0 produces different loss trajectory via LR modulation."""
        config = OmenConfig.v1_dense()
        model = OmenJEPA(config=config)
        trainer = OmenTrainer(model, config=config)

        scene_graph = _make_real_scene_graph()
        gt, noisy = _render_real_pair()

        m0 = trainer.train_step(noisy, gt, scene_graph, z_score=0.0)
        m1 = trainer.train_step(noisy, gt, scene_graph, z_score=2.0)

        assert m0["iteration"] == 1
        assert m1["iteration"] == 2
        assert np.isfinite(m0["total_loss"])
        assert np.isfinite(m1["total_loss"])
