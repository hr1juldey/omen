"""Online training data generator for JEPA self-supervised learning.

Renders GT (high SPP) and noisy (low SPP) image pairs in-memory.
Images are NOT saved to disk by default.
"""

import hashlib
import json
import logging
import os
import time

import numpy as np

try:
    import mitsuba as mi
    MITSUBA_AVAILABLE = True
except ImportError:
    MITSUBA_AVAILABLE = False

logger = logging.getLogger("omen.training.online_gen")


def _render_pair(scene, gt_spp: int, noisy_spp: int, seed: int) -> tuple:
    """Render one GT + noisy image pair. Returns (gt, noisy) numpy arrays."""
    gt = np.array(mi.render(scene, spp=gt_spp, seed=0))[:, :, :3]
    noisy = np.array(mi.render(scene, spp=noisy_spp, seed=seed + 1000))[:, :, :3]
    return gt.astype(np.float32), noisy.astype(np.float32)


def _step_dict(gt, noisy, scene_graph, seed: int, **extra) -> dict:
    """Build a training step data dict."""
    return {
        "gt_image": gt,
        "noisy_image": noisy,
        "scene_graph": scene_graph,
        "residual": (gt - noisy).astype(np.float32),
        "seed": seed,
        **extra,
    }


def _save_debug(step_data: dict, output_dir: str, step_idx: int, suffix: str = ""):
    """Save GT and noisy images to disk for debugging."""
    os.makedirs(output_dir, exist_ok=True)
    gt_path = os.path.join(output_dir, f"step{step_idx:04d}{suffix}_gt.exr")
    noisy_path = os.path.join(output_dir, f"step{step_idx:04d}{suffix}_noisy.exr")
    try:
        mi.Bitmap(step_data["gt_image"]).write(gt_path)
        mi.Bitmap(step_data["noisy_image"]).write(noisy_path)
        logger.info("Saved: %s, %s", gt_path, noisy_path)
    except Exception as exc:
        logger.warning("Failed to save debug images: %s", exc)


class TrainingDataGenerator:
    """Online training data generator — renders GT+noisy pairs in-memory.

    Usage::

        gen = TrainingDataGenerator(resolution=(1920, 1080), gt_spp=256)
        for step_data in gen.train_step(build_cornell_box):
            loss = model.train_step(step_data)
            del step_data
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (1920, 1080),
        gt_spp: int = 256,
        noisy_spp: int = 4,
        save_images: bool = False,
        output_dir: str = "./debug/",
    ):
        if not MITSUBA_AVAILABLE:
            raise ImportError("Mitsuba required for TrainingDataGenerator")
        available = set(mi.variants())
        variant = next(v for v in ("cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb") if v in available)
        mi.set_variant(variant)
        logger.info("Mitsuba variant: %s", variant)
        self.resolution = resolution
        self.gt_spp = gt_spp
        self.noisy_spp = noisy_spp
        self.save_images = save_images
        self.output_dir = output_dir
        self._step = 0
        logger.info("Init: res=%s, gt_spp=%d, noisy_spp=%d", resolution, gt_spp, noisy_spp)

    def train_step(self, scene_builder, camera: str = "default", seed: int | None = None):
        """Render one GT + noisy pair. Yields step_data dict."""
        if seed is None:
            seed = self._step
        self._step += 1
        scene, sg = scene_builder(resolution=self.resolution)
        if camera == "all":
            yield from self._all_cameras(scene, sg, seed)
            return
        t0 = time.perf_counter()
        gt, noisy = _render_pair(scene, self.gt_spp, self.noisy_spp, seed)
        logger.info("Step %d: rendered in %.2fs", self._step, time.perf_counter() - t0)
        data = _step_dict(gt, noisy, sg, seed)
        if self.save_images:
            _save_debug(data, self.output_dir, self._step)
        yield data

    def _all_cameras(self, scene, sg, base_seed):
        """Train on all camera positions with independent seeds."""
        for cam_idx in range(5):
            seed = base_seed + cam_idx * 100
            gt, noisy = _render_pair(scene, self.gt_spp, self.noisy_spp, seed)
            self._step += 1
            data = _step_dict(gt, noisy, sg, seed, camera_idx=cam_idx)
            if self.save_images:
                _save_debug(data, self.output_dir, self._step)
            yield data

    def train_animation(self, animation_frames, scene_graph):
        """Train on animation frames — camera motion, mesh, material, light.

        Args:
            animation_frames: iterable of mi.Scene objects (from
                ``cornell_animations()`` generators).
            scene_graph: shared scene graph dict for the animation.
        """
        for frame_idx, scene in enumerate(animation_frames):
            seed = self._step * 100 + frame_idx
            self._step += 1
            t0 = time.perf_counter()
            gt, noisy = _render_pair(scene, self.gt_spp, self.noisy_spp, seed)
            dt = time.perf_counter() - t0
            logger.info("Anim frame %d: rendered in %.2fs", frame_idx, dt)
            data = _step_dict(gt, noisy, scene_graph, seed, frame_idx=frame_idx)
            if self.save_images:
                _save_debug(data, self.output_dir, self._step, suffix=f"_f{frame_idx}")
            yield data

    def generate_batch(self, scene_builder, count: int = 10, replay_buffer=None):
        """Generate N pairs for replay buffer. Returns list of step_data."""
        scene, sg = scene_builder(resolution=self.resolution)
        sg_str = json.dumps(
            {k: v.tolist() if isinstance(v, np.ndarray) else str(v) for k, v in sg.items()},
            sort_keys=True,
        )
        scene_hash = hashlib.md5(sg_str.encode()).hexdigest()[:12]
        pairs = []
        for i in range(count):
            seed = self._step + i
            gt, noisy = _render_pair(scene, self.gt_spp, self.noisy_spp, seed)
            self._step += 1
            data = _step_dict(gt, noisy, sg, seed, scene_hash=scene_hash)
            if replay_buffer is not None:
                replay_buffer.add(scene_hash, data)
            pairs.append(data)
        logger.info("Generated %d pairs for scene %s", count, scene_hash)
        return pairs
