"""Animation sequence data generator for temporal JEPA training.

Renders animation frames (GT + noisy) for temporal world-model training.
Delegates core rendering to omen.training.online_gen helpers.
"""

import logging

import numpy as np

try:
    import mitsuba as mi
    MITSUBA_AVAILABLE = True
except ImportError:
    MITSUBA_AVAILABLE = False

from omen.training.online_gen import _render_pair, _step_dict, _save_debug

logger = logging.getLogger("omen.training.anim_gen")

# Max cameras per scene for multi-camera animation
_MAX_CAMERAS = 5


class AnimationDataGenerator:
    """Renders animation frame sequences for temporal training.

    Usage::

        gen = AnimationDataGenerator(resolution=(1920, 1080), gt_spp=256)
        for frame_data in gen.animate(build_cornell_box, anim="camera_orbit"):
            loss = model.temporal_step(frame_data)
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
            raise ImportError("Mitsuba required for AnimationDataGenerator")
        mi.set_variant("scalar_rgb")
        self.resolution = resolution
        self.gt_spp = gt_spp
        self.noisy_spp = noisy_spp
        self.save_images = save_images
        self.output_dir = output_dir
        self._step = 0

    def animate(self, scene_builder, anim_name: str = "camera_orbit",
                n_frames: int | None = None):
        """Render animation frames. Yields step_data per frame."""
        from omen.scenes import SCENE_REGISTRY
        scene_name = next(
            (n for n, b in SCENE_REGISTRY.items() if b == scene_builder), None
        )
        frames = self._get_frames(scene_name, anim_name)
        if frames is None:
            logger.warning("No animations for scene '%s'", scene_name)
            return

        for frame_idx, scene in enumerate(frames):
            if n_frames is not None and frame_idx >= n_frames:
                break
            gt, noisy = _render_pair(scene, self.gt_spp, self.noisy_spp, frame_idx + 2000)
            self._step += 1
            data = _step_dict(gt, noisy, {}, frame_idx + 2000,
                              frame_idx=frame_idx, anim_name=anim_name)
            if self.save_images:
                _save_debug(data, self.output_dir, self._step, f"_frame{frame_idx:04d}")
            yield data

    def _get_frames(self, scene_name: str | None, anim_name: str):
        """Resolve animation frames for the given scene."""
        from omen.scenes import (
            cornell_animations, veach_animations, shaderball_animations,
            studio_animations, foggy_animations,
        )
        anim_funcs = {
            "cornell": cornell_animations,
            "veach": veach_animations,
            "shaderball": shaderball_animations,
            "studio": studio_animations,
            "foggy": foggy_animations,
        }
        if scene_name not in anim_funcs:
            return None
        anims = anim_funcs[scene_name](base_resolution=self.resolution)
        if anim_name not in anims:
            logger.warning("Animation '%s' not found, using camera_orbit", anim_name)
        return anims.get(anim_name, anims.get("camera_orbit"))
