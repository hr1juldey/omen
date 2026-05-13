"""OmenSession — Pipeline orchestrator for a single render.

Routes through the existing AI pipeline in src/omen/:
  sync depsgraph → build mi.Scene → render_denoiser → clean RGBA

Does NOT implement its own render or denoising logic.
All AI inference is delegated to src/omen/modes/denoiser.py.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omen_engine.sync import OmenSync

logger = logging.getLogger(__name__)


class OmenSession:
    """Render session: sync → build scene → full AI pipeline → output."""

    def __init__(self) -> None:
        self._sync = OmenSync()
        self._bridge: Any = None

    def _get_bridge(self) -> Any:
        """Lazy-init JEPABridge on first render, cache for reuse."""
        if self._bridge is None:
            try:
                from omen.jepa_bridge import JEPABridge
                self._bridge = JEPABridge()
                logger.info("JEPABridge initialized")
            except Exception as exc:
                logger.warning("JEPABridge unavailable: %s", exc)
                self._bridge = None
        return self._bridge

    def render_scene(
        self,
        depsgraph: Any,
        spp: int = 4,
        max_depth: int = 8,
        tier: str = "medium",
        mode: str = "denoise",
        train: bool = True,
    ) -> NDArray[np.float32]:
        """Full pipeline: sync → build scene → render_denoiser → clean RGBA.

        Routes through src/omen/modes/denoiser.render_denoiser() which runs:
          mi.render(AOV) → scene_extractor → Mojo kernels → JEPA → clean
        """
        try:
            scene_data = self._sync.sync(depsgraph)

            from omen_engine.backends.mitsuba_backend import build_scene
            mi_scene = build_scene(
                vertices=scene_data["vertices"],
                faces=scene_data["faces"],
                camera_matrix=scene_data["camera_matrix"],
                camera_fov=scene_data["camera_fov"],
                width=scene_data["width"],
                height=scene_data["height"],
                lights=scene_data["lights"],
                materials=scene_data["materials"],
            )

            if mode == "denoise":
                from omen.modes.denoiser import render_denoiser
                bridge = self._get_bridge()
                result = render_denoiser(mi_scene, bridge, spp=spp, tier=tier, train=train)
            elif mode == "adaptive":
                from omen.modes.adaptive import render_adaptive
                bridge = self._get_bridge()
                result = render_adaptive(mi_scene, bridge, spp=spp, tier=tier)
            elif mode == "multires":
                from omen.modes.multires import render_multires
                bridge = self._get_bridge()
                result = render_multires(mi_scene, bridge, spp=spp, tier=tier)
            else:
                # mode == "path" — raw render, no AI
                result = self._render_path_only(mi_scene, spp, max_depth)

            if result.ndim == 3 and result.shape[-1] == 4:
                return np.clip(result, 0.0, 1.0)
            elif result.ndim == 3 and result.shape[-1] == 3:
                h, w = result.shape[:2]
                rgba = np.zeros((h, w, 4), dtype=np.float32)
                rgba[:, :, :3] = result
                rgba[:, :, 3] = 1.0
                return np.clip(rgba, 0.0, 1.0)

            return result

        except Exception as exc:
            logger.error("Render failed: %s", exc, exc_info=True)
            return np.zeros((1, 1, 4), dtype=np.float32)

    @staticmethod
    def _render_path_only(mi_scene: Any, spp: int, max_depth: int) -> NDArray[np.float32]:
        """Fallback: render without JEPA, just path tracing."""
        import mitsuba as mi
        integrator = mi.load_dict({"type": "path", "max_depth": max_depth})
        result = mi.render(mi_scene, integrator=integrator, spp=spp)
        raw = np.array(result, copy=False, dtype=np.float32)
        if raw.ndim == 2:
            raw = raw[:, :, np.newaxis]
        h, w = raw.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = raw[:, :, :3]
        rgba[:, :, 3] = 1.0
        return rgba

    def close(self) -> None:
        """Save checkpoint before session ends."""
        if self._bridge is not None and self._bridge.available:
            self._bridge.save_checkpoint()
            logger.info("Session checkpoint saved")

    def __del__(self) -> None:
        self.close()
