"""OmenSession — Pipeline orchestrator for a single render.

Owns the backend, sync, and optional JEPA denoising pass.
Created per-render by the Blender engine callback.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omen_engine.backends import Backend
from omen_engine.sync import OmenSync

logger = logging.getLogger(__name__)


class OmenSession:
    """One-shot render session: sync → trace → denoise → output."""

    def __init__(self, backend: Backend | None = None) -> None:
        self._sync = OmenSync()
        self._backend = backend or self._default_backend()

    @staticmethod
    def _default_backend() -> Backend:
        from omen_engine.backends.mitsuba_backend import MitsubaBackend
        return MitsubaBackend()

    def render_scene(
        self,
        depsgraph: Any,
        spp: int = 4,
        max_depth: int = 8,
    ) -> NDArray[np.float32]:
        """Full pipeline: sync depsgraph → render → (optional denoise)."""
        scene_data = self._sync.sync(depsgraph)
        self._backend.load_scene(
            vertices=scene_data["vertices"],
            faces=scene_data["faces"],
            camera_matrix=scene_data["camera_matrix"],
            camera_fov=scene_data["camera_fov"],
            width=scene_data["width"],
            height=scene_data["height"],
            lights=scene_data["lights"],
        )
        aovs = self._backend.render(spp=spp, max_depth=max_depth)
        color = aovs.get("color", np.zeros((1, 1, 3), dtype=np.float32))

        if color.shape[-1] == 4:
            color = color[:, :, :3]

        return np.clip(color, 0.0, 1.0)

    def render_tile(
        self,
        depsgraph: Any,
        spp: int = 4,
        max_depth: int = 8,
        tile_x: int = 0,
        tile_y: int = 0,
        tile_w: int = 0,
        tile_h: int = 0,
    ) -> NDArray[np.float32]:
        """Render a tile region. Falls back to full render + crop."""
        full = self.render_scene(depsgraph, spp, max_depth)
        if tile_w > 0 and tile_h > 0:
            h, w = full.shape[:2]
            x2 = min(tile_x + tile_w, w)
            y2 = min(tile_y + tile_h, h)
            return full[tile_y:y2, tile_x:x2, :]
        return full
