"""Backend abstraction for pluggable path tracers."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Backend(ABC):
    """Abstract base for render backends (Mitsuba, Cycles, LuxCore, etc.).

    Each backend receives scene data as numpy arrays and returns
    rendered AOV buffers as numpy arrays. No file I/O.
    """

    @abstractmethod
    def load_scene(
        self,
        vertices: NDArray[np.float32],
        faces: NDArray[np.int32],
        camera_matrix: NDArray[np.float32],
        camera_fov: float,
        width: int,
        height: int,
        lights: list[dict[str, Any]],
    ) -> None:
        """Build an internal scene from raw geometry data."""

    @abstractmethod
    def render(self, spp: int, max_depth: int) -> dict[str, NDArray[np.float32]]:
        """Render the loaded scene. Returns AOV dict with keys:
        'color', 'albedo', 'normal', 'depth'."""

    @abstractmethod
    def get_aov_buffers(self) -> dict[str, NDArray[np.float32]]:
        """Return the last render's AOV buffers."""
