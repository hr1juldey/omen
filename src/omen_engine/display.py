"""Display driver placeholder for future viewport rendering.

Will provide live render preview in Blender's 3D viewport
using OpenGL interop for zero-copy texture updates.
"""


class ViewportDisplay:
    """Future: real-time viewport display via GL texture."""

    def __init__(self) -> None:
        self._texture_id: int = 0
        self._width: int = 0
        self._height: int = 0

    def update(self, pixels, width: int, height: int) -> None:
        """Upload pixel buffer to GPU texture."""
        self._width = width
        self._height = height
        # TODO: implement GL texture upload

    def draw(self) -> None:
        """Draw texture to viewport framebuffer."""
        # TODO: implement GL draw call
