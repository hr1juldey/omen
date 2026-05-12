"""Omen render engine implementation for Blender."""

import bpy
from src.python.test_pattern import generate_gradient


class OmenRenderEngine(bpy.types.RenderEngine):
    """Omen render engine - JEPA-accelerated path tracing."""

    bl_idname = "OMEN"
    bl_label = "Omen"
    bl_use_preview = True

    def render(self, depsgraph: bpy.types.Depsgraph) -> None:
        """Render callback invoked by Blender.

        Args:
            depsgraph: Blender dependency graph containing scene data
        """
        width, height = self._get_dimensions(depsgraph)
        pixels = generate_gradient(width, height)

        result = self.begin_result(0, 0, width, height)
        result.layers[0].passes["Combined"].rect = pixels
        self.end_result(result)

    def _get_dimensions(self, depsgraph: bpy.types.Depsgraph) -> tuple[int, int]:
        """Extract render dimensions from dependency graph.

        Args:
            depsgraph: Blender dependency graph

        Returns:
            Tuple of (width, height) in pixels
        """
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)
        return width, height
