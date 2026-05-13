"""OmenRenderEngine — Blender RenderEngine subclass.

Delegates all rendering to omen_engine.OmenSession.
Thin wrapper: no scene logic lives here.
"""

import logging

import bpy

log = logging.getLogger(__name__)


class OmenRenderEngine(bpy.types.RenderEngine):
    """Omen render engine registered with Blender."""

    bl_idname = "OMEN"
    bl_label = "Omen"
    bl_use_preview = False
    bl_use_shading_nodes = False
    bl_use_eevee_viewport = False

    def render(self, depsgraph):
        """Called by Blender on F12 / render animation."""
        scene = depsgraph.scene
        omen = scene.omen

        from omen_blender.bridge import get_session
        session = get_session()

        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)

        pixels = session.render_scene(
            depsgraph,
            spp=omen.spp,
            max_depth=omen.max_depth,
            tier=omen.tier,
            mode=omen.mode,
        )

        result = self.begin_result(0, 0, width, height)
        layer = result.layers[0].passes["Combined"]
        layer.rect = self._to_rgba(pixels, width, height)
        self.end_result(result)

    @staticmethod
    def _to_rgba(pixels, w, h):
        """Convert (H,W,4) float32 to Blender's (H*W, 4) list-of-tuples."""
        import numpy as np
        arr = np.zeros((h, w, 4), dtype=np.float32)
        arr[:, :, :3] = pixels[:h, :w, :3]
        arr[:, :, 3] = 1.0
        return arr.reshape(-1, 4).tolist()
