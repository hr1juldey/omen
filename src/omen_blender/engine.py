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

        spp = omen.spp
        max_depth = omen.max_depth
        tile_size = omen.tile_size

        if tile_size > 0:
            self._render_tiled(session, depsgraph, width, height,
                               spp, max_depth, tile_size)
        else:
            self._render_full(session, depsgraph, width, height,
                              spp, max_depth)

    def _render_full(self, session, depsgraph, w, h, spp, max_depth):
        pixels = session.render_scene(depsgraph, spp=spp, max_depth=max_depth)
        result = self.begin_result(0, 0, w, h)
        layer = result.layers[0].passes["Combined"]
        flat = self._to_rgba(pixels, w, h)
        layer.rect = flat
        self.end_result(result)

    def _render_tiled(self, session, depsgraph, w, h, spp, max_depth, ts):
        y = 0
        while y < h:
            x = 0
            while x < w:
                tw = min(ts, w - x)
                th = min(ts, h - y)
                pixels = session.render_tile(
                    depsgraph, spp=spp, max_depth=max_depth,
                    tile_x=x, tile_y=y, tile_w=tw, tile_h=th,
                )
                result = self.begin_result(x, y, tw, th)
                layer = result.layers[0].passes["Combined"]
                layer.rect = self._to_rgba(pixels, tw, th)
                self.end_result(result)
                x += ts
            y += ts
            pct = min(100, int(100.0 * y / h))
            self.update_progress(pct / 100.0)

    @staticmethod
    def _to_rgba(pixels, w, h):
        """Convert (H,W,3) float32 to Blender's (H*W, 4) list-of-tuples."""
        import numpy as np
        arr = np.zeros((h, w, 4), dtype=np.float32)
        arr[:, :, :3] = pixels[:h, :w, :3]
        arr[:, :, 3] = 1.0
        return arr.reshape(-1, 4).tolist()
