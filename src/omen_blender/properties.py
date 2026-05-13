"""Omen render engine properties and UI panels.

Scene-level settings for Omen: render mode, tier, SPP, tile size.
No pixi path — engine is loaded in-process via omen_engine package.
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty
from bpy.types import Panel, PropertyGroup


class OmenSettings(PropertyGroup):
    """Scene-level Omen render settings attached to bpy.types.Scene."""

    mode: EnumProperty(
        name="Mode",
        items=[
            ("denoise", "Denoiser", "Low spp + JEPA denoise"),
            ("adaptive", "Adaptive", "Confidence-guided multi-pass"),
            ("multires", "Multi-Res", "Low-res clean + high-res noisy merge"),
            ("path", "Path Only", "Standard path tracing, no JEPA"),
        ],
        default="denoise",
    )

    tier: EnumProperty(
        name="Model Tier",
        items=[
            ("fast", "Fast (~4M)", "~5ms, basic denoise"),
            ("medium", "Medium (~16M)", "~15ms, balanced quality"),
            ("high", "High (~64M)", "~50ms, best quality"),
        ],
        default="medium",
    )

    spp: IntProperty(
        name="Samples Per Pixel",
        description="Base spp for the noisy render pass",
        default=4,
        min=1,
        max=4096,
    )

    max_depth: IntProperty(
        name="Max Bounces",
        default=8,
        min=1,
        max=64,
    )

    tile_size: IntProperty(
        name="Tile Size",
        description="Render tile size in pixels (0 = no tiling)",
        default=0,
        min=0,
        max=512,
    )

    use_gpu: BoolProperty(
        name="GPU Acceleration",
        default=True,
    )


class OMEN_RENDER_PT_settings(Panel):
    """Main Omen settings panel in render properties."""

    bl_idname = "OMEN_RENDER_PT_settings"
    bl_label = "Omen Settings"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    COMPAT_ENGINES = {"OMEN"}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        omen = context.scene.omen
        layout.prop(omen, "mode")
        layout.prop(omen, "tier")
        layout.prop(omen, "spp")
        layout.prop(omen, "max_depth")
        layout.prop(omen, "tile_size")


class OMEN_RENDER_PT_performance(Panel):
    """Performance settings panel."""

    bl_idname = "OMEN_RENDER_PT_performance"
    bl_label = "Performance"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    COMPAT_ENGINES = {"OMEN"}
    bl_parent_id = "OMEN_RENDER_PT_settings"

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        omen = context.scene.omen
        layout.prop(omen, "use_gpu")
