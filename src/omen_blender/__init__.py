"""Omen Render Engine - Lightweight Blender addon connector.

No mitsuba/nabla/mojo/pixi required inside Blender.
Calls Omen via subprocess for JEPA-accelerated path tracing.
"""

bl_info = {
    "name": "Omen Render Engine",
    "author": "Omen Team",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "Info Header, render engine menu",
    "description": "JEPA-accelerated path tracing via Omen",
    "warning": "Experimental",
    "wiki_url": "",
    "category": "Render",
}


def register():
    """Register Omen render engine and all UI panels."""
    import bpy
    from omen_blender.engine import OmenRenderEngine
    from omen_blender.properties import (
        OmenSettings,
        OMEN_RENDER_PT_settings,
        OMEN_RENDER_PT_performance,
    )
    bpy.utils.register_class(OmenRenderEngine)
    bpy.utils.register_class(OmenSettings)
    bpy.utils.register_class(OMEN_RENDER_PT_settings)
    bpy.utils.register_class(OMEN_RENDER_PT_performance)
    bpy.types.Scene.omen = bpy.props.PointerProperty(type=OmenSettings)


def unregister():
    """Unregister Omen render engine and all UI panels."""
    import bpy
    from omen_blender.engine import OmenRenderEngine
    from omen_blender.properties import (
        OmenSettings,
        OMEN_RENDER_PT_settings,
        OMEN_RENDER_PT_performance,
    )
    bpy.utils.unregister_class(OmenRenderEngine)
    bpy.utils.unregister_class(OmenSettings)
    bpy.utils.unregister_class(OMEN_RENDER_PT_settings)
    bpy.utils.unregister_class(OMEN_RENDER_PT_performance)
    del bpy.types.Scene.omen
