"""Omen Render Engine - Mojo/Nabla-accelerated path tracing for Blender.

Tier 1 in-process integration: same process, same GPU, zero-copy.
Mojo .so kernels loaded via ctypes. Modular nightly provides runtime.
"""

import os

bl_info = {
    "name": "Omen Render Engine",
    "author": "Omen Team",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "Info Header, render engine menu",
    "description": "JEPA-accelerated path tracing via Mojo/Nabla",
    "warning": "Experimental",
    "wiki_url": "",
    "category": "Render",
}

_CLASSES = []


def _setup_mojo_runtime():
    """Set LD_LIBRARY_PATH to modular/lib/ for Mojo .so loading."""
    try:
        from importlib.metadata import version
        mod_ver = version("modular")
        if ".dev" not in mod_ver:
            print(f"Omen WARNING: modular {mod_ver} is stable, not nightly.")
            print("  Nabla requires nightly: pip install --pre modular")
            print("  --index https://whl.modular.com/nightly/simple/")

        import importlib.util
        spec = importlib.util.find_spec("modular")
        if spec and spec.submodule_search_locations:
            for search_path in spec.submodule_search_locations:
                lib_dir = os.path.join(search_path, "lib")
                if os.path.isdir(lib_dir):
                    existing = os.environ.get("LD_LIBRARY_PATH", "")
                    if lib_dir not in existing:
                        os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + existing
                    break
    except Exception:
        pass


def register():
    """Register Omen render engine and all UI panels."""
    _setup_mojo_runtime()

    import bpy
    from omen_blender.engine import OmenRenderEngine
    from omen_blender.properties import (
        OmenSettings,
        OMEN_RENDER_PT_settings,
        OMEN_RENDER_PT_performance,
    )

    _CLASSES.clear()
    _CLASSES.extend([
        OmenRenderEngine,
        OmenSettings,
        OMEN_RENDER_PT_settings,
        OMEN_RENDER_PT_performance,
    ])

    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.omen = bpy.props.PointerProperty(type=OmenSettings)


def unregister():
    """Unregister Omen render engine and all UI panels."""
    import bpy

    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    _CLASSES.clear()

    if hasattr(bpy.types.Scene, "omen"):
        del bpy.types.Scene.omen
