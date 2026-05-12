"""Omen Render Engine - Blender addon entry point.

Blender 5.1+ render engine built with Mojo for GPU compute
and JEPA-based scene analysis.
"""

bl_info = {
    "name": "Omen Render Engine",
    "author": "Omen Contributors",
    "version": (0, 1, 0),
    "blender": (5, 1, 0),
    "location": "Render Properties > Render Engine",
    "description": "JEPA-accelerated path tracing render engine",
    "category": "Render",
}


import sys
from pathlib import Path


def _add_src_to_path() -> None:
    """Add src/ directory to Python path for absolute imports."""
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def register() -> None:
    """Register Omen addon with Blender."""
    _add_src_to_path()
    from src.python import register as omen_register

    omen_register()


def unregister() -> None:
    """Unregister Omen addon from Blender."""
    from src.python import unregister as omen_unregister

    omen_unregister()
