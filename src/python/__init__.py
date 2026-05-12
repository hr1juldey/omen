"""Omen render engine Python module."""

import bpy
from src.python.render_engine import OmenRenderEngine


__all__ = ["register", "unregister", "OmenRenderEngine"]


def register() -> None:
    """Register Omen render engine with Blender."""
    bpy.utils.register_class(OmenRenderEngine)


def unregister() -> None:
    """Unregister Omen render engine from Blender."""
    bpy.utils.unregister_class(OmenRenderEngine)
