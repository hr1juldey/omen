"""Blender-to-Mitsuba scene converter.

Converts Blender scenes loaded via bpy (headless) into Mitsuba scene dicts
suitable for Omen rendering. Handles geometry, materials, lights, camera,
textures, modifiers, hair, and volumetrics.
"""

from omen.converter.blend_to_mitsuba import convert_scene

__all__ = ["convert_scene"]
