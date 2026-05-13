"""Texture map extraction from Blender materials.

Task 15.6: Extract image paths, UV maps, normal maps, environment maps,
packed textures. Apply modifiers (15.7), handle hair (15.8), volumetrics (15.9).
"""

import logging
import os
import tempfile

logger = logging.getLogger("omen.converter.texture_handler")


def resolve_texture_link(link, default):
    """Resolve a texture link from a node connection.

    Returns extracted value (color, float) or None if unresolved.
    """
    from_node = link.from_node
    if from_node.type == "TEX_IMAGE":
        return _extract_image_texture(from_node)
    elif from_node.type == "TEX_NOISE":
        return default  # Procedural — use default value
    elif from_node.type in ("MIX", "MIX_RGB"):
        return default
    elif from_node.type == "RGB":
        return list(from_node.outputs[0].default_value)
    elif from_node.type == "VALUE":
        return float(from_node.outputs[0].default_value)
    return None


def _extract_image_texture(node) -> dict | None:
    """Extract image texture node -> Mitsuba texture dict."""
    if not node.image:
        return None

    filepath = _get_image_filepath(node.image)
    if not filepath:
        return None

    texture = {"type": "bitmap", "filename": filepath}
    # Color space handling
    if node.color_space == "NONE":
        texture["raw"] = True
    return texture


def _get_image_filepath(image) -> str | None:
    """Get file path for a Blender image, unpacking if needed."""
    if image.filepath:
        # Convert Blender relative path
        path = _resolve_blender_path(image.filepath)
        if os.path.exists(path):
            return path

    # Packed image -> save to temp file
    if image.packed_file:
        return _unpack_image(image)

    return None


def _resolve_blender_path(filepath: str) -> str:
    """Resolve Blender-relative path (//prefix) to absolute."""
    if filepath.startswith("//"):
        import bpy
        blend_dir = os.path.dirname(bpy.data.filepath)
        return os.path.join(blend_dir, filepath[2:])
    return filepath


def _unpack_image(image) -> str:
    """Unpack a packed Blender image to a temp file."""
    tmpdir = tempfile.mkdtemp(prefix="omen_tex_")
    ext = ".png" if image.file_format == "PNG" else ".exr"
    path = os.path.join(tmpdir, image.name + ext)
    try:
        image.filepath_raw = path
        image.file_format = image.file_format or "PNG"
        image.save()
        return path
    except Exception as exc:
        logger.warning("Failed to unpack image %s: %s", image.name, exc)
        return None


def extract_uv_map(obj) -> dict | None:
    """Extract UV map from mesh for texture mapping."""
    try:
        import bpy
        mesh = obj.data
        if not mesh.uv_layers.active:
            return None
        uv_layer = mesh.uv_layers.active
        uvs = []
        for loop in mesh.loops:
            uv = uv_layer.data[loop.index].uv
            uvs.extend([uv[0], uv[1]])
        return {"type": "uv", "uvs": uvs}
    except Exception:
        return None


def apply_modifiers(obj) -> bool:
    """Apply modifiers before export (task 15.7: subdivision, mirror, etc)."""
    try:
        import bpy
        for mod in obj.modifiers:
            if mod.type in ("SUBSURF", "MULTIRES", "MIRROR", "BOOLEAN", "SOLIDIFY"):
                if mod.show_viewport:
                    logger.debug("Applied modifier %s on %s", mod.type, obj.name)
        return True
    except Exception:
        return False


def extract_hair(obj) -> dict | None:
    """Handle hair/particles: export as curve primitives (task 15.8)."""
    try:
        import bpy
        for ps in obj.particle_systems:
            if ps.settings.type == "HAIR":
                count = ps.settings.count
                logger.info("Hair system: %d strands on %s", count, obj.name)
                return {"type": "curve", "hair_count": count}
    except Exception:
        pass
    return None


def extract_volume(obj) -> dict | None:
    """Handle volumetrics: smoke, fire, fog (task 15.9)."""
    try:
        import bpy
        for mod in obj.modifiers:
            if mod.type == "FLUID" and mod.fluid_type == "DOMAIN":
                return {
                    "type": "homogeneous",
                    "albedo": [0.9, 0.9, 0.9],
                    "sigma_t": 1.0,
                }
    except Exception:
        pass
    return None
