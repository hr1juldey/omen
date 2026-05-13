"""Material converter: Blender BSDFs to Mitsuba equivalents.

Tasks 15.3-15.4: Principled, Glass, Emission, Diffuse BSDF conversion.
Task 15.6: Texture map extraction (delegates to texture_handler).
"""

import logging

import numpy as np

logger = logging.getLogger("omen.converter.material_converter")

# Blender node socket type -> value extractor
_SOCKET_TYPES = {
    "VALUE": lambda s: s.default_value, "RGBA": lambda s: list(s.default_value),
    "VECTOR": lambda s: list(s.default_value), "INT": lambda s: int(s.default_value),
    "BOOLEAN": lambda s: bool(s.default_value), "FLOAT": lambda s: float(s.default_value),
}


def convert_material(mat) -> dict | None:
    """Convert a Blender material to Mitsuba BSDF dict."""
    if not mat or not mat.node_tree:
        return _fallback_diffuse()

    # Find output node
    output_node = None
    for node in mat.node_tree.nodes:
        if node.type == "OUTPUT_MATERIAL":
            output_node = node
            break
    if not output_node:
        return _fallback_diffuse()

    # Trace linked input to find BSDF
    surface_input = output_node.inputs.get("Surface")
    if not surface_input or not surface_input.links:
        return _fallback_diffuse()

    bsdf_node = surface_input.links[0].from_node
    return _convert_bsdf_node(bsdf_node, mat.node_tree)


def _convert_bsdf_node(node, node_tree) -> dict | None:
    """Dispatch BSDF conversion by node type."""
    if node.type == "BSDF_PRINCIPLED":
        return _convert_principled(node)
    elif node.type == "BSDF_GLASS":
        return _convert_glass(node)
    elif node.type == "EMISSION":
        return _convert_emission(node)
    elif node.type == "BSDF_DIFFUSE":
        return _convert_diffuse(node)
    else:
        logger.warning("Unsupported BSDF type: %s, using diffuse", node.type)
        return _fallback_diffuse()


def _convert_principled(node) -> dict:
    """Convert Blender Principled BSDF -> Mitsuba roughplastic/conductor."""
    base_color = _get_input(node, "Base Color", [0.8, 0.8, 0.8])
    roughness = _get_input(node, "Roughness", 0.5)
    metallic = _get_input(node, "Metallic", 0.0)
    specular = _get_input(node, "Specular IOR Level", 0.5)
    transmission = _get_input(node, "Transmission Weight", 0.0)
    clearcoat = _get_input(node, "Clearcoat Weight", 0.0)

    if metallic > 0.5:
        return {
            "type": "roughconductor",
            "material": "Al",
            "alpha": max(roughness, 0.01),
            "specular_reflectance": _to_srgb(base_color),
        }
    elif transmission > 0.5:
        return {
            "type": "roughdielectric",
            "alpha": max(roughness, 0.01),
            "int_ior": 1.5,
            "ext_ior": 1.0,
        }
    else:
        result = {
            "type": "roughplastic",
            "alpha": max(roughness, 0.01),
            "diffuse_reflectance": _to_srgb(base_color),
        }
        if clearcoat > 0.01:
            result["clearcoat"] = float(clearcoat)
        return result


def _convert_glass(node) -> dict:
    """Convert Blender Glass BSDF -> Mitsuba roughdielectric."""
    color = _get_input(node, "Color", [1.0, 1.0, 1.0])
    roughness = _get_input(node, "Roughness", 0.0)
    ior = _get_input(node, "IOR", 1.5)
    return {
        "type": "roughdielectric",
        "alpha": max(float(roughness), 0.001),
        "int_ior": float(ior),
        "ext_ior": 1.0,
        "specular_transmittance": _to_srgb(color),
    }


def _convert_emission(node) -> dict:
    """Convert Blender Emission -> Mitsuba emissive material."""
    color = _get_input(node, "Color", [1.0, 1.0, 1.0])
    strength = _get_input(node, "Strength", 1.0)
    radiance = [c * float(strength) for c in color]
    return {"type": "diffuse", "reflectance": _to_srgb([0.0, 0.0, 0.0]),
            "emission": _to_srgb(radiance)}


def _convert_diffuse(node) -> dict:
    """Convert Blender Diffuse BSDF -> Mitsuba diffuse."""
    color = _get_input(node, "Color", [0.8, 0.8, 0.8])
    return {"type": "diffuse", "reflectance": _to_srgb(color)}


def _get_input(node, name, default):
    """Extract input value from a Blender node socket."""
    inp = node.inputs.get(name)
    if inp is None:
        return default
    if inp.links:
        from omen.converter.texture_handler import resolve_texture_link
        result = resolve_texture_link(inp.links[0], default)
        if result is not None:
            return result
    socket_type = inp.type
    if socket_type in _SOCKET_TYPES:
        try:
            return _SOCKET_TYPES[socket_type](inp)
        except Exception:
            return default
    return default


def _to_srgb(color) -> dict:
    """Convert color list to Mitsuba srgb dict."""
    if isinstance(color, (int, float)):
        color = [color, color, color]
    return {"type": "srgb", "value": [float(c) for c in color[:3]]}


def _fallback_diffuse() -> dict:
    """Default grey diffuse material."""
    return {"type": "diffuse", "reflectance": {"type": "srgb", "value": [0.5, 0.5, 0.5]}}
