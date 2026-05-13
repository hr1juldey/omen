"""Material-to-Mitsuba BSDF converter.

Converts material dicts (from depsgraph sync) into Mitsuba BSDF
objects. Uses Mitsuba 3 plugin names: diffuse, conductor, twosided.
"""

from typing import Any


def material_to_bsdf(mat: dict[str, Any]) -> Any:
    """Convert a material dict to a Mitsuba BSDF.

    Dispatches on metallic flag and emission strength.
    """
    import mitsuba as mi

    metallic = mat.get("metallic", 0.0)
    if metallic > 0.5:
        return _conductor_bsdf(mat, mi)

    return _diffuse_bsdf(mat, mi)


def material_is_emissive(mat: dict[str, Any]) -> bool:
    """Check if material has significant emission."""
    emission = mat.get("emission", [0, 0, 0])
    return max(emission) > 0.01


def build_emitter(mat: dict[str, Any]) -> Any:
    """Build an area emitter for emissive materials."""
    import mitsuba as mi
    emission = mat.get("emission", [0, 0, 0])
    return mi.load_dict({
        "type": "area",
        "radiance": {
            "type": "uniform",
            "value": mi.ScalarColor3f(*emission),
        },
    })


def _diffuse_bsdf(mat: dict[str, Any], mi: Any) -> Any:
    color = mat.get("base_color", [0.8, 0.8, 0.8])
    return mi.load_dict({
        "type": "diffuse",
        "reflectance": {
            "type": "srgb",
            "color": mi.ScalarColor3f(*color),
        },
    })


def _conductor_bsdf(mat: dict[str, Any], mi: Any) -> Any:
    return mi.load_dict({
        "type": "roughconductor",
        "alpha": max(mat.get("roughness", 0.5), 0.01),
        "material": "Al",
    })
