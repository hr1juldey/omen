"""Material-to-Mitsuba BSDF converter.

Converts material dicts (from depsgraph sync) into Mitsuba BSDF
objects: roughdiffuse, roughconductor, or emitter.
"""

from typing import Any


def material_to_bsdf(mat: dict[str, Any]) -> Any:
    """Convert a material dict to a Mitsuba BSDF.

    Dispatches on metallic flag and emission strength.
    """
    import mitsuba as mi

    emission = mat.get("emission", [0, 0, 0])
    if max(emission) > 0.01:
        return _emitter_bsdf(emission, mi)

    metallic = mat.get("metallic", 0.0)
    if metallic > 0.5:
        return _conductor_bsdf(mat, mi)

    return _diffuse_bsdf(mat, mi)


def _emitter_bsdf(emission: list[float], mi: Any) -> Any:
    return mi.load_dict({
        "type": "two-sided",
        "material": {
            "type": "emitter",
            "radiance": {
                "type": "srgb",
                "color": mi.ScalarColor3f(*emission),
            },
        },
    })


def _conductor_bsdf(mat: dict[str, Any], mi: Any) -> Any:
    roughness = max(mat.get("roughness", 0.5), 0.01)
    return mi.load_dict({
        "type": "roughconductor",
        "alpha": roughness,
        "material": "Al",
    })


def _diffuse_bsdf(mat: dict[str, Any], mi: Any) -> Any:
    color = mat.get("base_color", [0.8, 0.8, 0.8])
    roughness = max(mat.get("roughness", 0.5), 0.01)
    return mi.load_dict({
        "type": "roughdiffuse",
        "reflectance": {
            "type": "srgb",
            "color": mi.ScalarColor3f(*color),
        },
        "alpha": roughness,
    })
