"""Complex scene generators for Omen testing and demos.

Each scene is a Mitsuba dict designed to stress-test specific JEPA capabilities:
MoE expert routing (material/light/geometry diversity), MLA skip compression,
temporal coherence, and motion blur handling.
"""

import numpy as np


def create_e_shaped_room(width: float = 10.0, height: float = 3.0) -> dict:
    """E-shaped room with diverse materials and complex lighting.

    Layout (top view):
    ┌────────────────────┐
    │  ██████            │  <- top wing (mirror floor)
    │  ██████   ┌───┐    │
    │           │   │    │  <- center stem (glass table)
    │  ██████   │   │    │
    │  ██████   └───┘    │  <- bottom wing (colored walls)
    └────────────────────┘

    Materials: mirror, glass, diffuse RGB, metal, SSS-like
    Lights: point light (warm), area light (cool), spot (focused)
    """
    # Floor (mirror)
    floor = _make_rect(10, 10, [0, 0, 0], [0.95, 0.95, 0.95],
                       roughness=0.02, metallic=1.0)

    # Ceiling (white diffuse)
    ceiling = _make_rect(10, 10, [0, 3, 0], [0.9, 0.9, 0.9], flip=True)

    # E-shape walls — 5 wall segments with different materials
    walls = {
        "wall_top_left": _make_wall(4, 3, [-3, 0, -5], [0.8, 0.2, 0.2], "x"),
        "wall_top_right": _make_wall(4, 3, [3, 0, -5], [0.2, 0.8, 0.2], "x"),
        "wall_bottom": _make_wall(10, 3, [0, 0, 5], [0.2, 0.2, 0.8], "x"),
        "wall_left": _make_wall(10, 3, [-5, 0, 0], [0.9, 0.9, 0.7], "z"),
        "wall_right": _make_wall(10, 3, [5, 0, 0], [0.7, 0.9, 0.9], "z"),
        "stem_left": _make_wall(2, 3, [1, 0, -1], [0.6, 0.4, 0.8], "z"),
        "stem_right": _make_wall(2, 3, [-1, 0, -1], [0.4, 0.8, 0.6], "z"),
    }

    # Glass table in center
    table = _make_rect(2, 2, [0, 1.0, 0], [0.9, 0.95, 1.0],
                       roughness=0.0, ior=1.5, transmission=1.0)

    # Metallic sphere on table
    sphere = {
        "type": "sphere", "center": [0, 1.5, 0], "radius": 0.4,
        "bsdf": {"type": "roughconductor", "material": "Cu",
                 "alpha": 0.05},
    }

    # Lights: warm point + cool area + focused spot
    lights = {
        "warm_point": {"type": "point", "position": [-3, 2.8, -3],
                       "intensity": [15.0, 10.0, 6.0]},
        "cool_area": {"type": "rectangle",
                      "to_world": _translate_scale(2, 2, [3, 2.9, 3]),
                      "radiance": [6.0, 8.0, 12.0]},
        "spot": {"type": "spot", "position": [0, 2.9, 0],
                 "direction": [0, -1, 0], "cutoff_angle": 15,
                 "intensity": [20.0, 18.0, 15.0]},
    }

    scene = {"type": "scene", "integrator": {"type": "path", "max_depth": 6}}
    scene["floor"] = floor
    scene["ceiling"] = ceiling
    for name, wall in walls.items():
        scene[name] = wall
    scene["table"] = table
    scene["sphere"] = sphere
    for name, light in lights.items():
        scene[name] = light
    scene["sensor"] = _default_sensor([0, 1.5, 4], [0, 1.5, 0])
    return scene


def create_c_shaped_room() -> dict:
    """C-shaped room with caustics, volumes, and hair-like geometry.

    Layout (top view):
    ┌──────────┐
    │          │
    │    ┌─┐   │  <- glass panel (creates caustics)
    │    │ │   │
    │    └─┘   │
    │          │
    └────  ────┘  <- gap in bottom wall (open doorway)
    """
    scene = {"type": "scene", "integrator": {"type": "path", "max_depth": 8}}

    # Dark floor with checkerboard pattern (implicit)
    scene["floor"] = _make_rect(12, 12, [0, 0, 0], [0.3, 0.3, 0.3])
    scene["ceiling"] = _make_rect(12, 12, [0, 3.5, 0], [0.8, 0.8, 0.8], flip=True)

    # C-shape walls
    scene["wall_top"] = _make_wall(12, 3.5, [0, 0, -6], [0.85, 0.85, 0.75], "x")
    scene["wall_left"] = _make_wall(12, 3.5, [-6, 0, 0], [0.75, 0.85, 0.85], "z")
    scene["wall_right"] = _make_wall(12, 3.5, [6, 0, 0], [0.85, 0.75, 0.85], "z")
    scene["wall_bottom_left"] = _make_wall(4, 3.5, [-4, 0, 6], [0.8, 0.8, 0.8], "x")
    scene["wall_bottom_right"] = _make_wall(4, 3.5, [4, 0, 6], [0.8, 0.8, 0.8], "x")

    # Glass panel in center (caustics source)
    scene["glass_panel"] = {
        "type": "rectangle",
        "to_world": _translate_scale(3, 3, [0, 1.75, -2]),
        "bsdf": {"type": "roughdielectric", "alpha": 0.001,
                 "int_ior": 1.5, "ext_ior": 1.0},
    }

    # Colored emissive sphere (neon sign effect)
    scene["neon_sphere"] = {
        "type": "sphere", "center": [-3, 2.5, -4], "radius": 0.3,
        "bsdf": {"type": "diffuse",
                 "reflectance": {"type": "srgb", "value": [0.1, 0.1, 0.1]},
                 "emission": {"type": "srgb", "value": [5.0, 0.5, 8.0]}},
    }

    # Metallic torus (complex geometry)
    scene["ring"] = {
        "type": "sphere", "center": [3, 0.5, 2], "radius": 0.5,
        "bsdf": {"type": "roughconductor", "material": "Au", "alpha": 0.1},
    }

    # Lighting: strong sunlight through gap + fill light
    scene["sun"] = {"type": "directional", "direction": [-0.3, -1, 0.5],
                    "irradiance": [8.0, 7.5, 6.0]}
    scene["fill"] = {"type": "point", "position": [0, 3, 0],
                     "intensity": [3.0, 3.0, 4.0]}

    scene["sensor"] = _default_sensor([0, 2, 8], [0, 1.5, 0])
    return scene


def _make_rect(sx, sz, pos, color, roughness=0.8, metallic=0.0,
               flip=False, ior=1.5, transmission=0.0):
    """Create a horizontal rectangle (floor/ceiling/table)."""
    bsdf = {"type": "roughplastic", "alpha": roughness,
            "diffuse_reflectance": {"type": "srgb", "value": list(color)}}
    if transmission > 0.5:
        bsdf = {"type": "roughdielectric", "alpha": roughness,
                "int_ior": ior, "ext_ior": 1.0}
    elif metallic > 0.5:
        bsdf = {"type": "roughconductor", "material": "Al", "alpha": roughness}
    transform = list(np.eye(4).flatten())
    transform[3 * 4 + 0] = pos[0]
    transform[3 * 4 + 1] = pos[1]
    transform[3 * 4 + 2] = pos[2]
    transform[0] = sx
    transform[10] = sz
    if flip:
        transform[5] = -1
    return {"type": "rectangle", "to_world": {"type": "matrix", "value": transform},
            "bsdf": bsdf}


def _make_wall(sx, h, pos, color, axis="x"):
    """Create a vertical wall rectangle."""
    bsdf = {"type": "diffuse",
            "reflectance": {"type": "srgb", "value": list(color)}}
    mat = np.eye(4)
    mat[0, 0] = sx
    mat[1, 1] = h
    mat[3, :3] = pos
    if axis == "z":
        mat = mat @ np.array([[0, 0, 1, 0], [0, 1, 0, 0],
                               [-1, 0, 0, 0], [0, 0, 0, 1]])
    return {"type": "rectangle",
            "to_world": {"type": "matrix", "value": mat.T.flatten().tolist()},
            "bsdf": bsdf}


def _translate_scale(sx, sy, pos):
    """Simple translate + scale transform for area lights."""
    mat = np.eye(4)
    mat[0, 0], mat[1, 1] = sx, sy
    mat[3, :3] = pos
    return {"type": "matrix", "value": mat.T.flatten().tolist()}


def _default_sensor(origin, target):
    """Default perspective camera."""
    import mitsuba as mi
    mi.set_variant("scalar_rgb")
    transform = mi.ScalarTransform4f.look_at(
        origin=origin, target=target, up=[0, 1, 0])
    return {
        "type": "perspective", "fov": 45, "fov_axis": "x",
        "to_world": transform,
        "film": {"type": "hdrfilm", "width": 512, "height": 512,
                 "pixel_format": "rgba", "component_format": "float32"},
        "sampler": {"type": "independent"},
    }
