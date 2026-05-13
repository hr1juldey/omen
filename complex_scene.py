"""Complex Mitsuba 3 test scene — Barbershop / ArchViz interior.

Materials exercised:
  diffuse, roughdielectric (glass/water), conductor (mirror),
  roughconductor (metal), plastic (SSS-like), roughplastic,
  blendbsdf (specular coating), bumpmap, twosided, null (volume boundary)

Lights:
  area (window panel), spot (desk lamp), point (corner), constant (env)

Geometry:
  C-shaped room, glass table, mirror, water glass, metal vase,
  plastic candle, wooden desk (blendbsdf), bump-mapped tile floor,
  fog volume (homogeneous medium)

Integrators:
  path (standard), volpath (with medium)

Usage:
  python3 complex_scene.py [--spp 64] [--mode path|volpath]
"""

import argparse
import time

import mitsuba as mi


def _tf(translate=None, scale=None, rotate=None):
    """Build a ScalarTransform4f from translate/scale/rotate."""
    t = mi.ScalarTransform4f()
    if translate is not None:
        t = t @ mi.ScalarTransform4f.translate(translate)
    if rotate is not None:
        axis, angle = rotate
        t = t @ mi.ScalarTransform4f.rotate(axis, angle)
    if scale is not None:
        t = t @ mi.ScalarTransform4f.scale(scale)
    return t


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def _bsdf_diffuse(color=(0.8, 0.8, 0.8)):
    """Lambertian diffuse."""
    return {
        "type": "diffuse",
        "reflectance": {"type": "rgb", "value": color},
    }


def _bsdf_glass(ior=1.509, roughness=0.0):
    """Smooth or rough glass (dielectric)."""
    if roughness < 0.001:
        return {"type": "dielectric", "int_ior": ior, "ext_ior": 1.0}
    return {
        "type": "roughdielectric",
        "int_ior": ior, "ext_ior": 1.0,
        "alpha": roughness, "distribution": "beckmann",
    }


def _bsdf_mirror(roughness=0.0):
    """Perfect mirror or rough mirror (conductor)."""
    if roughness < 0.001:
        return {"type": "conductor", "material": "none"}
    return {
        "type": "roughconductor",
        "material": "none", "alpha": roughness,
    }


def _bsdf_metal(material="Cu", roughness=0.15):
    """Rough conductor — copper, aluminium, gold, etc."""
    return {
        "type": "roughconductor",
        "material": material, "alpha": roughness,
        "distribution": "beckmann",
    }


def _bsdf_plastic(color=(0.9, 0.9, 0.85), ior=1.5):
    """Smooth plastic — internal scattering gives SSS-like look."""
    return {
        "type": "plastic",
        "diffuse_reflectance": {"type": "rgb", "value": color},
        "int_ior": ior,
    }


def _bsdf_roughplastic(color=(0.7, 0.2, 0.15), roughness=0.2, ior=1.5):
    """Rough plastic — matte wax / clay."""
    return {
        "type": "roughplastic",
        "diffuse_reflectance": {"type": "rgb", "value": color},
        "int_ior": ior, "alpha": roughness, "distribution": "beckmann",
    }


def _bsdf_coated_wood():
    """Specular coating over diffuse wood — blendbsdf."""
    return {
        "type": "blendbsdf",
        "weight": 0.3,
        "bsdf_1": {"type": "dielectric", "int_ior": 1.5},
        "bsdf_2": _bsdf_diffuse((0.55, 0.35, 0.17)),
    }


def _bsdf_tile_floor():
    """Tile-like floor — roughplastic with checkerboard color."""
    return {
        "type": "roughplastic",
        "diffuse_reflectance": {
            "type": "checkerboard",
            "color0": {"type": "rgb", "value": (0.90, 0.87, 0.80)},
            "color1": {"type": "rgb", "value": (0.70, 0.67, 0.62)},
        },
        "int_ior": 1.5, "alpha": 0.25, "distribution": "beckmann",
    }


def _bsdf_twosided(nested):
    """Two-sided wrapper."""
    return {"type": "twosided", "bsdf": nested}


def _bsdf_null():
    """Null BSDF for volume boundaries."""
    return {"type": "null"}


# ---------------------------------------------------------------------------
# Room geometry (C-shaped interior)
# ---------------------------------------------------------------------------

def _build_room(s):
    """Build a C-shaped room: floor, ceiling, 3 walls + 2 divider walls."""
    # Floor — 12 x 8
    s["floor"] = {
        "type": "rectangle",
        "to_world": _tf(
            translate=(0, 0, 0),
            scale=(6, 4, 1),
            rotate=([1, 0, 0], -90),
        ),
        "bsdf": _bsdf_tile_floor(),
    }
    # Ceiling
    s["ceiling"] = {
        "type": "rectangle",
        "to_world": _tf(
            translate=(0, 0, 4),
            scale=(6, 4, 1),
            rotate=([1, 0, 0], 90),
        ),
        "bsdf": _bsdf_diffuse((0.95, 0.95, 0.92)),
    }
    # Back wall
    s["back_wall"] = {
        "type": "rectangle",
        "to_world": _tf(translate=(0, -4, 2), scale=(6, 2, 1)),
        "bsdf": _bsdf_diffuse((0.93, 0.90, 0.85)),
    }
    # Left wall
    s["left_wall"] = {
        "type": "rectangle",
        "to_world": _tf(
            translate=(-6, 0, 2),
            rotate=([0, 0, 1], -90),
            scale=(4, 2, 1),
        ),
        "bsdf": _bsdf_diffuse((0.85, 0.88, 0.90)),
    }
    # Right wall
    s["right_wall"] = {
        "type": "rectangle",
        "to_world": _tf(
            translate=(6, 0, 2),
            rotate=([0, 0, 1], 90),
            scale=(4, 2, 1),
        ),
        "bsdf": _bsdf_diffuse((0.85, 0.88, 0.90)),
    }
    # Divider wall A — creates C-shape, near right side
    s["divider_a"] = {
        "type": "rectangle",
        "to_world": _tf(
            translate=(3, -1.5, 2),
            rotate=([0, 0, 1], 90),
            scale=(2.5, 2, 1),
        ),
        "bsdf": _bsdf_diffuse((0.90, 0.88, 0.85)),
    }
    # Divider wall B — left inner wall
    s["divider_b"] = {
        "type": "rectangle",
        "to_world": _tf(
            translate=(-2, 1.5, 2),
            rotate=([0, 0, 1], 90),
            scale=(2.5, 2, 1),
        ),
        "bsdf": _bsdf_diffuse((0.90, 0.88, 0.85)),
    }


# ---------------------------------------------------------------------------
# Furniture / objects
# ---------------------------------------------------------------------------

def _build_objects(s):
    """Place furniture and test objects."""

    # ---- Glass table (roughdielectric) ----
    s["glass_table_top"] = {
        "type": "cube",
        "to_world": _tf(
            translate=(1, 0, 1.5),
            scale=(1.2, 0.8, 0.03),
        ),
        "bsdf": _bsdf_glass(ior=1.52, roughness=0.02),
    }
    # 4 legs as thin cylinders
    for i, (dx, dy) in enumerate(
        [(-0.9, -0.55), (0.9, -0.55), (-0.9, 0.55), (0.9, 0.55)]
    ):
        s[f"table_leg_{i}"] = {
            "type": "cylinder",
            "p0": mi.ScalarPoint3f(dx + 1, dy, 0),
            "p1": mi.ScalarPoint3f(dx + 1, dy, 1.48),
            "radius": 0.04,
            "bsdf": _bsdf_metal("Al", roughness=0.1),
        }

    # ---- Mirror on back wall (conductor) ----
    s["mirror"] = {
        "type": "rectangle",
        "to_world": _tf(
            translate=(0, -3.98, 2.5),
            scale=(1.5, 1, 1),
        ),
        "bsdf": _bsdf_mirror(roughness=0.002),
    }
    s["mirror_frame"] = {
        "type": "rectangle",
        "to_world": _tf(
            translate=(0, -3.97, 2.5),
            scale=(1.7, 1.2, 1),
        ),
        "bsdf": _bsdf_diffuse((0.15, 0.10, 0.07)),
    }

    # ---- Water glass (dielectric, IOR 1.33) ----
    s["water_glass_outer"] = {
        "type": "cylinder",
        "p0": mi.ScalarPoint3f(0.8, 0.2, 1.53),
        "p1": mi.ScalarPoint3f(0.8, 0.2, 1.95),
        "radius": 0.12,
        "bsdf": _bsdf_glass(ior=1.52, roughness=0.0),
        "interior": {
            "type": "homogeneous",
            "albedo": {"type": "rgb", "value": (0.4, 0.6, 0.9)},
            "sigma_t": 2.0,
        },
    }

    # ---- Metal vase (roughconductor copper) ----
    s["vase_body"] = {
        "type": "sphere",
        "center": mi.ScalarPoint3f(1.5, 0.3, 1.56),
        "radius": 0.12,
        "bsdf": _bsdf_metal("Cu", roughness=0.12),
    }
    s["vase_neck"] = {
        "type": "cylinder",
        "p0": mi.ScalarPoint3f(1.5, 0.3, 1.68),
        "p1": mi.ScalarPoint3f(1.5, 0.3, 1.80),
        "radius": 0.06,
        "bsdf": _bsdf_metal("Cu", roughness=0.12),
    }

    # ---- Wax candle (plastic — SSS-like internal scattering) ----
    s["candle_body"] = {
        "type": "cylinder",
        "p0": mi.ScalarPoint3f(-0.3, 0.1, 1.53),
        "p1": mi.ScalarPoint3f(-0.3, 0.1, 1.95),
        "radius": 0.08,
        "bsdf": _bsdf_plastic((0.95, 0.92, 0.75), ior=1.45),
    }
    s["candle_flame"] = {
        "type": "sphere",
        "center": mi.ScalarPoint3f(-0.3, 0.1, 2.02),
        "radius": 0.025,
        "emitter": {
            "type": "area",
            "radiance": {"type": "rgb", "value": (50, 35, 10)},
        },
    }

    # ---- Wooden desk with specular coating (blendbsdf) ----
    s["desk_top"] = {
        "type": "cube",
        "to_world": _tf(
            translate=(-3.5, -2, 1.0),
            scale=(2.0, 0.8, 0.05),
        ),
        "bsdf": _bsdf_coated_wood(),
    }
    # Desk legs
    for i, (dx, dy) in enumerate(
        [(-1.6, -0.6), (-1.6, 0.6), (1.6, -0.6), (1.6, 0.6)]
    ):
        s[f"desk_leg_{i}"] = {
            "type": "cylinder",
            "p0": mi.ScalarPoint3f(-3.5 + dx, -2 + dy, 0),
            "p1": mi.ScalarPoint3f(-3.5 + dx, -2 + dy, 0.97),
            "radius": 0.05,
            "bsdf": _bsdf_diffuse((0.45, 0.28, 0.12)),
        }

    # ---- Diffuse couch (roughplastic — fabric-like) ----
    s["couch_seat"] = {
        "type": "cube",
        "to_world": _tf(translate=(-2.5, 2.5, 0.4), scale=(1.5, 0.6, 0.4)),
        "bsdf": _bsdf_roughplastic((0.25, 0.30, 0.45), roughness=0.35),
    }
    s["couch_back"] = {
        "type": "cube",
        "to_world": _tf(translate=(-2.5, 3.2, 0.9), scale=(1.5, 0.15, 0.5)),
        "bsdf": _bsdf_roughplastic((0.25, 0.30, 0.45), roughness=0.35),
    }
    s["couch_arm_l"] = {
        "type": "cube",
        "to_world": _tf(translate=(-3.85, 2.7, 0.55), scale=(0.15, 0.5, 0.55)),
        "bsdf": _bsdf_roughplastic((0.22, 0.27, 0.42), roughness=0.35),
    }
    s["couch_arm_r"] = {
        "type": "cube",
        "to_world": _tf(translate=(-1.15, 2.7, 0.55), scale=(0.15, 0.5, 0.55)),
        "bsdf": _bsdf_roughplastic((0.22, 0.27, 0.42), roughness=0.35),
    }

    # ---- Large sphere with SSS (plastic — skin/wax) ----
    s["sss_sphere"] = {
        "type": "sphere",
        "center": mi.ScalarPoint3f(4, -2.5, 0.5),
        "radius": 0.5,
        "bsdf": _bsdf_plastic((0.85, 0.65, 0.55), ior=1.4),
    }

    # ---- Gold sphere (roughconductor Au) ----
    s["gold_sphere"] = {
        "type": "sphere",
        "center": mi.ScalarPoint3f(-4.5, -2.5, 0.35),
        "radius": 0.35,
        "bsdf": _bsdf_metal("Au", roughness=0.05),
    }


# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------

def _build_lights(s):
    """Window area light, spot, point, constant env."""

    # Large window on left wall — warm daylight
    s["window_light"] = {
        "type": "rectangle",
        "to_world": _tf(
            translate=(-5.95, -0.5, 2.8),
            rotate=([0, 0, 1], -90),
            scale=(1.5, 1.2, 1),
        ),
        "emitter": {
            "type": "area",
            "radiance": {"type": "rgb", "value": (20.0, 17.0, 13.0)},
        },
    }

    # Spot light above desk
    s["desk_spot"] = {
        "type": "spot",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[-3.5, -2, 3.9],
            target=[-3.5, -2, 1.0],
            up=[0, 1, 0],
        ),
        "intensity": {"type": "rgb", "value": (80, 70, 55)},
        "cutoff_angle": 25.0,
        "beam_width": 18.0,
    }

    # Point light in back-right corner
    s["corner_point"] = {
        "type": "point",
        "position": [4, -3.5, 3.5],
        "intensity": {"type": "rgb", "value": (30, 25, 18)},
    }

    # Constant environment — subtle ambient
    s["env"] = {
        "type": "constant",
        "radiance": {"type": "rgb", "value": (0.3, 0.3, 0.35)},
    }


# ---------------------------------------------------------------------------
# Fog volume
# ---------------------------------------------------------------------------

def _build_fog(s):
    """Thin homogeneous fog filling the room (volpath only)."""
    s["fog_box"] = {
        "type": "cube",
        "to_world": _tf(translate=(0, 0, 2), scale=(5.5, 3.5, 1.9)),
        "bsdf": _bsdf_null(),
        "interior": {
            "type": "homogeneous",
            "albedo": 0.95,
            "sigma_t": 0.15,
            "phase": {"type": "hg", "g": 0.3},
        },
    }


# ---------------------------------------------------------------------------
# Camera / sensor
# ---------------------------------------------------------------------------

def _build_sensor(s, w=1280, h=720):
    """Perspective camera looking into the room."""
    s["sensor"] = {
        "type": "perspective",
        "fov": 55,
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[2, 5.5, 2.2],
            target=[0, 0, 1.5],
            up=[0, 0, 1],
        ),
        "film": {
            "type": "hdrfilm",
            "width": w, "height": h,
            "pixel_format": "RGB",
            "component_format": "float32",
        },
        "sampler": {"type": "independent", "sample_count": 1},
    }


# ---------------------------------------------------------------------------
# Assemble scene
# ---------------------------------------------------------------------------

def build_scene(w=1280, h=720, fog=False):
    """Build the full scene dict and load it."""
    s: dict = {"type": "scene"}

    _build_room(s)
    _build_objects(s)
    _build_lights(s)
    if fog:
        _build_fog(s)
    _build_sensor(s, w, h)

    return mi.load_dict(s)


def main():
    parser = argparse.ArgumentParser(description="Complex Mitsuba test scene")
    parser.add_argument("--spp", type=int, default=64, help="Samples per pixel")
    parser.add_argument("--mode", choices=["path", "volpath"], default="path")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA GPU (cuda_ad_rgb)")
    args = parser.parse_args()

    variant = "cuda_ad_rgb" if args.gpu else "llvm_ad_rgb"
    mi.set_variant(variant)
    print(f"Variant: {variant}")

    fog = args.mode == "volpath"
    print(f"Building scene (fog={fog})...")
    scene = build_scene(w=args.width, h=args.height, fog=fog)
    print("Scene loaded.")

    if args.no_render:
        return

    integrator_type = "volpath" if fog else "path"
    integrator = mi.load_dict({
        "type": integrator_type,
        "max_depth": args.max_depth,
    })

    print(f"Rendering {args.width}x{args.height} @ {args.spp} spp "
          f"({integrator_type}, max_depth={args.max_depth})...")
    t0 = time.time()
    image = mi.render(scene, spp=args.spp, integrator=integrator)
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s ({args.spp / dt:.0f} spp/s)")

    out = "complex_scene_output.exr"
    mi.util.write_bitmap(out, image)
    print(f"Saved: {out}")

    # Also save LDR
    ldr = mi.util.convert_to_bitmap(image)
    out_ldr = "complex_scene_output.png"
    ldr.write(out_ldr)
    print(f"Saved: {out_ldr}")


if __name__ == "__main__":
    main()
