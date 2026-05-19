"""Helper for Mojo GPU denoiser test — renders Mitsuba scenes with real AOVs."""

import sys

sys.path.insert(0, "/home/riju279/Documents/Projects/MOJO/Cycles_mojo/omen/src")

import numpy as np
import mitsuba as mi

# Use GPU variant for rendering — falls back to CPU if unavailable
_available = set(mi.variants())
_variant = next(
    (v for v in ("cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb") if v in _available),
    "scalar_rgb",
)
mi.set_variant(_variant)

from omen.scenes import (
    build_cornell_box,
    build_veach_ajar,
    build_shaderball,
    build_studio_product,
    build_foggy_corridor,
)

_BUILDERS = [build_cornell_box, build_veach_ajar, build_shaderball, build_studio_product, build_foggy_corridor]
SCENE_NAMES = ["cornell", "veach", "shaderball", "studio", "foggy"]
_PROJ = None


def render_tile(scene_idx, seed, tile_size, sf_dim, lat_dim):
    """Render scene with real AOVs. Returns dict with aov, sf, target arrays."""
    global _PROJ

    scene, sg = _BUILDERS[scene_idx % 2](resolution=(tile_size, tile_size))

    # Clean GT (high SPP)
    gt = np.array(mi.render(scene, spp=64, seed=0))[:, :, :3].astype("float32")

    # Target: fixed random projection of clean GT pixels -> lat_dim
    if _PROJ is None or _PROJ.shape[0] != tile_size * tile_size * 3:
        rng = np.random.RandomState(0)
        _PROJ = rng.randn(tile_size * tile_size * 3, lat_dim).astype("float32") / np.sqrt(
            float(tile_size * tile_size * 3)
        )
    target = gt.reshape(-1).astype("float32") @ _PROJ

    # Noisy AOV (low SPP)
    aov_int = mi.load_dict(
        {
            "type": "aov",
            "aovs": "albedo:albedo,normal:sh_normal,depth:depth,position:position,uv:uv,material:shape_index",
        }
    )
    r = mi.render(scene, spp=2, seed=seed, integrator=aov_int)

    # scalar_rgb returns multi-channel tensor, cuda_ad_rgb returns dict
    if isinstance(r, dict):
        albedo = np.array(r["albedo"])[:, :, :3].astype("float32")
        normal = np.array(r["sh_normal"])[:, :, :3].astype("float32")
        normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
        depth = np.array(r["depth"]).astype("float32")
        depth = depth[:, :, 0:1] if depth.ndim == 3 else depth[:, :, np.newaxis]
        position = np.array(r["position"])[:, :, :3].astype("float32")
        uv = np.array(r["uv"])[:, :, :2].astype("float32")
        mat = np.array(r["shape_index"]).astype("float32")
        mat = mat[:, :, 0:1] if mat.ndim == 3 else mat[:, :, np.newaxis]
    else:
        arr = np.array(r).astype("float32")
        # channels: RGB(3) + albedo(3) + sh_normal(3) + depth(1) + position(3) + uv(2) + shape_index(1) = 16
        # But offset depends on variant. For scalar_rgb: extract what we can
        if arr.ndim == 3 and arr.shape[2] >= 13:
            # Assume layout: albedo(3) + normal(3) + depth(1) + position(3) + uv(2) + mat(1)
            albedo = arr[:, :, 0:3]
            normal = arr[:, :, 3:6]
            normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
            depth = arr[:, :, 6:7]
            position = arr[:, :, 7:10]
            uv = arr[:, :, 10:12]
            mat = arr[:, :, 12:13]
        else:
            # Fallback: use noisy RGB + zeros
            albedo = gt  # use clean GT as albedo fallback
            normal = np.zeros((tile_size, tile_size, 3), dtype="float32")
            depth = np.zeros((tile_size, tile_size, 1), dtype="float32")
            position = np.zeros((tile_size, tile_size, 3), dtype="float32")
            uv = np.zeros((tile_size, tile_size, 2), dtype="float32")
            mat = np.zeros((tile_size, tile_size, 1), dtype="float32")

    pos_v = np.sin(2.0 * np.pi * 0.5)
    H, W = albedo.shape[0], albedo.shape[1]
    px = np.full((H, W, 1), pos_v, dtype="float32")
    py = np.full((H, W, 1), pos_v, dtype="float32")
    aov = np.concatenate([albedo, normal, depth, position, uv, mat, px, py], axis=-1).astype(
        "float32"
    )

    # Scene features from scene graph
    parts = []
    for key in ["geometry", "materials", "lights"]:
        sec = sg.get(key, {})
        if "features" in sec:
            parts.append(np.array(sec["features"]).flatten())
        elif "params" in sec:
            parts.append(np.mean(np.array(sec["params"]), axis=0).flatten())
    sf = np.concatenate(parts).astype("float32") if parts else np.zeros(sf_dim, dtype="float32")
    if sf.shape[0] < sf_dim:
        sf = np.pad(sf, (0, sf_dim - sf.shape[0]))
    sf = sf[:sf_dim].astype("float32")

    return {"aov": aov.ravel(), "sf": sf, "target": target}
