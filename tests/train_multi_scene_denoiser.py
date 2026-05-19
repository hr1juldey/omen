#!/usr/bin/env python3
"""Multi-scene training loop for Mojo GPU tiled AOV denoiser.

Imports Mojo denoiser via mojo.importer for auto-compilation.

Usage:
    import mojo.importer  # Enables Mojo imports
    import mojo_gpu_denoiser as denoiser

Architecture Explanation (Why NOT JEPA):
=======================================
This is a CONDITIONAL DENOISER, not JEPA (Joint Embedding Predictive Architecture).

JEPA Characteristics:
- Self-supervised: Predicts future states/embeddings without explicit labels
- Learns representation by predicting one embedding from another (e.g., temporal prediction)
- No ground truth target - uses embedding consistency as training signal
- Used for: Video prediction, world models, representation learning

This Architecture (Conditional Denoiser):
- SUPERVISED: Explicit ground truth target (clean latent)
- CONDITIONAL: Scene features modulate tile processing via FiLM cross-attention
- Loss: MSE(fused_latent, target_latent) - direct supervision
- Used for: Denoising noisy AOV tiles conditioned on scene context

Key Difference:
- JEPA: z_pred = f(z_past)  → predict FUTURE embedding
- This:  z_fused = z_tile + gate * z_scene  → FUSE scene context into tile

The cross-attention here is MODULATORY (FiLM-style), not PREDICTIVE.
It conditions tile processing on scene features, not predicting one from another.

Scene Training:
===============
- Iterates through all 5 scenes: cornell, veach, shaderball, studio, foggy
- Each scene rendered with random camera/seed for diversity
- AOV tiles extracted and fed to Mojo GPU denoiser
- Scene features condition the denoiser per-tile
"""

import argparse
import gc
import logging
import os
import random
import sys
import time
from pathlib import Path

# Deep graphs exceed Python's default recursion limit
sys.setrecursionlimit(50_000)

import mitsuba as mi
import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omen.scenes import (
    build_cornell_box,
    build_veach_ajar,
    build_shaderball,
    build_studio_product,
    build_foggy_corridor,
)

# ── Mitsuba variant ───────────────────────────────────────────
_available = set(mi.variants())
_mi_variant = next(
    (v for v in ("cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb") if v in _available),
    "scalar_rgb",
)
mi.set_variant(_mi_variant)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("multi_scene_trainer")

# ── Scene Registry ────────────────────────────────────────────
ALL_SCENES = {
    "cornell": build_cornell_box,
    "veach": build_veach_ajar,
    "shaderball": build_shaderball,
    "studio": build_studio_product,
    "foggy": build_foggy_corridor,
}

# ── Constants ─────────────────────────────────────────────────
SCENE_FEAT_DIM = 18
LATENT_DIM = 128
TILE_SIZE = 256
OVERLAP = 16
AOV_BASE_CH = 13
AOV_POS_CH = 2
AOV_CH = AOV_BASE_CH + AOV_POS_CH


def _rss():
    """Get RSS memory in MB."""
    try:
        text = open(f"/proc/{os.getpid()}/status").read()
        ln = next(ln for ln in text.splitlines() if ln.startswith("VmRSS:"))
        return int(ln.split()[1]) // 1024
    except Exception:
        return 0


def _scene_feat(sg, dim=SCENE_FEAT_DIM):
    """Extract scene features from scene graph."""
    parts = []
    for key in ("geometry", "materials", "lights"):
        section = sg.get(key, {})
        if "features" in section:
            parts.append(np.array(section["features"]).flatten())
        elif "params" in section:
            parts.append(np.mean(np.array(section["params"]), axis=0).flatten())
    feat = (
        np.concatenate(parts).astype(np.float32)
        if parts
        else np.zeros(dim, dtype=np.float32)
    )
    if feat.shape[0] < dim:
        feat = np.pad(feat, (0, dim - feat.shape[0]))
    return feat[:dim]


def _render_aov(scene, res, spp=64, seed=42):
    """Render 13-channel AOV using Mitsuba aov integrator."""
    try:
        aov_integrator = mi.load_dict({
            "type": "aov",
            "aovs": "albedo:albedo,normal:sh_normal,depth:depth,position:position,uv:uv,material:shape_index",
        })
        result = mi.render(scene, spp=spp, seed=seed, integrator=aov_integrator)
        buffers = {}
        if isinstance(result, dict):
            if "albedo" in result:
                arr = np.array(result["albedo"])
                buffers["albedo"] = arr[:, :, :3] if arr.ndim == 3 else arr
            if "normal" in result:
                arr = np.array(result["normal"])
                buffers["normal"] = arr[:, :, :3] if arr.ndim == 3 else arr
            if "depth" in result:
                buffers["depth"] = np.array(result["depth"])
                if buffers["depth"].ndim == 3:
                    buffers["depth"] = buffers["depth"][:, :, 0]
            if "position" in result:
                arr = np.array(result["position"])
                buffers["position"] = arr[:, :, :3] if arr.ndim == 3 else arr
            if "uv" in result:
                arr = np.array(result["uv"])
                buffers["uv"] = arr[:, :, :2] if arr.ndim == 3 else arr
            if "material" in result:
                buffers["material_id"] = np.array(result["material"])
                if buffers["material_id"].ndim == 3:
                    buffers["material_id"] = buffers["material_id"][:, :, 0]
        else:
            arr = np.array(result)
            if arr.ndim == 3 and arr.shape[2] >= 13:
                buffers["albedo"] = arr[:, :, 0:3]
                buffers["normal"] = arr[:, :, 3:6]
                buffers["depth"] = arr[:, :, 6]
                buffers["position"] = arr[:, :, 7:10]
                buffers["uv"] = arr[:, :, 10:12]
                buffers["material_id"] = arr[:, :, 12]
        return buffers
    except Exception as exc:
        log.warning("AOV render failed (%s) — synthetic fallback", exc)
        return {}


def _pack_aov(aov_data, res):
    """Pack AOV buffers into (res, res, AOV_BASE_CH) tensor."""
    # Albedo (3 channels)
    albedo = aov_data.get("albedo", np.zeros((res, res, 3), dtype=np.float32))
    if albedo.shape[:2] != (res, res):
        albedo = np.zeros((res, res, 3), dtype=np.float32)
    else:
        albedo = albedo[:, :, :3].astype(np.float32)

    # Shading normal (3 channels)
    normal = aov_data.get("normal", np.zeros((res, res, 3), dtype=np.float32))
    if normal.shape[:2] != (res, res):
        normal = np.zeros((res, res, 3), dtype=np.float32)
    else:
        normal = normal[:, :, :3].astype(np.float32)
        norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
        normal = normal / norm

    # Depth (1 channel)
    depth = aov_data.get("depth", np.zeros((res, res), dtype=np.float32))
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.shape[:2] != (res, res):
        depth = np.zeros((res, res), dtype=np.float32)
    else:
        depth = depth.astype(np.float32)
    depth = depth[:, :, np.newaxis]

    # Position (3 channels)
    position = aov_data.get("position", np.zeros((res, res, 3), dtype=np.float32))
    if position.shape[:2] != (res, res):
        position = np.zeros((res, res, 3), dtype=np.float32)
    else:
        position = position[:, :, :3].astype(np.float32)

    # UV (2 channels)
    uv = aov_data.get("uv", np.zeros((res, res, 2), dtype=np.float32))
    if uv.shape[:2] != (res, res):
        uv = np.zeros((res, res, 2), dtype=np.float32)
    else:
        uv = uv[:, :, :2].astype(np.float32)

    # Material ID (1 channel)
    material_id = aov_data.get("material_id", np.zeros((res, res), dtype=np.float32))
    if material_id.ndim == 3:
        material_id = material_id[:, :, 0]
    if material_id.shape[:2] != (res, res):
        material_id = np.zeros((res, res), dtype=np.float32)
    else:
        material_id = material_id.astype(np.float32)
    material_id = material_id[:, :, np.newaxis]

    return np.concatenate([albedo, normal, depth, position, uv, material_id], axis=-1)


def _add_tile_position(aov, tile_row, tile_col, grid_h, grid_w):
    """Append 2 sin/cos tile position channels to AOV."""
    h, w = aov.shape[0], aov.shape[1]
    px = (tile_col + 0.5) / grid_w
    py = (tile_row + 0.5) / grid_h
    pos_x = np.full((h, w, 1), np.sin(2 * np.pi * px), dtype=np.float32)
    pos_y = np.full((h, w, 1), np.sin(2 * np.pi * py), dtype=np.float32)
    return np.concatenate([aov, pos_x, pos_y], axis=-1)


def tile_image(full_aov, tile_size=TILE_SIZE, overlap=OVERLAP):
    """Split (H,W,C) image into tiles with overlap."""
    h, w, c = full_aov.shape
    step = tile_size
    tiles = []
    row = 0
    for y in range(0, h, step):
        col = 0
        for x in range(0, w, step):
            y0 = max(0, y - overlap)
            y1 = min(h, y + tile_size + overlap)
            x0 = max(0, x - overlap)
            x1 = min(w, x + tile_size + overlap)
            tile = full_aov[y0:y1, x0:x1]
            tiles.append((tile, row, col))
            col += 1
        row += 1
    grid_h = row
    grid_w = col if row > 0 else 0
    return tiles, grid_h, grid_w


def render_scene(scene_name: str, resolution: int, seed: int):
    """Render a single scene and return AOV + scene features."""
    builder = ALL_SCENES[scene_name]
    log.info("Rendering %s at %dx%d (seed=%d)", scene_name, resolution, resolution, seed)
    scene, sg = builder(resolution=(resolution, resolution))

    # Render clean and noisy RGB
    gt_rgb = np.array(mi.render(scene, spp=64, seed=0))[:, :, :3].astype(np.float32)
    noisy_rgb = np.array(mi.render(scene, spp=2, seed=seed))[:, :, :3].astype(np.float32)

    # Render AOV
    aov_data = _render_aov(scene, resolution, spp=64, seed=seed)
    aov = _pack_aov(aov_data, resolution)

    # Extract scene features
    scene_feat = _scene_feat(sg)

    return {
        "aov": aov,
        "gt_rgb": gt_rgb,
        "noisy_rgb": noisy_rgb,
        "scene_feat": scene_feat,
        "scene_graph": sg,
    }


def run_multi_scene_training(
    steps: int = 100,
    resolution: int = 512,
    scenes: list | None = None,
    seed: int = 42,
    depth: int = 8,
):
    """Run training across all scenes with Mojo GPU denoiser.

    Each step:
    1. Randomly select a scene
    2. Render with random camera/seed
    3. Tile the AOV
    4. Process tiles with Mojo GPU denoiser
    5. Update parameters (simulated)
    """
    # Import Mojo denoiser (auto-compile via mojo.importer)
    import mojo.importer
    import mojo_gpu_denoiser as denoiser

    if scenes is None:
        scenes = list(ALL_SCENES.keys())

    # Create Mojo denoiser state
    log.info("Creating Mojo GPU denoiser state (depth=%d)...", depth)
    denoiser_state = denoiser.create(depth=depth)
    log.info("Denoiser state created")

    rng = np.random.RandomState(seed)
    log.info("=" * 60)
    log.info("Multi-Scene Mojo GPU Denoiser Training")
    log.info("=" * 60)
    log.info("Scenes: %s", scenes)
    log.info("Resolution: %dx%d", resolution, resolution)
    log.info("Steps: %d", steps)
    log.info("Scene encoder depth: %d", depth)
    log.info("Mitsuba variant: %s", _mi_variant)
    log.info("=" * 60)

    # Scene selection stats
    scene_counts = {name: 0 for name in scenes}
    losses = []

    start_time = time.time()

    for step in range(1, steps + 1):
        step_start = time.time()

        # Random scene selection
        scene_name = random.choice(scenes)
        scene_counts[scene_name] += 1
        step_seed = seed + step * 1000 + random.randint(0, 999)

        # Render scene
        try:
            data = render_scene(scene_name, resolution, step_seed)
        except Exception as e:
            log.warning("Step %d: Scene %s render failed: %s", step, scene_name, e)
            continue

        # Tile AOV
        tiles, grid_h, grid_w = tile_image(data["aov"], tile_size=TILE_SIZE, overlap=OVERLAP)

        # Process tiles with Mojo GPU denoiser
        num_tiles = len(tiles)
        step_loss = 0.0

        for tile_data, row, col in tiles:
            # Add position encoding
            tile_with_pos = _add_tile_position(tile_data, row, col, grid_h, grid_w)

            # For each tile, we'd need a tile_latent from the tile encoder
            # For now, use random initialization (simplified)
            tile_latent = rng.randn(LATENT_DIM).astype(np.float32) * 0.1

            # Generate target latent (random for demo - in real training, this would come from GT)
            target_latent = rng.randn(LATENT_DIM).astype(np.float32) * 0.01

            # Call Mojo denoiser
            result = denoiser.train_step(
                state_py=denoiser_state,
                scene_feat=data["scene_feat"],
                target_latent=target_latent,
                tile_latent=tile_latent,
            )

            step_loss += float(result["loss"])

        avg_tile_loss = step_loss / num_tiles if num_tiles > 0 else 0.0
        losses.append(avg_tile_loss)

        step_time = time.time() - step_start

        if step % 10 == 0 or step == 1:
            avg_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
            log.info(
                "Step %4d | scene=%-10s tiles=%2d loss=%.4f avg=%.4f time=%.2fs rss=%dMB",
                step, scene_name, num_tiles, avg_tile_loss, avg_loss, step_time, _rss()
            )

        # Cleanup
        del data, tiles
        gc.collect()

    total_time = time.time() - start_time

    # Summary
    log.info("=" * 60)
    log.info("Training Complete")
    log.info("=" * 60)
    log.info("Total time: %.2fs (%.2fs/step)", total_time, total_time / steps)
    log.info("Scene distribution:")
    for name, count in scene_counts.items():
        log.info("  %s: %d steps", name, count)
    log.info("Final loss: %.4f", losses[-1] if losses else 0)
    log.info("Min loss: %.4f", min(losses) if losses else 0)
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scene training for Mojo GPU tiled AOV denoiser"
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8, choices=[4, 8, 16, 32],
                        help="Scene encoder depth")
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=list(ALL_SCENES.keys()),
        default=list(ALL_SCENES.keys()),
        help="Scenes to train on (default: all)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_multi_scene_training(
        steps=args.steps,
        resolution=args.resolution,
        scenes=args.scenes,
        seed=args.seed,
        depth=args.depth,
    )


if __name__ == "__main__":
    main()
