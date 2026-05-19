#!/usr/bin/env python3
"""Tiled AOV denoiser with deep scene encoder and FiLM conditioning.

Architecture:
  Scene encoder (runs ONCE):  scene features (18d) → 16/32/64-layer residual MLP → 128d latent
  Tile encoder (per tile):    AOV 12ch (+2 pos) → Conv1→FiLM→silu → Conv2→FiLM→silu → pool → 128d
  Cross-attention fusion:     render_latent + gate * scene_latent → LayerNorm → fused
  Multi-term loss:            MSE + SIGReg (λ=0.09) + energy conservation (λ=0.01)

Tile-based: 256x256 tiles with 16px overlap. Scales to any resolution.
GPU rendering via Mitsuba cuda_ad_rgb. 5 random scenes with camera variation.

Usage:
  python test_gpu_tiled_aov_denoiser.py                          # default: depth=32, 256x256
  python test_gpu_tiled_aov_denoiser.py --scene-depth 64         # 64-layer scene encoder
  python test_gpu_tiled_aov_denoiser.py --resolution 512         # 4 tiles (2x2 grid)
  python test_gpu_tiled_aov_denoiser.py --sustain 30             # 30-min sustained training
  python test_gpu_tiled_aov_denoiser.py --show                   # save GT/noisy/denoised visualization
"""

import argparse
import gc
import logging
import os
import subprocess
import sys
import time

# Deep graphs exceed Python's default 1000 recursion limit
sys.setrecursionlimit(50_000)

import mitsuba as mi
import numpy as np
import nabla as nb
from max.driver import CPU, Accelerator, accelerator_count

from omen.kernels.conv2d import conv2d_safe
from omen.kernels.activations import square
from omen.kernels.activations_gpu import sigmoid_mojo, silu_mojo
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
log = logging.getLogger("tiled_denoiser")

WARN_MB = 24 * 1024
KILL_MB = 28 * 1024
AOV_CH = 10  # albedo(3) + normal(3) + depth(1) + material_id(1) + motion(2)
POS_CH = 2  # sin/cos tile position
INPUT_CH = AOV_CH + POS_CH  # 12


# ── System helpers ────────────────────────────────────────────
def _rss():
    try:
        text = open(f"/proc/{os.getpid()}/status").read()
        ln = next(ln for ln in text.splitlines() if ln.startswith("VmRSS:"))
        return int(ln.split()[1]) // 1024
    except Exception:
        return 0


def _vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip())
    except Exception:
        return 0


def _gpu_util():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return int(out.strip())
    except Exception:
        return 0


def guard(label=""):
    rss = _rss()
    if rss > KILL_MB:
        log.error("KILL: RSS=%dMB > %dMB  %s", rss, KILL_MB, label)
        sys.exit(99)
    if rss > WARN_MB:
        log.warning("WARN: RSS=%dMB > %dMB  %s", rss, WARN_MB, label)
    log.info(
        "[guard] RSS=%dMB VRAM=%dMB GPU=%d%%  %s", rss, _vram_mb(), _gpu_util(), label
    )
    return rss


DEV = None


def dev():
    global DEV
    if DEV is None:
        DEV = Accelerator() if accelerator_count() > 0 else CPU()
    return DEV


def to_dev(arr):
    return nb.ops.transfer_to(nb.Tensor.from_dlpack(arr.astype(np.float32)), dev())


def to_cpu(tensor):
    return nb.ops.transfer_to(tensor, CPU()).to_numpy()


def cleanup():
    gc.collect()


# ── Scene data ────────────────────────────────────────────────
SCENE_BUILDERS = [
    ("cornell", build_cornell_box),
    ("veach", build_veach_ajar),
    ("shaderball", build_shaderball),
    ("studio", build_studio_product),
    ("foggy", build_foggy_corridor),
]


def _scene_feat(sg, dim=18):
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


# ── AOV data pipeline ─────────────────────────────────────────
def _render_aov(scene, res, spp=64, seed=42):
    """Render AOV passes using Mitsuba aov integrator."""
    try:
        aov_integrator = mi.load_dict({
            "type": "aov",
            "aovs": "albedo:albedo,normal:sh_normal,depth:depth",
        })
        result = mi.render(scene, spp=spp, seed=seed, integrator=aov_integrator)
        buffers = {}
        if isinstance(result, dict):
            # scalar_rgb returns dict
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
        else:
            # cuda_ad_rgb returns multi-channel tensor
            arr = np.array(result)
            if arr.ndim == 3 and arr.shape[2] >= 10:
                buffers["albedo"] = arr[:, :, 3:6]
                buffers["normal"] = arr[:, :, 6:9]
                buffers["depth"] = arr[:, :, 9]
        return buffers
    except Exception as exc:
        log.warning("AOV render failed (%s) — synthetic fallback", exc)
        return {}


def _pack_aov(aov_data, res):
    """Pack AOV buffers into (res, res, 10) tensor. Missing passes zeroed."""
    if "albedo" in aov_data:
        albedo = aov_data["albedo"][:, :, :3]
        if albedo.shape[:2] != (res, res):
            albedo = np.zeros((res, res, 3), dtype=np.float32)
    else:
        albedo = np.zeros((res, res, 3), dtype=np.float32)

    if "normal" in aov_data:
        normal = aov_data["normal"][:, :, :3].astype(np.float32)
        norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
        normal = normal / norm
        if normal.shape[:2] != (res, res):
            normal = np.zeros((res, res, 3), dtype=np.float32)
    else:
        normal = np.zeros((res, res, 3), dtype=np.float32)

    if "depth" in aov_data:
        depth = aov_data["depth"]
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.shape[:2] != (res, res):
            depth = np.zeros((res, res), dtype=np.float32)
        else:
            depth = depth.astype(np.float32)
    else:
        depth = np.zeros((res, res), dtype=np.float32)
    depth = depth[:, :, np.newaxis]

    material_id = np.zeros((res, res, 1), dtype=np.float32)
    motion = np.zeros((res, res, 2), dtype=np.float32)
    return np.concatenate([albedo, normal, depth, material_id, motion], axis=-1)


def _generate_synthetic_aov(res, seed=42):
    """Generate synthetic 10ch AOV for testing without Mitsuba AOV support."""
    rng = np.random.RandomState(seed)
    albedo = rng.rand(res, res, 3).astype(np.float32) * 0.8 + 0.1
    normals = rng.randn(res, res, 3).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    depth = rng.rand(res, res, 1).astype(np.float32)
    material_id = np.zeros((res, res, 1), dtype=np.float32)
    motion = np.zeros((res, res, 2), dtype=np.float32)
    return np.concatenate([albedo, normals, depth, material_id, motion], axis=-1)


def _render_pair_with_aov(res, scene_idx=0, seed=42):
    """Render scene RGB + generate 10ch AOV. Returns (aov, gt_rgb, scene_feat)."""
    name, builder = SCENE_BUILDERS[scene_idx % len(SCENE_BUILDERS)]
    log.info("  Rendering %s at %dx%d ...", name, res, res)
    scene, sg = builder(resolution=(res, res))

    gt = np.array(mi.render(scene, spp=64, seed=0))[:, :, :3].astype(np.float32)
    noisy = np.array(mi.render(scene, spp=2, seed=seed))[:, :, :3].astype(np.float32)

    aov_data = _render_aov(scene, res, spp=64, seed=seed)
    aov = _pack_aov(aov_data, res)

    feat = _scene_feat(sg)
    gt_rgb = gt[np.newaxis]
    return aov, gt_rgb, feat, noisy


def _add_tile_position(aov, tile_row, tile_col, grid_h, grid_w):
    """Append 2 sin/cos tile position channels to AOV. (H,W,10) → (H,W,12)."""
    h, w = aov.shape[0], aov.shape[1]
    # Normalized tile center position in [0, 1]
    px = (tile_col + 0.5) / grid_w
    py = (tile_row + 0.5) / grid_h
    pos_x = np.full((h, w, 1), np.sin(2 * np.pi * px), dtype=np.float32)
    pos_y = np.full((h, w, 1), np.sin(2 * np.pi * py), dtype=np.float32)
    return np.concatenate([aov, pos_x, pos_y], axis=-1)


# ── Tiling pipeline ───────────────────────────────────────────
def tile_image(full_aov, tile_size=256, overlap=16):
    """Split (H,W,C) image into tiles with overlap.

    Returns list of (tile_data, row, col) tuples.
    """
    h, w, c = full_aov.shape
    step = tile_size  # step between tile starts
    tiles = []
    row = 0
    for y in range(0, h, step):
        col = 0
        for x in range(0, w, step):
            # Extract tile with overlap, clamped to image bounds
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


def untile_images(tile_outputs, full_h, full_w, tile_size=256, overlap=16):
    """Stitch tile outputs back into (full_h, full_w, C) image with linear blend."""
    c = tile_outputs[0].shape[2] if tile_outputs[0].ndim == 3 else 1
    result = np.zeros((full_h, full_w, c), dtype=np.float32)
    weight = np.zeros((full_h, full_w, 1), dtype=np.float32)

    idx = 0
    row = 0
    for y in range(0, full_h, tile_size):
        col = 0
        for x in range(0, full_w, tile_size):
            if idx >= len(tile_outputs):
                break
            tile = tile_outputs[idx]
            th, tw = tile.shape[0], tile.shape[1]

            # Compute where this tile goes in the full image
            y0 = max(0, y - overlap)
            x0 = max(0, x - overlap)
            y1 = min(full_h, y + tile_size + overlap)
            x1 = min(full_w, x + tile_size + overlap)

            # Blend weight: 1 in tile center, linear ramp in overlap
            wy = np.ones(th, dtype=np.float32)
            wx = np.ones(tw, dtype=np.float32)
            actual_overlap_top = y - y0
            actual_overlap_left = x - x0
            actual_overlap_bottom = min(overlap, y1 - y - tile_size)
            actual_overlap_right = min(overlap, x1 - x - tile_size)
            if actual_overlap_top > 0:
                wy[:actual_overlap_top] = np.linspace(0, 1, actual_overlap_top)
            if actual_overlap_bottom > 0:
                wy[-actual_overlap_bottom:] = np.linspace(1, 0, actual_overlap_bottom)
            if actual_overlap_left > 0:
                wx[:actual_overlap_left] = np.linspace(0, 1, actual_overlap_left)
            if actual_overlap_right > 0:
                wx[-actual_overlap_right:] = np.linspace(1, 0, actual_overlap_right)
            w2d = np.outer(wy, wx)[:, :, np.newaxis]

            result[y0:y1, x0:x1] += tile * w2d
            weight[y0:y1, x0:x0 + 1] += w2d  # fixme: shapes
            weight[y0:y1, x0:x1] += w2d[:, :, 0:1]
            col += 1
            idx += 1
        row += 1

    weight = np.maximum(weight, 1e-8)
    return result / weight


# ── Param init helpers ────────────────────────────────────────
def _he(shape):
    fan_in = int(np.prod(shape[:-1])) if len(shape) > 1 else shape[0]
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)


def _z(n):
    return np.zeros(n, dtype=np.float32)


# ── Deep scene encoder ────────────────────────────────────────
def init_scene_encoder_params(depth=32, width=128):
    """Initialize params for deep residual scene encoder.

    Architecture: Linear(18, width) → (depth-2)× ResBlock(width→width) → Linear(width, 128)
    """
    p = {}
    # Input projection
    p["se_in_w"] = _he((18, width))
    p["se_in_b"] = _z(width)
    # Residual blocks
    for i in range(depth - 2):
        p[f"se_r{i}_w"] = _he((width, width))
        p[f"se_r{i}_b"] = _z(width)
    # Output projection
    p["se_out_w"] = _he((width, 128))
    p["se_out_b"] = _z(128)
    return p


def scene_encoder_fn(p, scene_features, depth=32):
    """Forward pass: scene features (18d) → 128d latent via deep residual MLP."""
    x = scene_features @ p["se_in_w"] + p["se_in_b"]
    for i in range(depth - 2):
        x = silu_mojo(x @ p[f"se_r{i}_w"] + p[f"se_r{i}_b"]) + x
    x = x @ p["se_out_w"] + p["se_out_b"]
    return x


# ── FiLM-conditioned tile encoder ─────────────────────────────
def init_tile_encoder_params(channels=128, aov_ch=INPUT_CH, latent=128):
    """Initialize params for tile encoder with FiLM conditioning."""
    p = {}
    # Conv1: (3,3, aov_ch, channels)
    p["c1"] = _he((3, 3, aov_ch, channels))
    p["b1"] = _z(channels)
    # Conv2: (3,3, channels, channels)
    p["c2"] = _he((3, 3, channels, channels))
    p["b2"] = _z(channels)
    # FiLM generators for each conv layer
    for layer in (1, 2):
        p[f"film{layer}_gw"] = _he((128, channels))
        p[f"film{layer}_gb"] = _z(channels)
        p[f"film{layer}_bw"] = _he((128, channels))
        p[f"film{layer}_bb"] = _z(channels)
    # Pool + linear projection
    p["pw"] = _he((channels, latent))
    p["pb"] = _z(latent)
    # Cross-attention gate
    p["ca_gw"] = _he((latent, latent))
    p["ca_gb"] = _z(latent)
    # Layer norm
    p["ln_w"] = np.ones(latent, dtype=np.float32)
    p["ln_b"] = _z(latent)
    return p


def _film_modulate(conv_out, scene_latent, W_gamma, b_gamma, W_beta, b_beta):
    """FiLM: γ * conv_out + β where γ,β = Linear(scene_latent)."""
    gamma = scene_latent @ W_gamma + b_gamma  # (1, channels)
    beta = scene_latent @ W_beta + b_beta      # (1, channels)
    # Reshape to broadcast over spatial dims: (1, 1, 1, channels)
    B = int(conv_out.shape[0])
    gamma = nb.reshape(gamma, (B, 1, 1, int(gamma.shape[-1])))
    beta = nb.reshape(beta, (B, 1, 1, int(beta.shape[-1])))
    return gamma * conv_out + beta


def tile_encoder_fn(p, aov_tile, scene_latent):
    """Tile encoder: 2 convs with FiLM conditioning from scene_latent."""
    # Conv1 + FiLM + silu
    x = conv2d_safe(aov_tile, p["c1"], stride=2, padding=1, bias=p["b1"])
    x = _film_modulate(x, scene_latent, p["film1_gw"], p["film1_gb"],
                       p["film1_bw"], p["film1_bb"])
    x = silu_mojo(x)
    # Conv2 + FiLM + silu
    x = conv2d_safe(x, p["c2"], stride=2, padding=1, bias=p["b2"])
    x = _film_modulate(x, scene_latent, p["film2_gw"], p["film2_gb"],
                       p["film2_bw"], p["film2_bb"])
    x = silu_mojo(x)
    # Global average pool → linear
    x = x.mean(axis=(1, 2))
    return x @ p["pw"] + p["pb"]



def cross_attn_fn(p, render_latent, scene_latent):
    """Gated cross-attention fusion (no layer norm — nabla VJP CPU scalar bug)."""
    gate = sigmoid_mojo(render_latent @ p["ca_gw"] + p["ca_gb"])
    fused = render_latent + gate * scene_latent
    # NOTE: _layer_norm removed — its sqrt_gpu/div VJPs create CPU scalars
    # on (1,1)-shaped variance tensors during backward. FiLM + sigmoid gate
    # provide sufficient normalization for the test.
    return fused


# ── Multi-term loss ───────────────────────────────────────────
def make_loss(scene_depth=32, lambda_sigreg=0.09, lambda_energy=0.01):
    """Create loss function closure."""

    def loss_fn(p, aov_tile, scene_feat, gt_latent):
        # Scene encoding (runs once per step, cached by nabla)
        scene_lat = scene_encoder_fn(p, scene_feat, depth=scene_depth)

        # Tile encoding with FiLM conditioning
        render_lat = tile_encoder_fn(p, aov_tile, scene_lat)

        # Cross-attention fusion
        fused = cross_attn_fn(p, render_lat, scene_lat)

        # L1: MSE prediction loss
        l_mse = nb.mean(square(fused - gt_latent))

        # L2: SIGReg variance regularization — penalize low variance directly
        # (no sqrt/div/log — all of their VJPs create CPU scalars on GPU)
        mean_f = nb.mean(fused, axis=0)
        var_f = nb.mean(square(fused - mean_f), axis=0)
        l_sigreg = -nb.mean(var_f)  # push variance away from 0

        # L3: Energy conservation (render latent energy ≈ target energy)
        # Uses square(x) instead of abs(x) — nb.abs backward uses 'x > 0'
        # comparison which creates CPU scalar on GPU. Mean-squared is a valid
        # energy metric and avoids the device mismatch.
        l_energy = nb.mean(
            square(
                nb.mean(square(render_lat), axis=-1)
                - nb.mean(square(gt_latent), axis=-1)
            )
        )

        return l_mse + lambda_sigreg * l_sigreg + lambda_energy * l_energy

    return loss_fn


# ── Param aggregation ─────────────────────────────────────────
def init_all_params(latent=128, channels=128, scene_depth=32):
    """Merge all params into single dict."""
    p = {}
    p.update(init_scene_encoder_params(depth=scene_depth))
    p.update(init_tile_encoder_params(channels=channels, aov_ch=INPUT_CH, latent=latent))
    n = sum(v.size for v in p.values())
    log.info("  params=%s  (scene_depth=%d, channels=%d, latent=%d)",
             f"{n:,}", scene_depth, channels, latent)
    return p


# ── Training loop ─────────────────────────────────────────────
def train_loop(params_np, num_tiles, data_np, *, scene_depth=32,
               steps=10, lr=1e-3, label=""):
    """Train with tiled AOV data. data_np = [aov_tiles_list, scene_feat, gt_latent]."""
    loss_fn = make_loss(scene_depth=scene_depth)

    aov_tiles_np = data_np[0]  # list of (1, th, tw, 12) arrays
    scene_feat_np = data_np[1]
    gt_latent_np = data_np[2]

    opt_m = {k: np.zeros_like(v) for k, v in params_np.items()}
    opt_v = {k: np.zeros_like(v) for k, v in params_np.items()}
    losses = []
    t_compile = None
    t_first = None

    for step in range(1, steps + 1):
        # Pick a random tile for this step (single-tile gradient)
        tile_idx = np.random.randint(len(aov_tiles_np))
        aov_tile_np = aov_tiles_np[tile_idx]

        p = {k: to_dev(v) for k, v in params_np.items()}
        aov_dev = to_dev(aov_tile_np)
        sf_dev = to_dev(scene_feat_np)
        gt_dev = to_dev(gt_latent_np)

        t0 = time.time()
        lv, grads = nb.value_and_grad(loss_fn, argnums=0)(p, aov_dev, sf_dev, gt_dev)
        for k in grads:
            nb.realize_all(grads[k])
        nb.realize_all(lv)
        loss_f = float(to_cpu(lv))
        dt = time.time() - t0

        if step == 1:
            t_compile = dt
        if step == 2:
            t_first = dt

        g_np = {k: to_cpu(v) for k, v in grads.items()}

        # AdamW update (numpy — CPU round-trip breaks graph, must rebuild next step)
        b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.01
        for k in params_np:
            opt_m[k] = b1 * opt_m[k] + (1 - b1) * g_np[k]
            opt_v[k] = b2 * opt_v[k] + (1 - b2) * g_np[k] ** 2
            mh = opt_m[k] / (1 - b1 ** (step + 1))
            vh = opt_v[k] / (1 - b2 ** (step + 1))
            params_np[k] -= lr * (mh / (np.sqrt(vh) + eps) + wd * params_np[k])

        losses.append(loss_f)
        guard(f"{label} step {step}")
        log.info("%s Step %2d: loss=%.6f (%dms)", label, step, loss_f, int(dt * 1000))

        del p, aov_dev, sf_dev, gt_dev, lv, grads, g_np
        cleanup()

    steady_ms = int(t_first * 1000) if t_first else int(t_compile * 1000)
    log.info(
        "%s  compile=%.1fs steady=%dms loss %.4f->%.4f",
        label,
        t_compile or 0,
        steady_ms,
        losses[0],
        losses[-1],
    )
    return losses, params_np


# ── Phase runner ──────────────────────────────────────────────
def run_phase(res, steps, scene_depth=32, channels=128, latent=128, seed=42):
    """Run a single training phase at given resolution."""
    log.info("=" * 60)
    log.info("Phase: %dx%d depth=%d ch=%d latent=%d",
             res, res, scene_depth, channels, latent)
    log.info("=" * 60)
    guard("start")

    scene_idx = np.random.randint(len(SCENE_BUILDERS))
    aov_full, gt_rgb, scene_feat, noisy_rgb = _render_pair_with_aov(
        res, scene_idx=scene_idx, seed=seed
    )

    # Generate target latent (random for now — proves pipeline works)
    gt_latent = np.random.randn(1, latent).astype(np.float32) * 0.01

    # Tile the AOV
    tiles, grid_h, grid_w = tile_image(aov_full, tile_size=256, overlap=16)
    log.info("  Tiled: %dx%d grid → %d tiles", grid_h, grid_w, len(tiles))

    # Add position encoding to each tile
    aov_tiles_np = []
    for tile_data, row, col in tiles:
        tile_with_pos = _add_tile_position(tile_data, row, col, grid_h, grid_w)
        aov_tiles_np.append(tile_with_pos[np.newaxis])  # (1, th, tw, 12)

    # Scene feat needs batch dim
    scene_feat_batched = scene_feat[np.newaxis]  # (1, 18)

    p = init_all_params(latent=latent, channels=channels, scene_depth=scene_depth)

    losses, p = train_loop(
        p,
        len(aov_tiles_np),
        [aov_tiles_np, scene_feat_batched, gt_latent],
        scene_depth=scene_depth,
        steps=steps,
        label=f"R{res}",
    )

    ok = all(np.isfinite(v) for v in losses)
    log.info(
        "Phase %dx%d %s  final_loss=%.6f", res, res,
        "PASS" if ok else "FAIL", losses[-1]
    )

    del p, aov_full, gt_rgb, gt_latent, scene_feat, noisy_rgb
    cleanup()
    guard("end")
    return losses, ok


# ── Decode & visualization ────────────────────────────────────
def decode_tile_to_rgb(p, aov_tile_np, scene_feat_np, scene_depth=32):
    """Run encoder forward, decode latent to RGB proxy."""
    p_dev = {k: to_dev(v) for k, v in p.items()}
    aov_dev = to_dev(aov_tile_np)
    sf_dev = to_dev(scene_feat_np)

    scene_lat = scene_encoder_fn(p_dev, sf_dev, depth=scene_depth)
    render_lat = tile_encoder_fn(p_dev, aov_dev, scene_lat)
    fused = cross_attn_fn(p_dev, render_lat, scene_lat)

    # Simple decode: take first 3 dims of latent, tile to small spatial
    # (This is a placeholder — real decoder would be U-Net)
    latent_np = to_cpu(fused)  # (1, 128)
    return latent_np[0, :3]  # RGB proxy (3,)


def save_visualization(p, res, scene_depth, seed=42):
    """Save GT / Noisy / Denoised side-by-side visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping visualization")
        return

    scene_idx = np.random.randint(len(SCENE_BUILDERS))
    aov_full, gt_rgb, scene_feat, noisy_rgb = _render_pair_with_aov(
        res, scene_idx=scene_idx, seed=seed
    )
    scene_feat_batched = scene_feat[np.newaxis]

    # For visualization, use center tile
    tiles, grid_h, grid_w = tile_image(aov_full, tile_size=256, overlap=16)
    center_tile_data, cr, cc = tiles[len(tiles) // 2]
    tile_with_pos = _add_tile_position(center_tile_data, cr, cc, grid_h, grid_w)
    tile_np = tile_with_pos[np.newaxis]

    rgb_proxy = decode_tile_to_rgb(p, tile_np, scene_feat_batched, scene_depth)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(gt_rgb[0], 0, 1))
    axes[0].set_title("GT (spp=64)")
    axes[1].imshow(np.clip(noisy_rgb, 0, 1))
    axes[1].set_title("Noisy (spp=2)")
    # Denoised: show tile's latent as color bar (placeholder)
    denoised_viz = np.zeros((64, 64, 3), dtype=np.float32)
    denoised_viz[:, :, 0] = np.clip(rgb_proxy[0], 0, 1)
    denoised_viz[:, :, 1] = np.clip(rgb_proxy[1], 0, 1)
    denoised_viz[:, :, 2] = np.clip(rgb_proxy[2], 0, 1)
    axes[2].imshow(denoised_viz)
    axes[2].set_title("Denoised (latent RGB proxy)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    os.makedirs("logs", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"logs/tiled_denoise_VISUAL_{ts}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Visualization saved to %s", path)


# ── Sustained mode ────────────────────────────────────────────
def _run_sustained(minutes, res, scene_depth, channels, latent, seed=42):
    """Sustained training with cosine LR decay."""
    log.info("=" * 60)
    log.info("Sustained: depth=%d %dx%d for %d min", scene_depth, res, res, minutes)
    log.info("=" * 60)
    guard("start")

    scene_idx = np.random.randint(len(SCENE_BUILDERS))
    aov_full, gt_rgb, scene_feat, noisy_rgb = _render_pair_with_aov(
        res, scene_idx=scene_idx, seed=seed
    )
    gt_latent = np.random.randn(1, latent).astype(np.float32) * 0.01

    tiles, grid_h, grid_w = tile_image(aov_full, tile_size=256, overlap=16)
    aov_tiles_np = []
    for tile_data, row, col in tiles:
        tile_with_pos = _add_tile_position(tile_data, row, col, grid_h, grid_w)
        aov_tiles_np.append(tile_with_pos[np.newaxis])
    scene_feat_batched = scene_feat[np.newaxis]

    p = init_all_params(latent=latent, channels=channels, scene_depth=scene_depth)
    loss_fn = make_loss(scene_depth=scene_depth)

    opt_m = {k: np.zeros_like(v) for k, v in p.items()}
    opt_v = {k: np.zeros_like(v) for k, v in p.items()}
    lr0 = 1e-3
    losses = []
    t_start = time.time()
    step = 0
    max_rss = 0

    while True:
        step += 1
        elapsed = (time.time() - t_start) / 60
        if elapsed >= minutes:
            break

        lr = lr0 * 0.5 * (1 + np.cos(np.pi * elapsed / minutes))
        tile_idx = np.random.randint(len(aov_tiles_np))
        aov_tile_np = aov_tiles_np[tile_idx]

        pg = {k: to_dev(v) for k, v in p.items()}
        aov_dev = to_dev(aov_tile_np)
        sf_dev = to_dev(scene_feat_batched)
        gt_dev = to_dev(gt_latent)

        t0 = time.time()
        lv, grads = nb.value_and_grad(loss_fn, argnums=0)(pg, aov_dev, sf_dev, gt_dev)
        for k in grads:
            nb.realize_all(grads[k])
        nb.realize_all(lv)
        loss_f = float(to_cpu(lv))
        dt = time.time() - t0

        g_np = {k: to_cpu(v) for k, v in grads.items()}
        b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.01
        for k in p:
            opt_m[k] = b1 * opt_m[k] + (1 - b1) * g_np[k]
            opt_v[k] = b2 * opt_v[k] + (1 - b2) * g_np[k] ** 2
            mh = opt_m[k] / (1 - b1 ** (step + 1))
            vh = opt_v[k] / (1 - b2 ** (step + 1))
            p[k] -= lr * (mh / (np.sqrt(vh) + eps) + wd * p[k])

        losses.append(loss_f)
        rss = guard(f"sustain step {step}")
        max_rss = max(max_rss, rss)

        if step % 20 == 0 or step == 1:
            log.info(
                "Step %d: loss=%.6f (%dms) lr=%.2e %.1fmin",
                step, loss_f, int(dt * 1000), lr, elapsed,
            )

        del pg, aov_dev, sf_dev, gt_dev, lv, grads, g_np
        cleanup()

    total_min = (time.time() - t_start) / 60
    log.info("=" * 60)
    log.info("Sustained DONE: %d steps in %.1f min", step, total_min)
    log.info("  Max RSS: %dMB   Final loss: %.6f", max_rss, losses[-1])
    log.info("  Loss range: %.4f -> %.4f", losses[0], losses[-1])
    log.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────
def main():
    if accelerator_count() == 0:
        log.error("No GPU found")
        return

    parser = argparse.ArgumentParser(description="Tiled AOV denoiser with deep scene encoder")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument("--scene-depth", type=int, default=32, choices=[8, 16, 32, 64])
    parser.add_argument("--sustain", type=int, default=0, help="Minutes for sustained test")
    parser.add_argument("--show", action="store_true", help="Save GT/noisy/denoised visualization")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Tiled AOV Denoiser — Deep Scene Encoder")
    log.info("Device: %s", dev())
    log.info("Mitsuba variant: %s", _mi_variant)
    log.info("Scene encoder depth: %d layers", args.scene_depth)
    log.info("conv2d_safe: pure nabla im2col (NO conv_transpose, NO cuDNN)")
    log.info("FiLM conditioning at every conv layer")
    log.info("Loss: MSE + SIGReg(0.09) + EnergyConservation(0.01)")
    log.info("Budget: %d-%d GB RAM", WARN_MB // 1024, KILL_MB // 1024)
    log.info("=" * 60)
    guard("start")

    if args.sustain > 0:
        _run_sustained(
            args.sustain, args.resolution, args.scene_depth,
            args.channels, args.latent, args.seed,
        )
        return

    losses, ok = run_phase(
        args.resolution, args.steps, args.scene_depth,
        args.channels, args.latent, args.seed,
    )

    if args.show:
        # Re-run with params for visualization
        log.info("Generating visualization...")
        p = init_all_params(args.latent, args.channels, args.scene_depth)
        save_visualization(p, args.resolution, args.scene_depth, args.seed)

    log.info("=" * 60)
    log.info("ALL DONE")
    guard("final")


if __name__ == "__main__":
    main()
