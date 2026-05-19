#!/usr/bin/env python3
"""10-channel AOV conv2d variant comparison — side-by-side benchmark.

Two conv2d backends tested with identical 10ch AOV input:
  Variant A: conv2d_safe (pure nabla im2col + matmul) — any depth, slower compile
  Variant B: native nb.conv2d (single MLIR op) — max 2 layers, fast compile

Architecture (both variants identical):
  Conv1: 10→256 channels, 3x3, stride=2, padding=1
  Conv2: 256→256 channels, 3x3, stride=1, padding=1
  Global avg pool → Linear 256→latent
  Scene encoder: 18→32→latent
  Cross-attention fusion (sigmoid gate)
  Loss: MSE with square()

AOV input (10 channels):
  albedo(3) + normal(3) + depth(1) + material_id(1) + motion_vectors(2)

Scene: random selection from 5 Mitsuba scenes, random camera, random seed.
RAM budget: 24GB WARN, 28GB KILL (sys.exit 99).
GPU: RTX 3060 12GB VRAM.

Usage:
  python test_gpu_aov_conv2d_compare.py                       # both variants
  python test_gpu_aov_conv2d_compare.py --variant safe        # conv2d_safe only
  python test_gpu_aov_conv2d_compare.py --variant native      # nb.conv2d only
  python test_gpu_aov_conv2d_compare.py --resolution 512      # 512x512
  python test_gpu_aov_conv2d_compare.py --steps 100           # 100 steps
  python test_gpu_aov_conv2d_compare.py --seed 12345          # fixed seed
  python test_gpu_aov_conv2d_compare.py --sustain 30          # 30-min sustained
"""

import argparse
import gc
import logging
import os
import random
import subprocess
import sys
import time

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
    build_foggy_corridor,
    build_shaderball,
    build_studio_product,
    build_veach_ajar,
)

# Prefer GPU rendering: cuda_ad_rgb > llvm_ad_rgb > scalar_rgb
_available = set(mi.variants())
_mi_variant = next(
    v for v in ("cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb") if v in _available
)
mi.set_variant(_mi_variant)
MI_GPU = "cuda" in _mi_variant

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aov_compare")

WARN_MB = 24 * 1024
KILL_MB = 28 * 1024

# ── Scene registry ───────────────────────────────────────────
SCENE_BUILDERS = [
    ("cornell", build_cornell_box),
    ("veach_ajar", build_veach_ajar),
    ("shaderball", build_shaderball),
    ("studio_product", build_studio_product),
    ("foggy_corridor", build_foggy_corridor),
]


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
        "[guard] RSS=%dMB VRAM=%dMB GPU=%d%%  %s",
        rss,
        _vram_mb(),
        _gpu_util(),
        label,
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


# ── AOV data generation ──────────────────────────────────────
AOV_CH = 10  # albedo(3) + normal(3) + depth(1) + material_id(1) + motion(2)


def _render_pair_with_aov(res, scene_idx=0, seed=42):
    """Render scene RGB + generate 10ch AOV.

    Uses Mitsuba AOV integrator for albedo/normal/depth.
    Missing passes (material_id, motion_vectors) zero-filled.
    Random camera selection per scene.

    Returns:
        aov: (1, res, res, 10)
        gt_rgb: (1, res, res, 3)
        scene_feat: (1, 18)
    """
    rng = np.random.RandomState(seed)
    name, builder = SCENE_BUILDERS[scene_idx % len(SCENE_BUILDERS)]
    log.info("  Rendering %s at %dx%d (seed=%d)...", name, res, res, seed)
    scene, sg = builder(resolution=(res, res))

    # Ground truth + noisy render
    gt = np.array(mi.render(scene, spp=64, seed=seed))[:, :, :3].astype(np.float32)

    # AOV render via Mitsuba integrator
    aov_data = _render_aov(scene, res, spp=64, seed=seed)

    # Pack into 10ch: albedo(3)+normal(3)+depth(1)+material_id(1)+motion(2)
    aov = _pack_aov(aov_data, res)

    gt_rgb = gt[np.newaxis]
    aov = aov[np.newaxis]
    feat = _scene_feat(sg)[np.newaxis]
    return aov, gt_rgb, feat


def _render_aov(scene, res, spp=64, seed=42):
    """Render AOV passes using Mitsuba aov integrator.

    Handles both scalar_rgb (returns dict) and cuda_ad_rgb (returns tensor).
    """
    try:
        aov_integrator = mi.load_dict({
            "type": "aov",
            "aovs": "albedo:albedo,normal:sh_normal,depth:depth",
        })
        result = mi.render(scene, spp=spp, seed=seed, integrator=aov_integrator)

        # scalar_rgb returns dict {name: Bitmap}
        if isinstance(result, dict):
            buffers = {}
            for key, tensor in result.items():
                if hasattr(tensor, "shape"):
                    arr = np.array(tensor)
                    if arr.ndim >= 2:
                        buffers[key] = arr.astype(np.float32)
            return buffers

        # cuda_ad_rgb / llvm_ad_rgb returns multi-channel tensor (H, W, C)
        # Layout: RGB(0-2) + albedo(3-5) + sh_normal(6-8) + depth(9)
        arr = np.array(result).astype(np.float32)
        if arr.ndim >= 2:
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            buffers = {}
            if arr.shape[-1] >= 6:
                buffers["albedo"] = arr[:, :, 3:6]
            if arr.shape[-1] >= 9:
                buffers["normal"] = arr[:, :, 6:9]
            if arr.shape[-1] >= 10:
                buffers["depth"] = arr[:, :, 9:10]
            return buffers

        return {}
    except Exception as exc:
        log.warning("AOV render failed (%s) — synthetic fallback", exc)
        return {}


def _pack_aov(aov_data, res):
    """Pack AOV buffers into (res, res, 10) tensor. Missing passes zeroed."""
    # Albedo (3ch)
    if "albedo" in aov_data:
        albedo = aov_data["albedo"][:, :, :3]
        if albedo.shape[:2] != (res, res):
            albedo = np.zeros((res, res, 3), dtype=np.float32)
    else:
        albedo = np.zeros((res, res, 3), dtype=np.float32)

    # Normal (3ch) — normalize to unit vectors
    if "normal" in aov_data:
        normal = aov_data["normal"][:, :, :3].astype(np.float32)
        norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
        normal = normal / norm
        if normal.shape[:2] != (res, res):
            normal = np.zeros((res, res, 3), dtype=np.float32)
    else:
        normal = np.zeros((res, res, 3), dtype=np.float32)

    # Depth (1ch)
    if "depth" in aov_data:
        depth = aov_data["depth"]
        if depth.ndim == 3:
            depth = depth[:, :, :1]
        elif depth.ndim == 2:
            depth = depth[:, :, np.newaxis]
        if depth.shape[:2] != (res, res):
            depth = np.zeros((res, res, 1), dtype=np.float32)
    else:
        depth = np.zeros((res, res, 1), dtype=np.float32)

    # Material ID (1ch) — not available from Mitsuba, zero-fill
    material_id = np.zeros((res, res, 1), dtype=np.float32)

    # Motion vectors (2ch) — not available from Mitsuba, zero-fill
    motion = np.zeros((res, res, 2), dtype=np.float32)

    return np.concatenate([albedo, normal, depth, material_id, motion], axis=-1)


def _generate_synthetic_aov(res, seed=42):
    """Generate synthetic 10ch AOV for testing without Mitsuba AOV support."""
    rng = np.random.RandomState(seed)
    albedo = rng.rand(res, res, 3).astype(np.float32) * 0.8 + 0.1
    normals = rng.randn(res, res, 3).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    depth = rng.rand(res, res, 1).astype(np.float32) * 10.0
    material_id = np.zeros((res, res, 1), dtype=np.float32)
    motion = rng.randn(res, res, 2).astype(np.float32) * 0.1
    return np.concatenate([albedo, normals, depth, material_id, motion], axis=-1)


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


# ── Param init ────────────────────────────────────────────────
def _he(shape):
    fan_in = int(np.prod(shape[:-1])) if len(shape) > 1 else shape[0]
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)


def _z(n):
    return np.zeros(n, dtype=np.float32)


def init_params_aov(latent=128, channels=256):
    """Init params for both variants (identical shapes).

    Conv1: (3,3,10,channels)  — 10ch AOV input
    Conv2: (3,3,channels,channels)
    Pool/Linear: (channels, latent)
    Scene encoder: (18,32, latent)
    Cross-attn gate: (latent, latent)
    """
    p = {}
    p["c1"] = _he((3, 3, AOV_CH, channels))
    p["b1"] = _z(channels)
    p["c2"] = _he((3, 3, channels, channels))
    p["b2"] = _z(channels)
    p["pw"] = _he((channels, latent))
    p["pb"] = _z(latent)
    # Scene encoder
    p["se_w1"] = _he((18, 32))
    p["se_b1"] = _z(32)
    p["se_pw"] = _he((32, latent))
    p["se_pb"] = _z(latent)
    # Cross-attention gate
    p["ca_gw"] = _he((latent, latent))
    p["ca_gb"] = _z(latent)
    return p


def _param_count(p):
    return sum(v.size for v in p.values())


# ── Model architectures ──────────────────────────────────────
def aov_encoder_safe(p, aov, silu_fn=silu_mojo):
    """Variant A: conv2d_safe encoder — 10→256→256, pool→linear."""
    x = silu_fn(
        conv2d_safe(aov, p["c1"], stride=2, padding=1, bias=p["b1"])
    )
    x = silu_fn(
        conv2d_safe(x, p["c2"], stride=1, padding=1, bias=p["b2"])
    )
    return x.mean(axis=(1, 2))


def aov_encoder_native(p, aov, silu_fn=silu_mojo):
    """Variant B: native nb.conv2d encoder — 10→256→256, pool→linear."""
    x = silu_fn(
        nb.conv2d(aov, p["c1"], stride=(2, 2), padding=(1, 1)) + p["b1"]
    )
    x = silu_fn(
        nb.conv2d(x, p["c2"], stride=(1, 1), padding=(1, 1)) + p["b2"]
    )
    return x.mean(axis=(1, 2))


def make_loss(variant="safe"):
    """Create loss function closure for a given variant."""

    encoder_fn = aov_encoder_safe if variant == "safe" else aov_encoder_native

    def loss_fn(p, aov, scene_f, gt_lat):
        render_lat = encoder_fn(p, aov)
        render_lat = render_lat @ p["pw"] + p["pb"]

        h = silu_mojo(scene_f @ p["se_w1"] + p["se_b1"])
        scene_lat = h @ p["se_pw"] + p["se_pb"]

        g = sigmoid_mojo(render_lat @ p["ca_gw"] + p["ca_gb"])
        fused = render_lat + g * scene_lat

        return nb.mean(square(fused - gt_lat))

    return loss_fn


# ── Training loop ─────────────────────────────────────────────
def train_loop(params_np, data_np, *, variant="safe", steps=10, lr=1e-3, label=""):
    loss_fn = make_loss(variant)

    opt_m = {k: np.zeros_like(v) for k, v in params_np.items()}
    opt_v = {k: np.zeros_like(v) for k, v in params_np.items()}
    losses = []
    t_compile = None
    t_first = None

    for step in range(1, steps + 1):
        p = {k: to_dev(v) for k, v in params_np.items()}
        dev_data = [to_dev(d) for d in data_np]

        t0 = time.time()
        lv, grads = nb.value_and_grad(loss_fn, argnums=0)(p, *dev_data)
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

        # AdamW
        b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.01
        for k in params_np:
            opt_m[k] = b1 * opt_m[k] + (1 - b1) * g_np[k]
            opt_v[k] = b2 * opt_v[k] + (1 - b2) * g_np[k] ** 2
            mh = opt_m[k] / (1 - b1 ** (step + 1))
            vh = opt_v[k] / (1 - b2 ** (step + 1))
            params_np[k] -= lr * (mh / (np.sqrt(vh) + eps) + wd * params_np[k])

        losses.append(loss_f)
        guard(f"{label} step {step}")
        log.info(
            "%s Step %2d: loss=%.6f (%dms)", label, step, loss_f, int(dt * 1000)
        )

        del p, dev_data, lv, grads, g_np
        cleanup()

    steady_ms = int(t_first * 1000) if t_first else int(t_compile * 1000)
    log.info(
        "%s  variant=%s compile=%.1fs steady=%dms loss %.4f->%.4f",
        label,
        variant,
        t_compile or 0,
        steady_ms,
        losses[0],
        losses[-1],
    )
    return losses, params_np


# ── Variant runner ────────────────────────────────────────────
def run_variant(variant, res, steps, seed=42, latent=128, channels=256):
    log.info("=" * 60)
    log.info(
        "Variant %s: %dx%d latent=%d channels=%d seed=%d",
        variant,
        res,
        res,
        latent,
        channels,
        seed,
    )
    log.info("=" * 60)
    guard("start")

    # Random scene selection
    rng = random.Random(seed)
    scene_idx = rng.randint(0, len(SCENE_BUILDERS) - 1)
    render_seed = rng.randint(0, 2**31 - 1)

    aov, gt_rgb, scene_feat = _render_pair_with_aov(
        res, scene_idx=scene_idx, seed=render_seed
    )
    gt_latent = np.random.randn(1, latent).astype(np.float32) * 0.01

    p = init_params_aov(latent=latent, channels=channels)
    n_p = _param_count(p)
    log.info("  params=%s  (%s variant)", f"{n_p:,}", variant)

    losses, p = train_loop(
        p,
        [aov, scene_feat, gt_latent],
        variant=variant,
        steps=steps,
        label=f"V-{variant}",
    )

    ok = all(np.isfinite(v) for v in losses)
    log.info(
        "Variant %s %s  final_loss=%.6f",
        variant,
        "PASS" if ok else "FAIL",
        losses[-1],
    )

    del p, aov, gt_rgb, gt_latent, scene_feat
    cleanup()
    guard("end")
    return losses, ok


# ── Sustained test ────────────────────────────────────────────
def _run_sustain(minutes, res, variant, seed, latent, channels=256):
    log.info("=" * 60)
    log.info(
        "Sustained: %s variant %dx%d ch=%d for %d min",
        variant,
        res,
        res,
        channels,
        minutes,
    )
    log.info("=" * 60)
    guard("start")

    rng = random.Random(seed)
    scene_idx = rng.randint(0, len(SCENE_BUILDERS) - 1)
    render_seed = rng.randint(0, 2**31 - 1)

    aov, gt_rgb, scene_feat = _render_pair_with_aov(
        res, scene_idx=scene_idx, seed=render_seed
    )
    gt_latent = np.random.randn(1, latent).astype(np.float32) * 0.01

    p = init_params_aov(latent=latent, channels=channels)
    loss_fn = make_loss(variant)

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

        pg = {k: to_dev(v) for k, v in p.items()}
        noisy = to_dev(aov)
        sf = to_dev(scene_feat)
        gt = to_dev(gt_latent)

        t0 = time.time()
        lv, grads = nb.value_and_grad(loss_fn, argnums=0)(pg, noisy, sf, gt)
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
                step,
                loss_f,
                int(dt * 1000),
                lr,
                elapsed,
            )

        del pg, noisy, sf, gt, lv, grads, g_np
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

    parser = argparse.ArgumentParser(
        description="10ch AOV conv2d variant comparison"
    )
    parser.add_argument(
        "--variant",
        choices=["safe", "native", "both"],
        default="both",
        help="Which conv2d variant to test (default: both)",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: time-based)"
    )
    parser.add_argument(
        "--latent", type=int, default=128
    )
    parser.add_argument(
        "--sustain", type=int, default=0, help="Minutes for sustained test"
    )
    parser.add_argument(
        "--channels", type=int, default=256, help="Conv channel width (reduce for high res)"
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time()) % (2**31)
    np.random.seed(seed)
    random.seed(seed)

    ch = args.channels
    log.info("=" * 60)
    log.info("10-Channel AOV Conv2d Variant Comparison")
    log.info("Mitsuba: %s (%s)", _mi_variant, "GPU" if MI_GPU else "CPU")
    log.info("Device: %s", dev())
    log.info("Seed: %d", seed)
    log.info("Resolution: %dx%d", args.resolution, args.resolution)
    log.info("Latent: %d  Channels: %d", args.latent, ch)
    log.info("Budget: %d-%d GB RAM", WARN_MB // 1024, KILL_MB // 1024)
    log.info("AOV channels: %d (albedo3+normal3+depth1+mat_id1+motion2)", AOV_CH)
    log.info("=" * 60)
    guard("start")

    if args.sustain > 0:
        variants = (
            ["safe", "native"] if args.variant == "both" else [args.variant]
        )
        for v in variants:
            _run_sustain(args.sustain, args.resolution, v, seed, args.latent, ch)
        return

    results = {}

    variants = ["safe", "native"] if args.variant == "both" else [args.variant]
    for v in variants:
        log.info("--- Next: Variant %s ---", v)
        try:
            losses, ok = run_variant(
                v, args.resolution, args.steps, seed=seed,
                latent=args.latent, channels=ch
            )
            results[v] = {"losses": losses, "ok": ok}
        except (RuntimeError, ValueError, MemoryError) as e:
            log.warning("Variant %s FAILED: %s", v, e)
            results[v] = {"losses": [], "ok": False}
            cleanup()
            if _rss() > KILL_MB:
                log.error("RAM critical — stopping")
                break

    # Summary
    log.info("=" * 60)
    log.info("COMPARISON SUMMARY")
    log.info("=" * 60)
    for v, r in results.items():
        if r["losses"]:
            log.info(
                "  %s: %s  loss %.4f->%.4f  (%d steps)",
                v,
                "PASS" if r["ok"] else "FAIL",
                r["losses"][0],
                r["losses"][-1],
                len(r["losses"]),
            )
        else:
            log.info("  %s: CRASHED", v)
    guard("final")
    log.info("ALL DONE")


if __name__ == "__main__":
    main()
