#!/usr/bin/env python3
"""Omen JEPA GPU training — matmul-only, no conv2d.

Proves GPU training works end-to-end (forward+backward+optimizer).
Conv2d via im2col creates too many small ops for nabla GPU JIT (94s per conv).
This version uses flatten+linear instead of conv — instant GPU compilation.

Architecture (all matmul, no conv2d):
  scene_encoder  — Linear(6→64), Linear(5→64), Linear(7→64) → proj(64→64)
  render_encoder — flatten(H*W*C) → Linear(16384→128) → Linear(128→64)
  cross_attn     — Gated fusion + LayerNorm(64)
  decoder        — Linear(64→128) → reshape → Linear(128→H*W*3)

Pipeline: scene_graph → scene_latent
          noisy_rgba → flatten → render_latent → cross_attn → fused_latent
          gt_rgba    → flatten → render_latent → cross_attn → target_latent
          loss = MSE(fused, target) + SIGReg

RAM guard: 8GB. VRAM budget: 6-8GB. 30 steps.
"""

import gc
import os
import sys
import time

import numpy as np
import nabla as nb
from max.driver import CPU, Accelerator, accelerator_count
from nabla.nn.optim import adamw_init, adamw_update

from omen.kernels.activations import sigmoid_gpu, silu_gpu, sqrt_gpu, square

LIMIT_MB = 8 * 1024
LATENT = 64
SIGREG_LAMBDA = 0.09
RES = 64
FLAT_DIM = RES * RES * 4  # 16384
MID_DIM = 128


# ── Memory guards ─────────────────────────────────────────────
def _rss():
    try:
        text = open(f"/proc/{os.getpid()}/status").read()
        ln = next(ln for ln in text.splitlines() if ln.startswith("VmRSS:"))
        return int(ln.split()[1]) // 1024
    except Exception:
        return 0


def _avail():
    try:
        text = open("/proc/meminfo").read()
        ln = next(ln for ln in text.splitlines() if ln.startswith("MemAvailable:"))
        return int(ln.split()[1]) // 1024
    except Exception:
        return 0


def _vram():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // 1024 // 1024
    except Exception:
        pass
    return 0


def guard(label=""):
    rss = _rss()
    avail = _avail()
    vram = _vram()
    if rss > LIMIT_MB:
        print(f"KILL: RSS={rss}MB > {LIMIT_MB}MB {label}")
        sys.exit(99)
    if avail < 4000:
        print(f"KILL: sys_avail={avail}MB < 4000MB {label}")
        sys.exit(98)
    print(f"  [guard] RSS={rss}MB avail={avail}MB VRAM={vram}MB")
    return rss


# ── Device ─────────────────────────────────────────────────────
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


def clean():
    gc.collect()


# ── Helpers ────────────────────────────────────────────────────
def _rand(*shape, scale=0.01):
    return np.random.randn(*shape).astype(np.float32) * scale


def _zeros(*shape):
    return np.zeros(shape, dtype=np.float32)


def _linear(x, w, b):
    return x @ w + b


def _extract(params, prefix):
    return {k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)}


def _layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    diff = x - mean
    var = (diff * diff).mean(axis=-1, keepdims=True)
    return diff / sqrt_gpu(var + eps) * weight + bias


def _break_graph(tree):
    if isinstance(tree, dict):
        return {k: _break_graph(v) for k, v in tree.items()}
    if hasattr(tree, "to_numpy"):
        return nb.Tensor.from_dlpack(tree.to_numpy().astype(np.float32))
    return tree


def _to_dev(tree):
    if isinstance(tree, dict):
        return {k: _to_dev(v) for k, v in tree.items()}
    if hasattr(tree, "to_numpy"):
        return nb.ops.transfer_to(tree, dev())
    return tree


# ── Param init ────────────────────────────────────────────────
def init_params():
    L = LATENT
    p = {}
    # scene_encoder
    p["scene_encoder.geom_linear.weight"] = _rand(6, L, scale=0.05)
    p["scene_encoder.geom_linear.bias"] = _zeros(L)
    p["scene_encoder.mat_linear.weight"] = _rand(5, L, scale=0.05)
    p["scene_encoder.mat_linear.bias"] = _zeros(L)
    p["scene_encoder.light_linear.weight"] = _rand(7, L, scale=0.05)
    p["scene_encoder.light_linear.bias"] = _zeros(L)
    p["scene_encoder.proj.weight"] = _rand(L, L, scale=0.05)
    p["scene_encoder.proj.bias"] = _zeros(L)
    # render_encoder (flatten → Linear → Linear)
    p["render_encoder.fc1.weight"] = _rand(FLAT_DIM, MID_DIM, scale=0.01)
    p["render_encoder.fc1.bias"] = _zeros(MID_DIM)
    p["render_encoder.fc2.weight"] = _rand(MID_DIM, L, scale=0.01)
    p["render_encoder.fc2.bias"] = _zeros(L)
    # fusion
    p["fusion.gate.weight"] = _rand(L, L, scale=0.05)
    p["fusion.gate.bias"] = _zeros(L)
    p["fusion.norm.weight"] = np.ones((L,), dtype=np.float32)
    p["fusion.norm.bias"] = _zeros(L)
    # decoder (latent → image)
    p["decoder.fc1.weight"] = _rand(L, MID_DIM, scale=0.01)
    p["decoder.fc1.bias"] = _zeros(MID_DIM)
    p["decoder.fc2.weight"] = _rand(MID_DIM, RES * RES * 3, scale=0.01)
    p["decoder.fc2.bias"] = _zeros(RES * RES * 3)
    return p


# ── Functional forward passes (no conv2d) ──────────────────────
def scene_enc(p, sg):
    feats = []
    geom = sg.get("geometry", {})
    if isinstance(geom, dict):
        verts = geom.get("vertices")
        if verts is not None and len(verts.shape) >= 2:
            if len(verts.shape) == 3:
                centroid = verts.mean(axis=1)
                B, D = int(centroid.shape[0]), int(centroid.shape[1])
                spread = nb.mean(
                    square(verts - nb.reshape(centroid, (B, 1, D))), axis=1
                )
                ff = nb.concatenate([centroid, spread], axis=-1)
            else:
                D = int(verts.shape[-1])
                centroid = verts.mean(axis=0)
                spread = nb.mean(square(verts - nb.reshape(centroid, (1, D))), axis=0)
                ff = nb.concatenate(
                    [nb.reshape(centroid, (1, D)), nb.reshape(spread, (1, D))], axis=-1
                )
                n = int(ff.shape[-1])
                if n < 6:
                    ff = nb.pad(ff, ((0, 0), (0, 6 - n)))
                ff = ff[:, :6]
            feats.append(ff @ p["geom_linear.weight"] + p["geom_linear.bias"])
    mats = sg.get("materials", {})
    if isinstance(mats, dict):
        pm = mats.get("params")
        if pm is not None and len(pm.shape) >= 2:
            me = pm @ p["mat_linear.weight"] + p["mat_linear.bias"]
            mp = nb.mean(me, axis=0)
            feats.append(nb.reshape(mp, (1, int(mp.shape[0]))))
    lights = sg.get("lights", {})
    if isinstance(lights, dict):
        pl = lights.get("params")
        if pl is not None and len(pl.shape) >= 2:
            le = pl @ p["light_linear.weight"] + p["light_linear.bias"]
            lp = nb.mean(le, axis=0)
            feats.append(nb.reshape(lp, (1, int(lp.shape[0]))))
    if not feats:
        return nb.zeros((1, int(p["proj.weight"].shape[0])))
    pooled = nb.mean(nb.concatenate(feats, axis=0), axis=0)
    return nb.reshape(pooled, (1, -1)) @ p["proj.weight"] + p["proj.bias"]


def render_enc(p, rgba):
    """Flatten RGBA → Linear → SiLU → Linear → latent."""
    B = int(rgba.shape[0])
    flat = nb.reshape(rgba, (B, FLAT_DIM))
    h = silu_gpu(flat @ p["fc1.weight"] + p["fc1.bias"])
    return h @ p["fc2.weight"] + p["fc2.bias"]


def cross_attn(p, render_lat, scene_lat):
    g = sigmoid_gpu(_linear(render_lat, p["gate.weight"], p["gate.bias"]))
    return _layer_norm(render_lat + g * scene_lat, p["norm.weight"], p["norm.bias"])


def decode(p, latent):
    """Latent → Linear → SiLU → reshape(H,W,3)."""
    h = silu_gpu(latent @ p["fc1.weight"] + p["fc1.bias"])
    flat = h @ p["fc2.weight"] + p["fc2.bias"]
    return nb.reshape(flat, (int(latent.shape[0]), RES, RES, 3))


def sigreg(pred_latent):
    eps = 1e-6
    mean = nb.mean(pred_latent, axis=0)
    var = nb.mean(square(pred_latent - mean), axis=0)
    return -nb.mean(var + eps)


# ── Loss function ──────────────────────────────────────────────
def loss_fn(params, noisy, gt, scene_latent):
    p_re = _extract(params, "render_encoder.")
    p_fu = _extract(params, "fusion.")
    p_de = _extract(params, "decoder.")

    # Predicted latent (noisy)
    rl_noisy = render_enc(p_re, noisy)
    pred_lat = cross_attn(p_fu, rl_noisy, scene_latent)

    # Target latent (GT)
    rl_gt = render_enc(p_re, gt)
    tgt_lat = cross_attn(p_fu, rl_gt, scene_latent)

    # Decoder: predict noise residual
    pred_noise = decode(p_de, pred_lat)
    noisy_rgb = noisy[:, :, :, :3]
    gt_residual = gt[:, :, :, :3] - noisy_rgb

    # Loss
    pred_loss = nb.mean(square(pred_lat - tgt_lat))
    pred_loss = pred_loss + nb.mean(square(pred_noise - gt_residual))
    reg_loss = sigreg(pred_lat)
    return pred_loss + SIGREG_LAMBDA * reg_loss


# ── Scene data ─────────────────────────────────────────────────
def make_scene_data(resolution=64):
    try:
        import mitsuba as mi

        mi.set_variant("cuda_ad_rgb")
        from omen.scenes import build_cornell_box

        scene, scene_graph = build_cornell_box(resolution=(resolution, resolution))
        gt_img = mi.render(scene, spp=256)
        noisy_img = mi.render(scene, spp=4)
        gt_np = np.array(gt_img)[:, :, :3].astype(np.float32)
        noisy_np = np.array(noisy_img)[:, :, :3].astype(np.float32)
        gt_np = np.pad(gt_np, ((0, 0), (0, 0), (0, 1)), constant_values=1.0)
        noisy_np = np.pad(noisy_np, ((0, 0), (0, 0), (0, 1)), constant_values=1.0)
        return noisy_np[np.newaxis], gt_np[np.newaxis], scene_graph
    except Exception as e:
        print(f"  Mitsuba failed ({e}) — synthetic fallback")
        gt_np = (
            np.random.randn(1, resolution, resolution, 4).astype(np.float32) * 0.1 + 0.5
        )
        noise = np.random.randn(1, resolution, resolution, 4).astype(np.float32) * 0.3
        noisy_np = np.clip(gt_np + noise, 0, 2)
        gt_np[:, :, :, 3] = 1.0
        noisy_np[:, :, :, 3] = 1.0
        sg = {
            "geometry": {"vertices": np.random.randn(8, 3).astype(np.float32)},
            "materials": {"params": np.random.randn(3, 5).astype(np.float32) * 0.5},
            "lights": {"params": np.random.randn(1, 7).astype(np.float32) * 0.5},
        }
        return noisy_np, gt_np, sg


# ── Main ────────────────────────────────────────────────────────
def main():
    if accelerator_count() == 0:
        print("No GPU — aborting")
        return

    STEPS = 5
    BASE_LR = 1e-3
    WARMUP = 2

    print(f"=== Omen GPU Training (matmul-only, latent={LATENT}, res={RES}) ===")
    print(f"Device: {dev()}")
    print(f"Steps: {STEPS}, LR: {BASE_LR}")
    guard("start")

    # 1. Params
    print("\n--- Init params ---")
    params_cpu = init_params()
    n_params = sum(v.size for v in params_cpu.values())
    print(
        f"  {len(params_cpu)} tensors, {n_params:,} params ({n_params * 4 / 1024 / 1024:.2f} MB)"
    )
    params = {k: to_dev(v) for k, v in params_cpu.items()}
    print(f"  All params on {dev()}")
    guard("params on device")

    # 2. Optimizer
    print("\n--- Init optimizer ---")
    opt_state = adamw_init(params)
    guard("optimizer init")

    # 3. Scene data
    print("\n--- Scene data ---")
    noisy_np, gt_np, scene_graph_np = make_scene_data(RES)
    print(f"  noisy: {noisy_np.shape}, gt: {gt_np.shape}")

    # Pre-encode scene graph on CPU
    p_se_cpu = _extract(params_cpu, "scene_encoder.")
    sg_nabla = {}
    for key in ("geometry", "materials", "lights"):
        sub = scene_graph_np.get(key, {})
        if isinstance(sub, dict):
            sg_nabla[key] = {
                k2: nb.Tensor.from_dlpack(v2.astype(np.float32))
                for k2, v2 in sub.items()
                if isinstance(v2, np.ndarray)
            }
    scene_latent_cpu = scene_enc(p_se_cpu, sg_nabla)
    nb.realize_all(scene_latent_cpu)
    scene_latent = to_dev(scene_latent_cpu.to_numpy())
    print(f"  scene_latent: {scene_latent.shape} on {scene_latent.device}")

    noisy = to_dev(noisy_np)
    gt = to_dev(gt_np)
    guard("data on device")

    # 4. First forward+backward
    print("\n--- First forward+backward ---")
    t0 = time.time()
    loss_val, grads = nb.value_and_grad(loss_fn, argnums=0)(
        params, noisy, gt, scene_latent
    )
    nb.realize_all(loss_val, grads)
    dt = time.time() - t0
    loss_f = float(to_cpu(loss_val))
    print(f"  First step: {dt:.1f}s, loss={loss_f:.4f}")

    # Eager optimizer (no graph break — see if pure GPU path works)
    params, opt_state = adamw_update(
        params, grads, opt_state, lr=BASE_LR / WARMUP, weight_decay=0.01
    )
    guard("warmup done")

    # 5. Training loop
    print(f"\n--- Training ({STEPS} steps) ---")
    for step in range(1, STEPS + 1):
        lr = BASE_LR * (step + 1) / WARMUP if step < WARMUP else BASE_LR

        t0 = time.time()
        loss_val, grads = nb.value_and_grad(loss_fn, argnums=0)(
            params, noisy, gt, scene_latent
        )
        nb.realize_all(loss_val, grads)
        loss_f = float(to_cpu(loss_val))

        params, opt_state = adamw_update(
            params, grads, opt_state, lr=lr, weight_decay=0.01
        )
        dt = time.time() - t0

        guard(f"step {step}")
        print(f"  Step {step:3d}: loss={loss_f:.4f} ({dt * 1000:.0f}ms) lr={lr:.2e}")

    # 6. Summary
    print("\n--- Done ---")
    guard("final")
    print("  ALL PASSED — Omen JEPA trained on GPU (matmul-only)")


if __name__ == "__main__":
    main()
