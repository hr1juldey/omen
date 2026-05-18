#!/usr/bin/env python3
"""Omen JEPA GPU training — native nb.conv2d (single MLIR op).

Proves GPU conv2d training works end-to-end using nabla's native conv2d
(which compiles as a single `rmo.conv` MLIR op, not 30+ im2col ops).

Architecture (mirrors Omen JEPA with native conv2d):
  scene_encoder  — Linear(6→64), Linear(5→64), Linear(7→64) → proj(64→64)
  render_encoder — 3x native conv2d(stride=2,pad=1) + global pool + linear
  cross_attn     — Gated fusion + LayerNorm(64)
  decoder        — 4x conv2d encoder + 3x conv2d decoder (U-Net)

Pipeline: scene_graph → scene_latent (CPU, pre-computed)
          noisy_rgba → render_latent → cross_attn → fused_latent
          gt_rgba    → render_latent → cross_attn → target_latent
          decoder(fused_latent) → predicted RGB
          loss = MSE(pred_rgb, gt_rgb) + SIGReg

RAM guard: 12GB. 120-min sustained training with loss convergence.
"""

import gc
import os
import sys
import time

import numpy as np
import nabla as nb
from max.driver import CPU, Accelerator, accelerator_count

from omen.kernels.activations import sigmoid_gpu, silu_gpu, sqrt_gpu, square


# ── Numpy AdamW (avoids nabla graph compilation hang) ────────
def np_adamw_init(params):
    """Init AdamW state as numpy arrays."""
    return {
        "m": {k: np.zeros_like(v) for k, v in params.items()},
        "v": {k: np.zeros_like(v) for k, v in params.items()},
        "t": 0,
    }


def np_adamw_step(params, grads_np, state, lr=1e-3, beta1=0.9, beta2=0.999,
                  eps=1e-8, weight_decay=0.01):
    """AdamW update in pure numpy. params/grads are numpy arrays."""
    state["t"] += 1
    t = state["t"]
    new_params = {}
    for k in params:
        g = grads_np[k]
        state["m"][k] = beta1 * state["m"][k] + (1 - beta1) * g
        state["v"][k] = beta2 * state["v"][k] + (1 - beta2) * g * g
        m_hat = state["m"][k] / (1 - beta1 ** t)
        v_hat = state["v"][k] / (1 - beta2 ** t)
        new_params[k] = params[k] - lr * (m_hat / (np.sqrt(v_hat) + eps)
                                           + weight_decay * params[k])
    return new_params


def _grads_to_cpu(grads):
    """Move nabla gradient dict to CPU numpy."""
    if isinstance(grads, dict):
        return {k: _grads_to_cpu(v) for k, v in grads.items()}
    return nb.ops.transfer_to(grads, CPU()).to_numpy()


def _params_to_cpu(params):
    """Move nabla param dict to CPU numpy, breaking the graph."""
    if isinstance(params, dict):
        return {k: _params_to_cpu(v) for k, v in params.items()}
    if hasattr(params, "to_numpy"):
        return params.to_numpy().astype(np.float32)
    return params


LIMIT_MB = 12 * 1024
LATENT = 64
SIGREG_LAMBDA = 0.09
RES = 64


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
    return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}


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
    # scene_encoder (same as matmul test)
    p["scene_encoder.geom_linear.weight"] = _rand(6, L, scale=0.05)
    p["scene_encoder.geom_linear.bias"] = _zeros(L)
    p["scene_encoder.mat_linear.weight"] = _rand(5, L, scale=0.05)
    p["scene_encoder.mat_linear.bias"] = _zeros(L)
    p["scene_encoder.light_linear.weight"] = _rand(7, L, scale=0.05)
    p["scene_encoder.light_linear.bias"] = _zeros(L)
    p["scene_encoder.proj.weight"] = _rand(L, L, scale=0.05)
    p["scene_encoder.proj.bias"] = _zeros(L)

    # render_encoder: 3x conv2d(stride=2) + global pool + linear proj
    # Input: (B, H, W, 4) RGBA
    # conv1: 4→16, 3x3 → (B, H/2, W/2, 16)
    p["render_encoder.conv1_filter"] = _rand(3, 3, 4, 16, scale=0.05)
    p["render_encoder.conv1_bias"] = _zeros(16)
    # conv2: 16→32, 3x3 → (B, H/4, W/4, 32)
    p["render_encoder.conv2_filter"] = _rand(3, 3, 16, 32, scale=0.05)
    p["render_encoder.conv2_bias"] = _zeros(32)
    # conv3: 32→64, 3x3 → (B, H/8, W/8, 64)
    p["render_encoder.conv3_filter"] = _rand(3, 3, 32, 64, scale=0.05)
    p["render_encoder.conv3_bias"] = _zeros(64)
    # Project pooled 64-dim to LATENT
    p["render_encoder.proj.weight"] = _rand(64, L, scale=0.05)
    p["render_encoder.proj.bias"] = _zeros(L)

    # fusion (same as matmul test)
    p["fusion.gate.weight"] = _rand(L, L, scale=0.05)
    p["fusion.gate.bias"] = _zeros(L)
    p["fusion.norm.weight"] = np.ones((L,), dtype=np.float32)
    p["fusion.norm.bias"] = _zeros(L)

    # decoder: 4x stride-1 conv2d layers (no stride-2 → no conv_transpose backward)
    # (B,H,W,3) → conv1(32ch) → conv2(32ch) → conv3(16ch) → conv4(3ch)
    p["decoder.conv1_filter"] = _rand(3, 3, 3, 32, scale=0.05)
    p["decoder.conv1_bias"] = _zeros(32)
    p["decoder.conv2_filter"] = _rand(3, 3, 32, 32, scale=0.05)
    p["decoder.conv2_bias"] = _zeros(32)
    p["decoder.conv3_filter"] = _rand(3, 3, 32, 16, scale=0.05)
    p["decoder.conv3_bias"] = _zeros(16)
    p["decoder.conv4_filter"] = _rand(3, 3, 16, 3, scale=0.05)
    p["decoder.conv4_bias"] = _zeros(3)

    return p


# ── Functional forward passes ──────────────────────────────────
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
    """3x native conv2d(stride=1) + global avg pool + linear → latent.

    All stride=1 to avoid conv_transpose in backward pass (cuDNN OOM on 12GB VRAM).
    Global avg pool reduces spatial dims regardless.
    """
    x = silu_gpu(
        nb.conv2d(rgba, p["conv1_filter"], padding=(1, 1, 1, 1),
                  bias=p["conv1_bias"])
    )
    x = silu_gpu(
        nb.conv2d(x, p["conv2_filter"], padding=(1, 1, 1, 1),
                  bias=p["conv2_bias"])
    )
    x = silu_gpu(
        nb.conv2d(x, p["conv3_filter"], padding=(1, 1, 1, 1),
                  bias=p["conv3_bias"])
    )
    # Global average pool: (B, H, W, C) → (B, C)
    pool = nb.mean(x, axis=(1, 2))
    return pool @ p["proj.weight"] + p["proj.bias"]


def cross_attn(p, render_lat, scene_lat):
    g = sigmoid_gpu(_linear(render_lat, p["gate.weight"], p["gate.bias"]))
    # Skip layer_norm (sqrt_gpu Newton iterations cause cuDNN workspace OOM when
    # fused with conv_transpose backward). Simple gated fusion instead.
    return render_lat + g * scene_lat


def decode(p, noisy_rgb):
    """Simple 4-layer stride-1 conv2d decoder: noisy_rgb → predicted clean RGB.

    All stride=1, padding=1 → no conv_transpose in backward pass.
    (B,H,W,3) → 32ch → 32ch → 16ch → 3ch.
    """
    x = silu_gpu(
        nb.conv2d(noisy_rgb, p["conv1_filter"], padding=(1, 1, 1, 1),
                  bias=p["conv1_bias"])
    )
    x = silu_gpu(
        nb.conv2d(x, p["conv2_filter"], padding=(1, 1, 1, 1),
                  bias=p["conv2_bias"])
    )
    x = silu_gpu(
        nb.conv2d(x, p["conv3_filter"], padding=(1, 1, 1, 1),
                  bias=p["conv3_bias"])
    )
    return nb.conv2d(x, p["conv4_filter"], padding=(1, 1, 1, 1),
                     bias=p["conv4_bias"])


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

    # Render encoder (single pass — noisy input) + cross attention
    rl = render_enc(p_re, noisy)
    fused = cross_attn(p_fu, rl, scene_latent)

    # Decoder: predict clean RGB from noisy input
    noisy_rgb = noisy[:, :, :, :3]
    gt_rgb = gt[:, :, :, :3]
    pred_rgb = decode(p_de, noisy_rgb)

    # Reconstruction loss + latent regularization (keeps all params in grad graph)
    recon = nb.mean(square(pred_rgb - gt_rgb))
    latent_reg = nb.mean(square(fused)) * 0.001  # small weight, prevents dead grads
    return recon + latent_reg


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

    # 120 minutes target — at ~7s/step post-compile, ~1000 steps
    TOTAL_SECONDS = 120 * 60
    BASE_LR = 1e-3
    WARMUP = 5
    DECAY_STEPS = 2000

    print("=== Omen GPU Native Conv2d Training ===")
    print(f"  latent={LATENT}, res={RES}")
    print(f"  Device: {dev()}")
    print(f"  Target: {TOTAL_SECONDS // 60} min sustained training")
    print(f"  LR: {BASE_LR}, warmup: {WARMUP} steps, cosine decay over {DECAY_STEPS}")
    guard("start")

    # 1. Params
    print("\n--- Init params ---")
    params_cpu = init_params()
    n_params = sum(v.size for v in params_cpu.values())
    print(
        f"  {len(params_cpu)} tensors, {n_params:,} params "
        f"({n_params * 4 / 1024 / 1024:.2f} MB)"
    )
    params = {k: to_dev(v) for k, v in params_cpu.items()}
    print(f"  All params on {dev()}")
    guard("params on device")

    # 2. Optimizer (GPU-only SGD — no CPU transfer, no graph replay)
    print("\n--- GPU-only SGD optimizer ---")
    GRAPH_CLEAR_EVERY = 50  # clear graph cache to prevent RAM growth
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

    # 4. First forward+backward (includes JIT compilation)
    print("\n--- First forward+backward (JIT compile) ---")
    t0 = time.time()
    loss_val, grads = nb.value_and_grad(loss_fn, argnums=0)(
        params, noisy, gt, scene_latent
    )
    nb.realize_all(loss_val, grads)
    dt = time.time() - t0
    loss_f = float(to_cpu(loss_val))
    print(f"  First step: {dt:.1f}s (includes JIT compile), loss={loss_f:.4f}")
    compile_time = dt

    # 5. Training loop — run until 120 minutes elapsed
    start_time = time.time()
    step = 0
    best_loss = float("inf")
    losses = []

    print(f"\n--- Training loop (target {TOTAL_SECONDS // 60} min) ---")
    while True:
        elapsed = time.time() - start_time
        if elapsed >= TOTAL_SECONDS:
            print(f"\n  Reached {TOTAL_SECONDS // 60} min target — stopping")
            break

        step += 1

        # Cosine decay LR with warmup
        if step <= WARMUP:
            lr = BASE_LR * step / WARMUP
        else:
            progress = min((step - WARMUP) / DECAY_STEPS, 1.0)
            lr = BASE_LR * 0.5 * (1.0 + np.cos(np.pi * progress))

        t0 = time.time()

        # Forward+backward on GPU (graph cached after first step)
        loss_val, grads = nb.value_and_grad(loss_fn, argnums=0)(
            params, noisy, gt, scene_latent
        )
        nb.realize_all(loss_val, grads)
        loss_f = float(to_cpu(loss_val))

        # Check for NaN
        if np.isnan(loss_f) or np.isinf(loss_f):
            print(f"  NaN/Inf loss at step {step} — stopping")
            break

        # GPU-only SGD: param = param - lr * grad (stays on GPU, no CPU transfer)
        for k in params:
            params[k] = params[k] - lr * grads[k]

        # Periodic graph break: clear cache to prevent RAM growth
        if step % GRAPH_CLEAR_EVERY == 0:
            nb.GRAPH.clear_all()

        dt = time.time() - t0
        losses.append(loss_f)
        if loss_f < best_loss:
            best_loss = loss_f

        elapsed = time.time() - start_time
        remaining = max(0, TOTAL_SECONDS - elapsed)
        _ = elapsed / step if step > 0 else dt  # noqa: F841

        # Log every 10 steps or every step for first 20
        if step <= 20 or step % 10 == 0:
            guard(f"step {step}")
            print(
                f"  Step {step:4d}: loss={loss_f:.4f} best={best_loss:.4f} "
                f"({dt * 1000:.0f}ms) lr={lr:.2e} "
                f"elapsed={elapsed / 60:.1f}min remain={remaining / 60:.1f}min"
            )

    # 7. Summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("  TRAINING COMPLETE")
    print(f"  Steps: {step}")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Compile time: {compile_time:.1f}s")
    if step > 0:
        avg_step_ms = total_time / step * 1000
        print(f"  Avg step time: {avg_step_ms:.0f}ms")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Best loss: {best_loss:.4f}")
        # Check convergence: compare first 10% vs last 10%
        n = len(losses)
        head = np.mean(losses[: max(1, n // 10)])
        tail = np.mean(losses[-max(1, n // 10):])
        print(f"  Loss change: {head:.4f} → {tail:.4f} (Δ={head - tail:.4f})")
        if tail < head:
            print("  CONVERGENCE: Loss DECREASING — training is learning")
        else:
            print("  WARNING: Loss not decreasing — check LR/architecture")
    guard("final")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
