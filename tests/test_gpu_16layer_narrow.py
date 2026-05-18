#!/usr/bin/env python3
"""16-layer narrow conv2d test — prove deep backward works.

Uses conv2d_safe (pure nabla im2col + matmul) — NO conv_transpose, NO cuDNN.
16 layers x 16 channels = ~35K conv params, <1MB feature maps.

Architecture: 4 groups of 4 layers, stride=2 every group start.
  Group 1: 64->32  (layers 1-4)
  Group 2: 32->16  (layers 5-8)
  Group 3: 16->8   (layers 9-12)
  Group 4: 8->4    (layers 13-16)
  Pool + Linear -> latent

+ Scene encoder (linear) + Cross-attention fusion
+ Mojo activations (silu_mojo, sigmoid_mojo)

Progressive: 4->8->16 layers at 64x64, then scale to 128->256.
RAM budget: 26GB system RAM.

Usage:
  python test_gpu_16layer_narrow.py              # all phases
  python test_gpu_16layer_narrow.py --phase 3    # 16 layers at 64x64
  python test_gpu_16layer_narrow.py --phase 5    # 16 layers at 256x256
  python test_gpu_16layer_narrow.py --sustain 30 # 30-min sustained
"""

import argparse
import gc
import logging
import os
import subprocess
import sys
import time

# Deep conv2d_safe graphs exceed Python's default 1000 recursion limit
# during nabla's graph fingerprinting (_fingerprint_obj recursion)
sys.setrecursionlimit(50_000)

import mitsuba as mi
import numpy as np
import nabla as nb
from max.driver import CPU, Accelerator, accelerator_count

from omen.kernels.conv2d import conv2d_safe
from omen.kernels.activations import square
from omen.kernels.activations_gpu import sigmoid_mojo, silu_mojo
from omen.scenes import build_shaderball, build_veach_ajar

mi.set_variant("scalar_rgb")

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("narrow16")

WARN_MB = 24 * 1024
KILL_MB = 28 * 1024


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
    ("shaderball", build_shaderball),
    ("veach_ajar", build_veach_ajar),
]


def _render_pair(res, scene_idx=0):
    name, builder = SCENE_BUILDERS[scene_idx % len(SCENE_BUILDERS)]
    log.info("  Rendering %s at %dx%d ...", name, res, res)
    scene, sg = builder(resolution=(res, res))
    gt = np.array(mi.render(scene, spp=64, seed=0))[:, :, :3].astype(np.float32)
    noisy = np.array(mi.render(scene, spp=2, seed=42))[:, :, :3].astype(np.float32)
    alpha = np.ones((res, res, 1), dtype=np.float32)
    noisy_rgba = np.concatenate([noisy, alpha], axis=-1)[np.newaxis]
    gt_rgb = gt[np.newaxis]
    feat = _scene_feat(sg)[np.newaxis]
    return noisy_rgba, gt_rgb, feat


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


def init_params(num_layers, latent=128):
    p = {}
    # First conv: RGBA (4ch) -> 16ch
    p["c1"] = _he((3, 3, 4, 16))
    p["b1"] = _z(16)
    # Remaining convs: 16ch -> 16ch
    for i in range(2, num_layers + 1):
        p[f"c{i}"] = _he((3, 3, 16, 16))
        p[f"b{i}"] = _z(16)
    # Pool + linear projection
    p["pw"] = _he((16, latent))
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


# ── Model ─────────────────────────────────────────────────────
def narrow_encoder(num_layers, p, rgba, silu_fn=silu_mojo):
    """num_layers of 16ch conv2d_safe. Stride=2 on layers 1,5,9,13."""
    x = rgba
    for i in range(1, num_layers + 1):
        stride = 2 if (i - 1) % 4 == 0 else 1
        x = silu_fn(
            conv2d_safe(x, p[f"c{i}"], stride=stride, padding=1, bias=p[f"b{i}"])
        )
    return x.mean(axis=(1, 2))


def make_loss(num_layers):
    """Create loss function closure for a given depth."""

    def loss_fn(p, noisy, scene_f, gt_lat):
        render_lat = narrow_encoder(num_layers, p, noisy)
        render_lat = render_lat @ p["pw"] + p["pb"]
        h = silu_mojo(scene_f @ p["se_w1"] + p["se_b1"])
        scene_lat = h @ p["se_pw"] + p["se_pb"]
        g = sigmoid_mojo(render_lat @ p["ca_gw"] + p["ca_gb"])
        fused = render_lat + g * scene_lat
        return nb.mean(square(fused - gt_lat))

    return loss_fn


# ── Training loop ─────────────────────────────────────────────
def train_loop(params_np, num_layers, data_np, *, steps=10, lr=1e-3, label=""):
    loss_fn = make_loss(num_layers)

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

        del p, dev_data, lv, grads, g_np
        cleanup()

    steady_ms = int(t_first * 1000) if t_first else int(t_compile * 1000)
    log.info(
        "%s  layers=%d compile=%.1fs steady=%dms loss %.4f->%.4f",
        label,
        num_layers,
        t_compile or 0,
        steady_ms,
        losses[0],
        losses[-1],
    )
    return losses, params_np


# ── Phase runner ──────────────────────────────────────────────
def run_phase(phase_name, num_layers, res, steps, scene_idx=0, latent=128):
    log.info("=" * 60)
    log.info(
        "Phase %s: %d layers at %dx%d latent=%d",
        phase_name,
        num_layers,
        res,
        res,
        latent,
    )
    log.info("=" * 60)
    guard("start")

    noisy_rgba, gt_rgb, scene_feat = _render_pair(res, scene_idx=scene_idx)
    gt_latent = np.random.randn(1, latent).astype(np.float32) * 0.01

    p = init_params(num_layers, latent=latent)
    n_p = _param_count(p)
    log.info("  params=%s  (%d conv layers x 16ch)", f"{n_p:,}", num_layers)

    losses, p = train_loop(
        p,
        num_layers,
        [noisy_rgba, scene_feat, gt_latent],
        steps=steps,
        label=f"P{phase_name}",
    )

    ok = all(np.isfinite(v) for v in losses)
    log.info(
        "Phase %s %s  final_loss=%.6f", phase_name, "PASS" if ok else "FAIL", losses[-1]
    )

    del p, noisy_rgba, gt_rgb, gt_latent, scene_feat
    cleanup()
    guard("end")
    return losses, ok


def _run_sustain(minutes, res, latent):
    """Sustained training at 16 layers."""
    log.info("=" * 60)
    log.info("Sustained: 16 layers %dx%d for %d min", res, res, minutes)
    log.info("=" * 60)
    guard("start")

    noisy_rgba, gt_rgb, scene_feat = _render_pair(res, scene_idx=0)
    gt_latent = np.random.randn(1, latent).astype(np.float32) * 0.01

    p = init_params(16, latent=latent)
    loss_fn = make_loss(16)

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
        noisy = to_dev(noisy_rgba)
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
PHASES = {
    0: ("2-layer seed", 2, 64),
    1: ("4-layer smoke", 4, 64),
    2: ("8-layer half", 8, 64),
    3: ("16-layer deep", 16, 64),
    4: ("16-layer 128x128", 16, 128),
    5: ("16-layer 256x256", 16, 256),
}


def main():
    if accelerator_count() == 0:
        log.error("No GPU found")
        return

    parser = argparse.ArgumentParser(description="16-layer narrow conv2d test")
    parser.add_argument("--phase", type=int, choices=list(PHASES.keys()))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--sustain", type=int, default=0, help="Minutes for sustained test")
    parser.add_argument("--latent", type=int, default=128)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("16-Layer Narrow Conv2d Test")
    log.info("Device: %s", dev())
    log.info("conv2d_safe: pure nabla (NO conv_transpose, NO cuDNN)")
    log.info("Budget: %d-%d GB RAM", WARN_MB // 1024, KILL_MB // 1024)
    log.info("=" * 60)
    guard("start")

    if args.sustain > 0:
        _run_sustain(args.sustain, 64, args.latent)
        return

    if args.phase is not None:
        name, nl, res = PHASES[args.phase]
        # Phase 0 uses smaller latent to seed JIT cache
        lat = 32 if args.phase == 0 else args.latent
        run_phase(str(args.phase), nl, res, args.steps, latent=lat)
    else:
        for pid, (name, nl, res) in PHASES.items():
            log.info("--- Next: Phase %d (%s) ---", pid, name)
            lat = 32 if pid == 0 else args.latent
            try:
                run_phase(str(pid), nl, res, args.steps, latent=lat)
            except (RuntimeError, ValueError, MemoryError) as e:
                log.warning("Phase %d FAILED: %s", pid, e)
                cleanup()
                if _rss() > KILL_MB:
                    log.error("RAM critical — stopping")
                    break

    log.info("=" * 60)
    log.info("ALL DONE")
    guard("final")


if __name__ == "__main__":
    main()
