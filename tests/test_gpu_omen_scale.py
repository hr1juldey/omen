#!/usr/bin/env python3
"""Omen JEPA GPU scale-up — find system limits with real architecture.

Mirrors the actual Omen JEPA model (11 conv2d + linear + fusion)
using Conv2dMojoOp (native forward + custom VJP, no conv_transpose).

Trains on REAL Mitsuba renders (Veach Ajar, Shaderball, Studio, Foggy Corridor).
NOT random noise. NOT Cornell Box.

Usage:
  python test_gpu_omen_scale.py              # all stages A-F
  python test_gpu_omen_scale.py --stage A    # render encoder only
  python test_gpu_omen_scale.py --stage B    # render + scene + fusion
  python test_gpu_omen_scale.py --stage C    # full U-Net decoder (11 conv2d)
  python test_gpu_omen_scale.py --stage D    # scale LATENT up (256/512/1024)
  python test_gpu_omen_scale.py --stage E    # scale resolution up (64/128)
  python test_gpu_omen_scale.py --stage F    # 60-min sustained at max safe
  python test_gpu_omen_scale.py --full       # full model (C + D)

Architecture (mirrors src/omen/model/):
  RenderEncoder  — 3 conv2d (4→32→64→128, stride=2) + pool + Linear
  SceneEncoder   — 3 linear (geom/mat/light) + proj
  CrossAttention — gated fusion + LayerNorm
  Decoder U-Net  — 4 enc conv2d + 4 dec conv2d + pixel shuffle + skips
  Total: 11 conv2d layers

Goal: find RAM ceiling at 20-24GB budget on RTX 3060 12GB.
Ultimate target: 1B params training + inference on this GPU.
"""

import argparse
import gc
import logging
import os
import subprocess
import sys
import time

import mitsuba as mi
import numpy as np
import nabla as nb
from max.driver import CPU, Accelerator, accelerator_count
from nabla.ops import Operation
from omen.kernels.activations import sigmoid_gpu, silu_gpu, square
from omen.kernels.activations_gpu import sigmoid_mojo, silu_mojo
from omen.scenes import (
    build_foggy_corridor,
    build_shaderball,
    build_studio_product,
    build_veach_ajar,
)

mi.set_variant("scalar_rgb")

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("omen_scale")

WARN_MB = 20 * 1024
KILL_MB = 28 * 1024

# Real scenes — NOT Cornell Box (too simple, nothing to learn)
SCENE_BUILDERS = [
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
        log.error("KILL: RSS=%dMB > %dMB %s", rss, KILL_MB, label)
        sys.exit(99)
    if rss > WARN_MB:
        log.warning("WARN: RSS=%dMB > %dMB %s", rss, WARN_MB, label)
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


def _param_count(d):
    return sum(v.size for v in d.values())


def _report(stage, params_np, t0, step_ms=None):
    rss = _rss()
    vram = _vram_mb()
    n_p = _param_count(params_np)
    log.info(
        "[%s] params=%s RSS=%dMB VRAM=%dMB GPU=%d%%",
        stage,
        f"{n_p:,}",
        rss,
        vram,
        _gpu_util(),
    )
    if step_ms:
        log.info("[%s] step=%dms", stage, step_ms)


# ── Real render data (Mitsuba, NOT random noise) ─────────────
def _render_pair(res, scene_idx=0, gt_spp=64, noisy_spp=2):
    """Render real GT+noisy pair from a complex Mitsuba scene."""
    name, builder = SCENE_BUILDERS[scene_idx % len(SCENE_BUILDERS)]
    log.info(
        "  Rendering %s at %dx%d (gt_spp=%d noisy_spp=%d)...",
        name,
        res,
        res,
        gt_spp,
        noisy_spp,
    )
    scene, sg = builder(resolution=(res, res))
    gt = np.array(mi.render(scene, spp=gt_spp, seed=0))[:, :, :3].astype(np.float32)
    noisy = np.array(mi.render(scene, spp=noisy_spp, seed=42))[:, :, :3].astype(
        np.float32
    )
    # RGBA: add alpha=1.0 channel
    alpha = np.ones((res, res, 1), dtype=np.float32)
    noisy_rgba = np.concatenate([noisy, alpha], axis=-1)[np.newaxis]  # (1, H, W, 4)
    gt_rgb = gt[np.newaxis]  # (1, H, W, 3)
    feat = _scene_feat(sg)[np.newaxis]  # (1, 18)
    log.info(
        "  noisy range=[%.4f,%.4f] gt range=[%.4f,%.4f]",
        noisy.min(),
        noisy.max(),
        gt.min(),
        gt.max(),
    )
    return noisy_rgba, gt_rgb, feat


def _scene_feat(sg, dim=18):
    """Extract fixed-size features from scene graph."""
    parts = []
    geom = sg.get("geometry", {})
    if "features" in geom:
        parts.append(np.array(geom["features"]).flatten())
    mat = sg.get("materials", {})
    if "params" in mat:
        parts.append(np.mean(np.array(mat["params"]), axis=0).flatten())
    light = sg.get("lights", {})
    if "params" in light:
        parts.append(np.mean(np.array(light["params"]), axis=0).flatten())
    feat = (
        np.concatenate(parts).astype(np.float32)
        if parts
        else np.zeros(dim, dtype=np.float32)
    )
    if feat.shape[0] < dim:
        feat = np.pad(feat, (0, dim - feat.shape[0]))
    return feat[:dim]


# ── Conv2dMojoOp (from test_gpu_mojo_conv2d_backward.py) ─────
def _parse_sp(stride, padding):
    sh = sw = stride if isinstance(stride, int) else stride[0]
    ph = pw = (
        padding
        if isinstance(padding, int)
        else (padding[0] if isinstance(padding, (tuple, list)) else 0)
    )
    return sh, sw, ph, pw


def _col2im(col, b, h, w, c_in, kh, kw, ph, pw, h_out, w_out):
    col_6d = nb.reshape(col, (b, h_out, w_out, kh, kw, c_in))
    h_pad, w_pad = h + 2 * ph, w + 2 * pw
    parts = []
    for ki in range(kh):
        for kj in range(kw):
            patch = col_6d[:, :, :, ki, kj, :]
            pt, pb = ki, h_pad - ki - h_out
            pl, pr = kj, w_pad - kj - w_out
            parts.append(nb.pad(patch, ((0, 0), (pt, pb), (pl, pr), (0, 0))))
    result = parts[0]
    for p in parts[1:]:
        result = result + p
    if ph > 0 or pw > 0:
        result = result[:, ph : ph + h, pw : pw + w, :]
    return result


class Conv2dMojoOp(Operation):
    @property
    def name(self) -> str:
        return "mojo_conv2d"

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        x, filt = args[0], args[1]
        b, h, w, _ = (int(d) for d in x.shape)
        kh, kw, c_in, c_out = (int(d) for d in filt.shape)
        sh, sw = kwargs["sh"], kwargs["sw"]
        ph, pw = kwargs["ph"], kwargs["pw"]
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1
        return ([(b, h_out, w_out, c_out)], [x.dtype], [x.device])

    def kernel(self, args, kwargs):
        from max.graph import ops as graph_ops

        x, filt = args[0], args[1]
        sh, sw = kwargs["sh"], kwargs["sw"]
        ph, pw = kwargs["ph"], kwargs["pw"]
        bias = args[2] if len(args) > 2 else None
        return [
            graph_ops.conv2d(
                x, filt, stride=(sh, sw), padding=(ph, ph, pw, pw), bias=bias
            )
        ]

    def vjp_rule(self, primals, cotangents, outputs, kwargs):
        x, filt = primals[0], primals[1]
        ct = cotangents[0]
        sh, sw = kwargs["sh"], kwargs["sw"]
        ph, pw = kwargs["ph"], kwargs["pw"]
        kh, kw, c_in, c_out = (int(d) for d in filt.shape)
        b, h, w, _ = (int(d) for d in x.shape)
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1

        x_pad = (
            nb.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0))) if (ph > 0 or pw > 0) else x
        )
        patch_list = []
        for ki in range(kh):
            for kj in range(kw):
                p = x_pad[:, ki : ki + h_out, kj : kj + w_out, :]
                patch_list.append(nb.reshape(p, (b * h_out * w_out, c_in)))
        patches = nb.concatenate(patch_list, axis=1)

        ct_flat = nb.reshape(ct, (b * h_out * w_out, c_out))
        filt_flat = nb.reshape(filt, (kh * kw * c_in, c_out))

        grad_filt = nb.reshape(
            nb.matmul(nb.transpose(patches, 1, 0), ct_flat), (kh, kw, c_in, c_out)
        )
        grad_input = _col2im(
            nb.matmul(ct_flat, nb.transpose(filt_flat, 1, 0)),
            b,
            h,
            w,
            c_in,
            kh,
            kw,
            ph,
            pw,
            h_out,
            w_out,
        )

        if len(primals) > 2:
            return [grad_input, grad_filt, nb.sum(ct, axis=(0, 1, 2))]
        return [grad_input, grad_filt]


_conv2d_op = Conv2dMojoOp()


def conv2d(x, filt, stride=1, padding=0, bias=None):
    sh, sw, ph, pw = _parse_sp(stride, padding)
    args = [x, filt] + ([bias] if bias is not None else [])
    return _conv2d_op(args, {"sh": sh, "sw": sw, "ph": ph, "pw": pw})[0]


# ── Model components (mirror src/omen/model/) ────────────────
def _linear(x, weight, bias):
    return x @ weight + bias


def _layer_norm(x, weight, bias, eps=1e-5):
    """Simple centering + affine (avoids nb.sqrt GPU VJP bug)."""
    mean = x.mean(axis=-1, keepdims=True)
    return (x - mean) * weight + bias


def _pixel_shuffle(x, r=2):
    B, H, W, C = (int(d) for d in x.shape)
    y = nb.reshape(x, (B, H, r, W, r, C // (r * r)))
    y = nb.permute(y, (0, 2, 1, 4, 3, 5))
    return nb.reshape(y, (B, H * r, W * r, C // (r * r)))


def render_encoder(p, rgba, silu_fn=silu_gpu):
    """3 conv2d (stride=2) → global pool → linear.  Mirrors RenderFeatureEncoder."""
    x = silu_fn(conv2d(rgba, p["re_f1"], stride=2, padding=1, bias=p["re_b1"]))
    x = silu_fn(conv2d(x, p["re_f2"], stride=2, padding=1, bias=p["re_b2"]))
    x = silu_fn(conv2d(x, p["re_f3"], stride=2, padding=1, bias=p["re_b3"]))
    x = x.mean(axis=(1, 2))
    return _linear(x, p["re_pw"], p["re_pb"])


def scene_encoder(p, scene_feat, silu_fn=silu_gpu):
    """Linear(18→64) → proj(64→LATENT).  Simplified scene graph encoder."""
    h = silu_fn(_linear(scene_feat, p["se_w1"], p["se_b1"]))
    return _linear(h, p["se_pw"], p["se_pb"])


def cross_attn(p, render_lat, scene_lat, sigmoid_fn=sigmoid_gpu):
    """Gated fusion. Mirrors CrossAttentionFusion (no LayerNorm — GPU-safe)."""
    g = sigmoid_fn(_linear(render_lat, p["ca_gw"], p["ca_gb"]))
    return render_lat + g * scene_lat


def unet_decoder(p, latent, noisy_img, silu_fn=silu_gpu, sigmoid_fn=sigmoid_gpu):
    """U-Net: 4 enc conv2d + bottleneck + 4 dec conv2d.  Mirrors Decoder."""
    # Encoder
    s1 = silu_fn(conv2d(noisy_img, p["de1"], padding=1))
    s2 = silu_fn(conv2d(s1, p["de2"], stride=2, padding=1))
    s3 = silu_fn(conv2d(s2, p["de3"], stride=2, padding=1))
    e4 = silu_fn(conv2d(s3, p["de4"], stride=2, padding=1))

    # Bottleneck: gated latent injection
    gate = sigmoid_fn(_linear(latent, p["lg_w"], p["lg_b"]))
    l_feat = gate * _linear(latent, p["lp_w"], p["lp_b"])
    C_bn = int(e4.shape[-1])
    bn = e4 * nb.reshape(l_feat, (1, 1, 1, C_bn))

    # Decoder: pixel shuffle + skip concat + conv
    d4 = _pixel_shuffle(_linear(bn, p["up4_w"], p["up4_b"]))
    d4 = silu_fn(conv2d(nb.concatenate([d4, s3], axis=-1), p["dd4"], padding=1))

    d3 = _pixel_shuffle(_linear(d4, p["up3_w"], p["up3_b"]))
    d3 = silu_fn(conv2d(nb.concatenate([d3, s2], axis=-1), p["dd3"], padding=1))

    d2 = _pixel_shuffle(_linear(d3, p["up2_w"], p["up2_b"]))
    d2 = silu_fn(conv2d(nb.concatenate([d2, s1], axis=-1), p["dd2"], padding=1))

    return conv2d(d2, p["dd1"], padding=1)


# ── Param init ────────────────────────────────────────────────
def _he(shape):
    """He normal init."""
    fan_in = int(np.prod(shape[:-1])) if len(shape) > 1 else shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape).astype(np.float32) * std


def _init_render_encoder(latent, ch_enc=(32, 64, 128)):
    c1, c2, c3 = ch_enc
    return {
        "re_f1": _he((3, 3, 4, c1)),
        "re_b1": np.zeros(c1, dtype=np.float32),
        "re_f2": _he((3, 3, c1, c2)),
        "re_b2": np.zeros(c2, dtype=np.float32),
        "re_f3": _he((3, 3, c2, c3)),
        "re_b3": np.zeros(c3, dtype=np.float32),
        "re_pw": _he((c3, latent)),
        "re_pb": np.zeros(latent, dtype=np.float32),
    }


def _init_scene_encoder(latent, scene_dim=18):
    return {
        "se_w1": _he((scene_dim, 64)),
        "se_b1": np.zeros(64, dtype=np.float32),
        "se_pw": _he((64, latent)),
        "se_pb": np.zeros(latent, dtype=np.float32),
    }


def _init_cross_attn(latent):
    return {
        "ca_gw": _he((latent, latent)),
        "ca_gb": np.zeros(latent, dtype=np.float32),
    }


def _init_decoder(latent, enc_ch=(64, 128, 256, 256)):
    c1, c2, c3, c4 = enc_ch
    return {
        "de1": _he((3, 3, 3, c1)),
        "de2": _he((3, 3, c1, c2)),
        "de3": _he((3, 3, c2, c3)),
        "de4": _he((3, 3, c3, c4)),
        "lg_w": _he((latent, c4)),
        "lg_b": np.zeros(c4, dtype=np.float32),
        "lp_w": _he((latent, c4)),
        "lp_b": np.zeros(c4, dtype=np.float32),
        "up4_w": _he((c4, c4 * 4)),
        "up4_b": np.zeros(c4 * 4, dtype=np.float32),
        "up3_w": _he((c4, c3 * 4)),
        "up3_b": np.zeros(c3 * 4, dtype=np.float32),
        "up2_w": _he((c3, c2 * 4)),
        "up2_b": np.zeros(c2 * 4, dtype=np.float32),
        "dd4": _he((3, 3, c4 + c3, c3)),
        "dd3": _he((3, 3, c3 + c2, c2)),
        "dd2": _he((3, 3, c2 + c1, c1)),
        "dd1": _he((3, 3, c1, 3)),
    }


# ── Training loop ─────────────────────────────────────────────
def train_loop(params_np, loss_fn, data_np, *, steps=10, lr=1e-3, label=""):
    """Run training steps with numpy AdamW (graph break pattern).

    Args:
        params_np: numpy param dict
        loss_fn: callable(params, *data_args) -> scalar loss
        data_np: list of numpy arrays transferred to device each step
        steps: number of training steps
        lr: learning rate
        label: logging label
    """
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
        "%s compile=%.1fs steady=%dms loss %.4f→%.4f",
        label,
        t_compile or 0,
        steady_ms,
        losses[0],
        losses[-1],
    )
    return losses, params_np


# ── Stage A: Render Encoder Only ──────────────────────────────
def stage_a(res=64, latent=256, steps=10, scene_idx=0, **_kw):
    log.info("=" * 60)
    log.info("Stage A: Render Encoder (3 conv2d) at %dx%d latent=%d", res, res, latent)
    log.info("=" * 60)
    guard("start")

    noisy_rgba, gt_rgb, scene_feat = _render_pair(res, scene_idx=scene_idx)
    gt_latent = np.random.randn(1, latent).astype(np.float32) * 0.01

    p = _init_render_encoder(latent)

    def loss_fn(p, noisy, gt):
        latent_out = render_encoder(p, noisy)
        return nb.mean(square(latent_out - gt))

    _report("A-init", p, time.time())
    losses, p = train_loop(p, loss_fn, [noisy_rgba, gt_latent], steps=steps, label="A")
    assert all(np.isfinite(v) for v in losses), "NaN in Stage A!"
    log.info("Stage A PASSED")
    del p, noisy_rgba, gt_rgb, gt_latent, scene_feat
    cleanup()
    guard("end")
    return losses


# ── Stage B: Render + Scene + Fusion ──────────────────────────
def stage_b(res=256, latent=128, steps=10, scene_idx=1, **_kw):
    log.info("=" * 60)
    log.info("Stage B: Render + Scene + Fusion 256x256 latent=%d", latent)
    log.info("  2 conv2d (4→16→16), Mojo activations, 24GB budget")
    log.info("=" * 60)
    guard("start")

    noisy_rgba, gt_rgb, scene_feat = _render_pair(res, scene_idx=scene_idx)
    gt_latent = np.random.randn(1, latent).astype(np.float32) * 0.01

    # Slim model: 2 conv2d (16D channels), latent=128
    p = {}
    # render_encoder: 2 conv2d stride=2 → pool → linear(16, 128)
    p["re_f1"] = _he((3, 3, 4, 16))
    p["re_b1"] = np.zeros(16, dtype=np.float32)
    p["re_f2"] = _he((3, 3, 16, 16))
    p["re_b2"] = np.zeros(16, dtype=np.float32)
    p["re_pw"] = _he((16, latent))
    p["re_pb"] = np.zeros(latent, dtype=np.float32)
    # scene_encoder: linear(18→32) → linear(32→128)
    p["se_w1"] = _he((18, 32))
    p["se_b1"] = np.zeros(32, dtype=np.float32)
    p["se_pw"] = _he((32, latent))
    p["se_pb"] = np.zeros(latent, dtype=np.float32)
    # cross_attn: linear(128→128)
    p["ca_gw"] = _he((latent, latent))
    p["ca_gb"] = np.zeros(latent, dtype=np.float32)

    def loss_fn(p, noisy, scene_f, gt_lat):
        # 2 conv2d render encoder (no re_f3)
        x = silu_mojo(conv2d(noisy, p["re_f1"], stride=2, padding=1, bias=p["re_b1"]))
        x = silu_mojo(conv2d(x, p["re_f2"], stride=2, padding=1, bias=p["re_b2"]))
        render_lat = x.mean(axis=(1, 2)) @ p["re_pw"] + p["re_pb"]
        # scene encoder
        h = silu_mojo(scene_f @ p["se_w1"] + p["se_b1"])
        scene_lat = h @ p["se_pw"] + p["se_pb"]
        # cross attention
        g = sigmoid_mojo(render_lat @ p["ca_gw"] + p["ca_gb"])
        fused = render_lat + g * scene_lat
        return nb.mean(square(fused - gt_lat))

    _report("B-init", p, time.time())
    losses, p = train_loop(
        p, loss_fn, [noisy_rgba, scene_feat, gt_latent], steps=steps, label="B"
    )
    assert all(np.isfinite(v) for v in losses), "NaN in Stage B!"
    log.info("Stage B PASSED")
    del p, noisy_rgba, gt_rgb, gt_latent, scene_feat
    cleanup()
    guard("end")
    return losses


# ── Stage C: Full U-Net Decoder (11 conv2d total) ────────────
def stage_c(
    res=64, latent=256, enc_ch=(64, 128, 256, 256), steps=10, scene_idx=2, **_kw
):
    log.info("=" * 60)
    log.info("Stage C: Full model (11 conv2d) %dx%d latent=%d", res, res, latent)
    log.info("  decoder channels: %s", enc_ch)
    log.info("=" * 60)
    guard("start")

    noisy_rgba, gt_rgb, scene_feat = _render_pair(res, scene_idx=scene_idx)

    p = {}
    p.update(_init_render_encoder(latent))
    p.update(_init_scene_encoder(latent))
    p.update(_init_cross_attn(latent))
    p.update(_init_decoder(latent, enc_ch))

    def loss_fn(p, noisy, gt, scene_f):
        scene_lat = scene_encoder(p, scene_f)
        render_lat = render_encoder(p, noisy)
        fused = cross_attn(p, render_lat, scene_lat)
        pred = unet_decoder(p, fused, noisy[:, :, :, :3])
        return nb.mean(square(pred - gt))

    _report("C-init", p, time.time())
    losses, p = train_loop(
        p, loss_fn, [noisy_rgba, gt_rgb, scene_feat], steps=steps, label="C"
    )
    assert all(np.isfinite(v) for v in losses), "NaN in Stage C!"
    log.info("Stage C PASSED")
    del p, noisy_rgba, gt_rgb, scene_feat
    cleanup()
    guard("end")
    return losses


# ── Stage D: Scale LATENT up ─────────────────────────────────
def stage_d(res=64, steps=5, latent=None, **_kw):
    log.info("=" * 60)
    log.info("Stage D: Scale LATENT (256/512/1024) at %dx%d", res, res)
    log.info("=" * 60)

    results = {}
    for idx, lat in enumerate([256, 512, 1024]):
        log.info("--- D: latent=%d ---", lat)
        guard(f"pre-latent={lat}")
        try:
            losses = stage_c(res=res, latent=lat, steps=steps, scene_idx=idx)
            results[lat] = ("PASS", _rss(), losses[-1])
        except (RuntimeError, ValueError, MemoryError) as e:
            results[lat] = ("FAIL", _rss(), str(e)[:80])
            log.warning("D: latent=%d FAILED: %s", lat, e)
            cleanup()
            if _rss() > KILL_MB:
                log.error("RAM critical after failure — stopping D sweep")
                break

    log.info("=" * 60)
    log.info("Stage D Summary: LATENT sweep at %dx%d", res, res)
    for lat, (status, rss, val) in sorted(results.items()):
        if status == "PASS":
            log.info("  latent=%d: PASS  RSS=%dMB  final_loss=%.4f", lat, rss, val)
        else:
            log.info("  latent=%d: FAIL  RSS=%dMB  %s", lat, rss, val)
    log.info("=" * 60)
    return results


# ── Stage E: Scale Resolution up ─────────────────────────────
def stage_e(latent=256, steps=5, res=None, **_kw):
    log.info("=" * 60)
    log.info("Stage E: Scale Resolution (64/128) at latent=%d", latent)
    log.info("=" * 60)

    results = {}
    for idx, resolution in enumerate([64, 128]):
        log.info("--- E: res=%d ---", resolution)
        guard(f"pre-res={resolution}")
        try:
            losses = stage_c(res=resolution, latent=latent, steps=steps, scene_idx=idx)
            results[resolution] = ("PASS", _rss(), losses[-1])
        except (RuntimeError, ValueError, MemoryError) as e:
            results[resolution] = ("FAIL", _rss(), str(e)[:80])
            log.warning("E: res=%d FAILED: %s", resolution, e)
            cleanup()
            if _rss() > KILL_MB:
                log.error("RAM critical — stopping E sweep")
                break

    log.info("=" * 60)
    log.info("Stage E Summary: Resolution sweep at latent=%d", latent)
    for resolution, (status, rss, val) in sorted(results.items()):
        if status == "PASS":
            log.info("  res=%d: PASS  RSS=%dMB  final_loss=%.4f", resolution, rss, val)
        else:
            log.info("  res=%d: FAIL  RSS=%dMB  %s", resolution, rss, val)
    log.info("=" * 60)
    return results


# ── Stage F: 60-min sustained at max safe config ──────────────
def stage_f(res=64, latent=256, target_min=60, scene_idx=3, **_kw):
    log.info("=" * 60)
    log.info(
        "Stage F: %d-min sustained training at %dx%d latent=%d",
        target_min,
        res,
        res,
        latent,
    )
    log.info("=" * 60)
    guard("start")

    noisy_rgba, gt_rgb, scene_feat = _render_pair(res, scene_idx=scene_idx)

    p = {}
    p.update(_init_render_encoder(latent))
    p.update(_init_scene_encoder(latent))
    p.update(_init_cross_attn(latent))
    p.update(_init_decoder(latent))
    _report("F-init", p, time.time())

    opt_m = {k: np.zeros_like(v) for k, v in p.items()}
    opt_v = {k: np.zeros_like(v) for k, v in p.items()}
    lr0 = 1e-3
    losses = []
    t_start = time.time()
    step = 0
    max_rss = 0

    def full_loss(params, noisy, gt, sf):
        scene_lat = scene_encoder(params, sf)
        render_lat = render_encoder(params, noisy)
        fused = cross_attn(params, render_lat, scene_lat)
        pred = unet_decoder(params, fused, noisy[:, :, :, :3])
        return nb.mean(square(pred - gt))

    while True:
        step += 1
        elapsed = (time.time() - t_start) / 60
        if elapsed >= target_min:
            break

        # Cosine decay
        lr = lr0 * 0.5 * (1 + np.cos(np.pi * elapsed / target_min))

        params = {k: to_dev(v) for k, v in p.items()}
        noisy = to_dev(noisy_rgba)
        gt = to_dev(gt_rgb)
        sf = to_dev(scene_feat)

        t0 = time.time()
        lv, grads = nb.value_and_grad(full_loss, argnums=0)(params, noisy, gt, sf)
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
        rss = guard(f"F step {step}")
        max_rss = max(max_rss, rss)

        if step % 20 == 0 or step == 1:
            log.info(
                "F Step %d: loss=%.6f (%dms) lr=%.2e elapsed=%.1fmin",
                step,
                loss_f,
                int(dt * 1000),
                lr,
                elapsed,
            )

        del params, noisy, gt, sf, lv, grads, g_np
        cleanup()

    total_min = (time.time() - t_start) / 60
    log.info("=" * 60)
    log.info("Stage F COMPLETE: %d steps in %.1f min", step, total_min)
    log.info("  Max RSS: %dMB   Final loss: %.6f", max_rss, losses[-1])
    log.info("  Loss range: %.4f → %.4f", losses[0], losses[-1])
    log.info("  All finite: %s", all(np.isfinite(v) for v in losses))
    log.info("=" * 60)
    return losses


# ── CLI ───────────────────────────────────────────────────────
STAGES = {
    "A": ("Render encoder (3 conv2d)", stage_a),
    "B": ("Render + Scene + Fusion", stage_b),
    "C": ("Full U-Net (11 conv2d)", stage_c),
    "D": ("Scale LATENT sweep", stage_d),
    "E": ("Scale Resolution sweep", stage_e),
    "F": ("60-min sustained", stage_f),
}


def main():
    if accelerator_count() == 0:
        log.info("No GPU — aborting")
        return

    parser = argparse.ArgumentParser(description="Omen GPU scale-up test")
    parser.add_argument("--stage", choices=list(STAGES.keys()), help="Run single stage")
    parser.add_argument(
        "--full", action="store_true", help="Run full model sweep (C + D)"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Steps per stage (default 10)"
    )
    parser.add_argument(
        "--minutes", type=int, default=60, help="Stage F duration (default 60)"
    )
    parser.add_argument(
        "--latent", type=int, default=256, help="Latent dim for stages (default 256)"
    )
    parser.add_argument(
        "--res", type=int, default=64, help="Resolution for stages (default 64)"
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Omen JEPA GPU Scale-Up Test")
    log.info("Device: %s", dev())
    log.info("Model: 11 conv2d (Conv2dMojoOp) + linear + fusion")
    log.info("Data: REAL Mitsuba renders (NOT Cornell Box, NOT random noise)")
    log.info("Target: find RAM ceiling at %d-%d GB", WARN_MB // 1024, KILL_MB // 1024)
    log.info("=" * 60)
    guard("start")

    if args.stage:
        name, fn = STAGES[args.stage]
        log.info("Running single stage %s: %s", args.stage, name)
        fn(res=args.res, latent=args.latent, steps=args.steps, target_min=args.minutes)
    elif args.full:
        log.info("Running FULL model sweep (C + D)")
        stage_c(res=args.res, latent=args.latent, steps=args.steps)
        stage_d(res=args.res, steps=max(3, args.steps // 2))
    else:
        log.info("Running ALL stages A-F")
        stage_a(res=args.res, latent=args.latent, steps=args.steps)
        stage_b(res=args.res, latent=args.latent, steps=args.steps)
        stage_c(res=args.res, latent=args.latent, steps=args.steps)
        stage_d(res=args.res, steps=max(3, args.steps // 2))
        stage_e(latent=args.latent, steps=max(3, args.steps // 2))
        stage_f(res=args.res, latent=args.latent, target_min=args.minutes)

    log.info("=" * 60)
    log.info("ALL DONE — Omen GPU scale-up complete")
    log.info("System: RTX 3060 12GB, 32GB RAM")
    log.info("Ultimate target: 1B params on this GPU")
    log.info("=" * 60)
    guard("final")


if __name__ == "__main__":
    main()
