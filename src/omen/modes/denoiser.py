"""Mode 1 - Single-pass denoiser with AOV + tile MoE routing.

Pipeline: mi.render(spp=4) -> AOV extract -> tile fingerprint
          -> MoE routing -> JEPA denoise -> clean RGBA
Target: SSIM > 0.90 vs 256spp, total < 300ms at 256x256
"""

import logging

import numpy as np

logger = logging.getLogger("omen.modes.denoiser")

# Mitsuba AOV integrator channel specification
_AOV_SPEC = "albedo:albedo,normal:sh_normal,depth:dd.y"


def _render_noisy(scene, spp: int = 4):
    """Render noisy image with Mitsuba path tracer (no AOV)."""
    import mitsuba as mi

    integrator = mi.load_dict({"type": "path", "max_depth": 8})
    return mi.render(scene, integrator=integrator, sensor=0, spp=spp)


def _render_with_aov(scene, spp: int = 4):
    """Render noisy image + AOV passes via Mitsuba aov integrator."""
    import mitsuba as mi

    integrator = mi.load_dict(
        {
            "type": "aov",
            "aovs": _AOV_SPEC,
            "my_integrator": {"type": "path", "max_depth": 8},
        }
    )
    return mi.render(scene, integrator=integrator, sensor=0, spp=spp)


def _extract_render_and_aux(render_result, height: int, width: int):
    """Split render result into RGB and packed AOV buffer."""
    from omen.aov import pack_aux_buffer, read_all_aov

    data = np.array(render_result, copy=False)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    rgb = data[:, :, :3].astype(np.float32)
    aov_dict = read_all_aov(render_result)
    aux = pack_aux_buffer(aov_dict, height, width)
    return rgb, aux, aov_dict


def _compute_fingerprints(aux: np.ndarray):
    """Compute 23-dim tile fingerprints from (H,W,10) aux buffer."""
    from omen.kernels import compute_tile_fingerprint_gpu

    return compute_tile_fingerprint_gpu(aux)


def render_denoiser(scene, bridge, spp: int = 4, tier: str = "medium") -> np.ndarray:
    """Render and denoise: noisy -> AOV -> fingerprints -> MoE -> clean RGBA.

    Args:
        scene: Mitsuba mi.Scene object
        bridge: JEPABridge instance
        spp: samples per pixel (default 4)
        tier: model tier (fast/medium/high)

    Returns:
        numpy array (H, W, 4) clean RGBA
    """
    from omen.model.tier_config import log_tier_config

    log_tier_config(tier)

    # Step 1: Render noisy + AOV (fallback to basic render)
    try:
        render_result = _render_with_aov(scene, spp)
    except Exception as exc:
        logger.warning("AOV render failed (%s) — falling back to basic", exc)
        render_result = _render_noisy(scene, spp)

    raw = np.array(render_result, copy=False)
    height, width = raw.shape[0], raw.shape[1]

    # Step 2: Extract RGB + AOV buffers
    try:
        rgb, aux, aov_dict = _extract_render_and_aux(render_result, height, width)
    except Exception as exc:
        logger.warning("AOV extraction failed (%s) — using raw RGB", exc)
        ch = min(3, raw.shape[-1] if raw.ndim == 3 else 1)
        rgb = raw[:, :, :ch].astype(np.float32)
        aux = np.zeros((height, width, 10), dtype=np.float32)
        aov_dict = {}

    # Step 3: Model unavailable -> return raw with alpha
    if not bridge.available:
        logger.info("JEPA unavailable — returning raw %dspp render", spp)
        return _add_alpha(rgb, height, width)

    # Step 4: Compute tile fingerprints for MoE routing
    try:
        fingerprints = _compute_fingerprints(aux)
    except Exception as exc:
        logger.warning("Fingerprint failed (%s) — using zeros", exc)
        ny, nx = height // 8, width // 8
        fingerprints = np.zeros((ny, nx, 23), dtype=np.float32)

    # Step 5: Build scene graph with aux + fingerprints
    from omen.scene_extractor import extract_scene_graph

    try:
        scene_graph = extract_scene_graph(scene)
    except Exception as exc:
        logger.warning("Scene extraction failed (%s) — empty graph", exc)
        scene_graph = {}

    scene_graph["aux"] = aux
    scene_graph["aov"] = aov_dict
    scene_graph["fingerprints"] = fingerprints

    # Step 6: Run JEPA denoise
    rgba = _add_alpha(rgb, height, width)
    clean_rgba = bridge.denoise(scene_graph, rgba, width, height)
    return clean_rgba


def _add_alpha(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Append alpha=1 channel to RGB image."""
    alpha = np.ones((height, width, 1), dtype=np.float32)
    return np.concatenate([rgb[:, :, :3], alpha], axis=-1)
