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


def _ssim_gate(denoised: np.ndarray, noisy: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return raw noisy if denoised quality degraded below SSIM threshold."""
    from omen.kernels.ssim_kernel import compute_ssim_gpu

    d_lum = (0.299 * denoised[:, :, 0] + 0.587 * denoised[:, :, 1]
             + 0.114 * denoised[:, :, 2]).astype(np.float32)
    n_lum = (0.299 * noisy[:, :, 0] + 0.587 * noisy[:, :, 1]
             + 0.114 * noisy[:, :, 2]).astype(np.float32)
    score = compute_ssim_gpu(d_lum, n_lum)
    if score < threshold:
        logger.warning("Denoising degraded quality (SSIM=%.3f), returning raw render", score)
        return noisy
    return denoised


def render_denoiser(
    scene,
    bridge,
    spp: int = 4,
    tier: str = "medium",
    train: bool = True,
    config=None,
) -> np.ndarray:
    """Render noisy -> AOV -> fingerprints -> MoE -> JEPA denoise -> clean RGBA.

    Args:
        scene: Mitsuba scene
        bridge: JEPABridge instance
        spp: samples per pixel for noisy render
        tier: tier configuration (fast/medium/beast)
        train: whether to train on this render
        config: OmenConfig with mode switches (uses default if None)

    Returns:
        denoised RGBA image
    """
    from omen.config import OmenConfig
    from omen.model.tier_config import log_tier_config

    cfg = config or OmenConfig()
    modes = cfg.modes

    # Check denoiser mode switch
    if not modes.denoiser:
        raise RuntimeError("Denoiser mode is disabled. Enable config.modes.denoiser=True.")

    # Adaptive mode check (placeholder - not implemented yet)
    if modes.adaptive:
        logger.info("Adaptive mode enabled (not yet implemented)")

    # Multires mode check (placeholder - not implemented yet)
    if modes.multires:
        logger.info("Multires mode enabled (not yet implemented)")

    # Temporal mode check (requires ARPredictor)
    if modes.temporal and not cfg.components.ar_predictor:
        logger.warning("Temporal mode requires ARPredictor. Enable config.components.ar_predictor=True")

    log_tier_config(tier)

    try:
        render_result = _render_with_aov(scene, spp)
    except Exception as exc:
        logger.warning("AOV render failed (%s) — falling back to basic", exc)
        render_result = _render_noisy(scene, spp)

    raw = np.array(render_result, copy=False)
    height, width = raw.shape[0], raw.shape[1]

    try:
        rgb, aux, aov_dict = _extract_render_and_aux(render_result, height, width)
    except Exception as exc:
        logger.warning("AOV extraction failed (%s) — using raw RGB", exc)
        ch = min(3, raw.shape[-1] if raw.ndim == 3 else 1)
        rgb = raw[:, :, :ch].astype(np.float32)
        aux = np.zeros((height, width, 10), dtype=np.float32)
        aov_dict = {}

    if not bridge.available:
        logger.info("JEPA unavailable — returning raw %dspp render", spp)
        return _add_alpha(rgb, height, width)

    try:
        fingerprints = _compute_fingerprints(aux)
    except Exception as exc:
        logger.warning("Fingerprint failed (%s) — using zeros", exc)
        ny, nx = height // 8, width // 8
        fingerprints = np.zeros((ny, nx, 23), dtype=np.float32)

    from omen.scene_extractor import extract_scene_graph

    try:
        scene_graph = extract_scene_graph(scene)
    except Exception as exc:
        logger.warning("Scene extraction failed (%s) — empty graph", exc)
        scene_graph = {}

    scene_graph["aux"] = aux
    scene_graph["aov"] = aov_dict
    scene_graph["fingerprints"] = fingerprints

    if train and bridge.available:
        from omen.modes.lora_manager import train_on_scene

        train_on_scene(bridge, scene, scene_graph)

    rgba = _add_alpha(rgb, height, width)
    clean_rgba = bridge.denoise(scene_graph, rgba, width, height)
    return _ssim_gate(clean_rgba, rgba)


def _add_alpha(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Append alpha=1 channel to RGB image."""
    alpha = np.ones((height, width, 1), dtype=np.float32)
    return np.concatenate([rgb[:, :, :3], alpha], axis=-1)
