"""Mode 3 - Multi-resolution render pipeline.

PASS 1: Low-res (25%) at 256spp + AOV -> clean but small
PASS 2: High-res (100%) at 4spp + AOV -> noisy but full detail
MERGE: JEPA geometry-aware merge with scene graph + fingerprints
Target: 8-16x speedup, PSNR > 30dB
"""

import logging
import time

import numpy as np

logger = logging.getLogger("omen.modes.multires")


def render_multires(scene, bridge, scale: int = 4) -> np.ndarray:
    """Render with multi-resolution pipeline.

    Args:
        scene: Mitsuba mi.Scene object
        bridge: JEPABridge instance
        scale: downscale factor (default 4 = 25% resolution)

    Returns:
        numpy array (H, W, 4) merged RGBA
    """
    import mitsuba as mi

    from omen.modes.denoiser import (
        _add_alpha,
        _compute_fingerprints,
        _extract_render_and_aux,
        _render_with_aov,
    )

    sensor = scene.sensors()[0]
    params = mi.traverse(sensor)
    original_size = list(params["film.size"])
    height, width = int(original_size[1]), int(original_size[0])

    # PASS 1: Low-res high-quality
    t0 = time.perf_counter()
    params["film.size"] = [width // scale, height // scale]
    params.update()
    low_res = mi.render(scene, spp=256)
    low_res_np = np.array(low_res)
    lr_h, lr_w = low_res_np.shape[0], low_res_np.shape[1]
    low_res_rgba = _add_alpha(low_res_np, lr_h, lr_w)
    t_pass1 = time.perf_counter() - t0

    # PASS 2: High-res noisy with AOV for merge guidance
    t1 = time.perf_counter()
    params["film.size"] = [width, height]
    params.update()

    try:
        high_res = _render_with_aov(scene, spp=4)
    except Exception:
        high_res = mi.render(scene, spp=4)

    high_res_np = np.array(high_res)
    high_res_rgba = _add_alpha(high_res_np, height, width)
    t_pass2 = time.perf_counter() - t1

    if not bridge.available:
        logger.info("JEPA unavailable — returning high-res noisy")
        return high_res_rgba

    # Extract scene graph with AOV for merge guidance
    from omen.scene_extractor import extract_scene_graph

    scene_graph = extract_scene_graph(scene)

    # Compute high-res aux + fingerprints from AOV render
    try:
        _, aux, aov_dict = _extract_render_and_aux(high_res, height, width)
        fingerprints = _compute_fingerprints(aux)
        scene_graph["aux"] = aux
        scene_graph["aov"] = aov_dict
        scene_graph["fingerprints"] = fingerprints
    except Exception:
        logger.warning("AOV extraction failed — merging without fingerprints")

    # JEPA merge: low-res clean + high-res noisy -> merged
    t2 = time.perf_counter()
    merged = bridge.merge_multires(scene_graph, low_res_rgba, high_res_rgba, scale)
    t_merge = time.perf_counter() - t2

    # Speedup measurement
    baseline_time = t_pass2 * (256 / 4)  # estimate full 256spp time
    total_time = t_pass1 + t_pass2 + t_merge
    speedup = baseline_time / max(total_time, 1e-6)
    logger.info(
        "Multires: %dx%d@256spp(%.0fms) + %dx%d@4spp(%.0fms) "
        "+ merge(%.0fms) = %.1fx speedup",
        lr_w,
        lr_h,
        t_pass1 * 1000,
        width,
        height,
        t_pass2 * 1000,
        t_merge * 1000,
        speedup,
    )

    return merged
