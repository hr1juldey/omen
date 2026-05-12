"""Mode 3 - Multi-resolution render pipeline.

Pipeline:
PASS 1: Low-res (25%) at 256spp -> clean but small
PASS 2: High-res (100%) at 4spp -> noisy but full detail
MERGE: JEPA geometry-aware merge with scene graph
Target: 8-16x speedup, PSNR > 30dB
"""

import logging
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

    sensor = scene.sensors()[0]
    params = mi.traverse(sensor)

    # Get original film size
    original_size = list(params['film.size'])
    height, width = int(original_size[1]), int(original_size[0])

    # PASS 1: Low-res high-quality
    params['film.size'] = [width // scale, height // scale]
    params.update()
    low_res = mi.render(scene, spp=256)
    low_res_np = np.array(low_res)
    lr_h, lr_w = low_res_np.shape[0], low_res_np.shape[1]

    # Add alpha to low-res
    alpha = np.ones((lr_h, lr_w, 1), dtype=low_res_np.dtype)
    low_res_rgba = np.concatenate([low_res_np, alpha], axis=-1)

    # PASS 2: High-res noisy
    params['film.size'] = [width, height]
    params.update()
    high_res = mi.render(scene, spp=4)
    high_res_np = np.array(high_res)

    # Add alpha to high-res
    alpha = np.ones((height, width, 1), dtype=high_res_np.dtype)
    high_res_rgba = np.concatenate([high_res_np, alpha], axis=-1)

    if not bridge.available:
        logger.info("JEPA unavailable, returning high-res noisy")
        return high_res_rgba

    # Extract scene graph
    from omen.scene_extractor import extract_scene_graph
    scene_graph = extract_scene_graph(scene)

    # JEPA merge
    merged = bridge.merge_multires(scene_graph, low_res_rgba, high_res_rgba, scale)

    logger.info(
        "Multires: %dx%d@256spp + %dx%d@4spp -> %dx%d merged",
        lr_w, lr_h, width, height, width, height
    )

    return merged
