"""Mode 1 - Single-pass denoiser.

Pipeline: mi.render(spp=4) -> scene extraction -> DLPack -> JEPA denoise -> clean RGBA
Target: SSIM > 0.90 vs 256spp, total < 300ms at 256x256
"""

import logging
import numpy as np

logger = logging.getLogger("omen.modes.denoiser")


def render_denoiser(scene, bridge, spp: int = 4) -> np.ndarray:
    """Render and denoise in a single pass.

    Args:
        scene: Mitsuba mi.Scene object
        bridge: JEPABridge instance
        spp: samples per pixel for noisy render (default 4)

    Returns:
        numpy array (H, W, 4) clean RGBA
    """
    import mitsuba as mi

    # Render noisy preview
    image = mi.render(scene, sensor=0, spp=spp)
    image_np = np.array(image)

    height, width = image_np.shape[0], image_np.shape[1]

    if not bridge.available:
        logger.info("JEPA unavailable, returning raw render")
        # Add alpha channel and return
        alpha = np.ones((height, width, 1), dtype=np.float32)
        return np.concatenate([image_np, alpha], axis=-1)

    # Extract scene graph
    from omen.scene_extractor import extract_scene_graph
    scene_graph = extract_scene_graph(scene)

    # Add alpha channel
    alpha = np.ones((height, width, 1), dtype=image_np.dtype)
    rgba = np.concatenate([image_np, alpha], axis=-1)

    # Run JEPA denoise
    clean_rgba = bridge.denoise(scene_graph, rgba, width, height)

    return clean_rgba
