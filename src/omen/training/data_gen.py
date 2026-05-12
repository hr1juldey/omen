"""Training data generation using Dr.Jit and Mitsuba differentiable rendering.

Generates training pairs:
- Denoiser: 4spp noisy + 256spp GT (same seed)
- Confidence: 8x 4spp renders -> variance map
- Multires: 25% 256spp + 100% 4spp + 100% 256spp GT
- Temporal: consecutive animation frames at 4spp + 256spp
"""

import logging
import numpy as np

try:
    import mitsuba as mi
    import drjit as dr
    MITSUBA_AVAILABLE = True
except ImportError:
    MITSUBA_AVAILABLE = False

logger = logging.getLogger("omen.training.data_gen")


def verify_ad_variant():
    """Verify Mitsuba variant supports autodiff."""
    if not MITSUBA_AVAILABLE:
        raise ImportError("Mitsuba required for training data generation")

    variant = mi.variant()
    if '_ad_' not in variant:
        raise RuntimeError(
            f"Training requires AD variant (cuda_ad_rgb or llvm_ad_rgb), "
            f"got '{variant}'"
        )
    logger.info("AD variant confirmed: %s", variant)


def generate_denoiser_pair(scene, seed=42, spp_noisy=4, spp_gt=256):
    """Generate a single denoiser training pair.

    Same seed ensures identical sample positions, only spp differs.

    Returns:
        (noisy_rgba, gt_rgba) tuple of numpy arrays (H, W, 4)
    """
    noisy = mi.render(scene, spp=spp_noisy, seed=seed)
    gt = mi.render(scene, spp=spp_gt, seed=seed)

    noisy_np = np.array(noisy)
    gt_np = np.array(gt)

    h, w = noisy_np.shape[0], noisy_np.shape[1]

    # Add alpha
    noisy_rgba = np.concatenate([noisy_np, np.ones((h, w, 1))], axis=-1)
    gt_rgba = np.concatenate([gt_np, np.ones((h, w, 1))], axis=-1)

    return noisy_rgba, gt_rgba


def generate_confidence_data(scene, num_renders=8, spp=4):
    """Generate variance-based confidence training data.

    Renders the same scene multiple times with different seeds,
    computes per-pixel variance as ground truth uncertainty.

    Returns:
        (noisy_render, confidence_label) tuple
    """
    renders = []
    for i in range(num_renders):
        r = mi.render(scene, spp=spp, seed=i)
        renders.append(np.array(r))

    # Stack and compute variance
    stack = np.stack(renders, axis=0)  # (N, H, W, 3)
    variance = np.var(stack, axis=0)  # (H, W, 3)
    uncertainty = np.mean(variance, axis=-1)  # (H, W)

    # Normalize to [0, 1] and invert for confidence
    if uncertainty.max() > 0:
        confidence_label = 1.0 - (uncertainty / uncertainty.max())
    else:
        confidence_label = np.ones_like(uncertainty)

    # Return first render as input
    noisy = renders[0]
    h, w = noisy.shape[0], noisy.shape[1]
    noisy_rgba = np.concatenate([noisy, np.ones((h, w, 1))], axis=-1)

    return noisy_rgba, confidence_label.astype(np.float32)


def generate_multires_triplet(scene, scale=4, spp_high=256, spp_low=4):
    """Generate multi-resolution training triplet.

    Returns:
        (low_res_clean, high_res_noisy, ground_truth) tuple
    """
    sensor = scene.sensors()[0]
    params = mi.traverse(sensor)
    original_size = list(params['film.size'])
    width, height = int(original_size[0]), int(original_size[1])

    # Low-res clean
    params['film.size'] = [width // scale, height // scale]
    params.update()
    low_res = np.array(mi.render(scene, spp=spp_high))

    # High-res noisy
    params['film.size'] = [width, height]
    params.update()
    high_res = np.array(mi.render(scene, spp=spp_low))

    # Ground truth at full res
    gt = np.array(mi.render(scene, spp=spp_high))

    # Add alpha
    lr_h, lr_w = low_res.shape[0], low_res.shape[1]
    low_rgba = np.concatenate([low_res, np.ones((lr_h, lr_w, 1))], axis=-1)
    high_rgba = np.concatenate([high_res, np.ones((height, width, 1))], axis=-1)
    gt_rgba = np.concatenate([gt, np.ones((height, width, 1))], axis=-1)

    return low_rgba, high_rgba, gt_rgba


def generate_temporal_pair(scene, frame_t_params, frame_t1_params, spp=4, spp_gt=256):
    """Generate temporal training pair (frame T, frame T+1).

    Args:
        scene: Mitsuba scene
        frame_t_params: dict of scene params for frame T
        frame_t1_params: dict of scene params for frame T+1

    Returns:
        (latent_T, latent_T1, gt_T1, delta) tuple
    """
    # Render frame T
    params = mi.traverse(scene)
    for k, v in frame_t_params.items():
        params[k] = v
    params.update()
    render_t = np.array(mi.render(scene, spp=spp))

    # Render frame T+1
    for k, v in frame_t1_params.items():
        params[k] = v
    params.update()
    render_t1 = np.array(mi.render(scene, spp=spp))
    gt_t1 = np.array(mi.render(scene, spp=spp_gt))

    # Compute delta
    delta = {k: frame_t1_params[k] - frame_t_params[k]
             for k in frame_t_params if k in frame_t1_params}

    # Add alpha
    h, w = render_t.shape[0], render_t.shape[1]
    rgba_t = np.concatenate([render_t, np.ones((h, w, 1))], axis=-1)
    rgba_t1 = np.concatenate([render_t1, np.ones((h, w, 1))], axis=-1)
    gt_rgba = np.concatenate([gt_t1, np.ones((h, w, 1))], axis=-1)

    return rgba_t, rgba_t1, gt_rgba, delta


def random_camera_orbit(scene, frame_idx, total_frames=100):
    """Compute random camera position using spherical coordinates for orbit.

    Returns:
        dict with 'sensor.to_world' transform for the given frame
    """
    import math

    # Spherical orbit
    theta = 2 * math.pi * frame_idx / total_frames
    phi = math.pi / 4  # 45 degrees elevation
    radius = 2.8

    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.cos(phi)
    z = radius * math.sin(phi) * math.sin(theta)

    # Look-at transform (towards origin)
    # Simplified: return position for scene parameter update
    return {
        'sensor.to_world': mi.ScalarTransform4f.look_at(
            origin=[x, y, z],
            target=[0, 0, 0],
            up=[0, 1, 0]
        )
    }
