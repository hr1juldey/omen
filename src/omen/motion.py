"""Motion vector processing and temporal reprojection.

Tasks 18.1-18.12: Motion vectors, temporal reprojection, coherence,
occlusion, reprojection weight, motion expert routing.
"""

import logging

import numpy as np

logger = logging.getLogger("omen.motion")

MAX_VELOCITY = 50.0


def read_motion_vectors(aov_dict: dict, height: int, width: int) -> np.ndarray:
    """Read (H, W, 2) motion vectors. Returns zeros if unavailable (task 18.10)."""
    for key in ("motion", "vector"):
        if key in aov_dict:
            mv = np.asarray(aov_dict[key])
            if mv.ndim == 3 and mv.shape[2] >= 2:
                return mv[:, :, :2].astype(np.float32)
    logger.info("No motion vectors — static denoise mode")
    return np.zeros((height, width, 2), dtype=np.float32)


def temporal_reproject(prev_clean: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Bilinear warp prev_clean using motion vectors (task 18.3)."""
    h, w, c = prev_clean.shape
    y_c, x_c = np.mgrid[0:h, 0:w].astype(np.float32)
    src_x = x_c + motion[:, :, 0]
    src_y = y_c + motion[:, :, 1]

    x0 = np.clip(np.floor(src_x).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(src_y).astype(int), 0, h - 1)
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    fx = (src_x - x0).clip(0, 1)[..., np.newaxis]
    fy = (src_y - y0).clip(0, 1)[..., np.newaxis]

    return (prev_clean[y0, x0] * (1 - fx) * (1 - fy) +
            prev_clean[y0, x1] * fx * (1 - fy) +
            prev_clean[y1, x0] * (1 - fx) * fy +
            prev_clean[y1, x1] * fx * fy).astype(np.float32)


def compute_motion_coherence(motion: np.ndarray) -> np.ndarray:
    """Per-pixel coherence: 1 - |velocity|/max_vel (task 18.4)."""
    vel = np.sqrt(motion[:, :, 0] ** 2 + motion[:, :, 1] ** 2)
    return (1.0 - np.clip(vel / MAX_VELOCITY, 0, 1)).astype(np.float32)


def compute_occlusion_mask(motion: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    """Detect occlusion via velocity discontinuity (task 18.5). Returns (H, W)."""
    vel = np.sqrt(motion[:, :, 0] ** 2 + motion[:, :, 1] ** 2)
    gx = np.abs(np.diff(vel, axis=1, prepend=vel[:, :1]))
    gy = np.abs(np.diff(vel, axis=0, prepend=vel[:1, :]))
    return ((gx + gy) > threshold).astype(np.float32)


def compute_reprojection_weight(prev_confidence: np.ndarray,
                                motion: np.ndarray) -> np.ndarray:
    """Weight: confidence * coherence * (1 - occluded). Tasks 18.4-18.6."""
    coherence = compute_motion_coherence(motion)
    occluded = compute_occlusion_mask(motion)
    return np.clip(prev_confidence * coherence * (1.0 - occluded), 0, 1).astype(np.float32)


def merge_reprojected(reprojected: np.ndarray, current_noisy: np.ndarray,
                      weight: np.ndarray) -> np.ndarray:
    """Merge: alpha * reproj + (1-alpha) * noisy. Task 18.7."""
    w = weight[..., np.newaxis]
    return (w * reprojected + (1.0 - w) * current_noisy).astype(np.float32)


def extend_fingerprint_motion(fingerprint: np.ndarray,
                              motion: np.ndarray) -> np.ndarray:
    """Extend fingerprint with 6 motion features (task 18.11).

    Adds: vel_mean(2) + vel_var(2) + vel_max(1) + occ_frac(1) = 6 dims.
    """
    tile_h, tile_w = fingerprint.shape[:2]
    mh, mw = motion.shape[0], motion.shape[1]
    sh, sw = mh // tile_h, mw // tile_w
    if sh == 0 or sw == 0:
        pad = np.zeros((*fingerprint.shape[:2], 6), dtype=np.float32)
        return np.concatenate([fingerprint, pad], axis=-1)

    tiles = motion[:tile_h * sh, :tile_w * sw].reshape(tile_h, sh, tile_w, sw, 2)
    tile_m = tiles.mean(axis=(1, 3))
    vel_var = tiles.var(axis=(1, 3))
    vel_max = np.sqrt(tile_m[:, :, 0] ** 2 + tile_m[:, :, 1] ** 2).max()

    occ = compute_occlusion_mask(motion)
    occ_tiles = occ[:tile_h * sh, :tile_w * sw].reshape(tile_h, sh, tile_w, sw)
    occ_frac = occ_tiles.mean(axis=(1, 3))

    features = np.stack([
        tile_m[:, :, 0], tile_m[:, :, 1],
        vel_var[:, :, 0], vel_var[:, :, 1],
        np.full((tile_h, tile_w), vel_max), occ_frac,
    ], axis=-1).astype(np.float32)
    return np.concatenate([fingerprint, features], axis=-1)


def route_motion_expert(fp23: np.ndarray) -> np.ndarray:
    """Route tiles to 4 motion experts. Task 18.12: static/linear/fast/occlusion."""
    if fp23.shape[-1] < 23:
        return np.zeros((*fp23.shape[:2], 4), dtype=np.float32)

    vx, vy = fp23[:, :, 17], fp23[:, :, 18]
    occ = fp23[:, :, 22]
    vel = np.sqrt(vx ** 2 + vy ** 2)

    scores = np.stack([
        1.0 / (1.0 + vel),        # static
        vel / (1.0 + vel),        # linear
        np.minimum(vel / 20, 1),  # fast
        occ,                      # occlusion
    ], axis=-1)
    return (scores / (scores.sum(axis=-1, keepdims=True) + 1e-6)).astype(np.float32)
