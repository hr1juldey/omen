"""Temporal coherence: scene delta encoding, surprise detection, jump cuts.

Tasks 14.1-14.5: SceneDeltaEncoder, scene graph diff, surprise detection,
auto-surprise for structural changes, jump cut detection.
Used by animation.py for frame-to-frame temporal prediction.
"""

import logging

import numpy as np

logger = logging.getLogger("omen.temporal")

SURPRISE_THRESHOLD = 2.0  # z-score sigma
JUMP_CUT_TRANSLATION = 1.0  # world units
JUMP_CUT_ROTATION = 0.785  # ~45 degrees in radians
DELTA_VECTOR_SIZE = 50  # fixed-size flattened delta vector


def compute_scene_delta(prev_graph: dict, curr_graph: dict) -> dict:
    """Compute frame-to-frame scene graph diff.

    Returns dict with per-category deltas and structural change flags.
    """
    delta = {"has_structural_change": False}

    # Geometry: vertex position changes
    prev_geom = np.asarray(prev_graph.get("geometry", np.array([])))
    curr_geom = np.asarray(curr_graph.get("geometry", np.array([])))
    if prev_geom.shape != curr_geom.shape:
        delta["has_structural_change"] = True
        delta["geometry_birth"] = True
    elif prev_geom.size > 0:
        delta["geometry_delta"] = np.mean(np.abs(curr_geom - prev_geom))
    else:
        delta["geometry_delta"] = 0.0

    # Materials: value changes (not structural)
    prev_mat = np.asarray(prev_graph.get("materials", np.array([])))
    curr_mat = np.asarray(curr_graph.get("materials", np.array([])))
    if prev_mat.shape != curr_mat.shape:
        delta["has_structural_change"] = True
        delta["material_birth"] = True
    elif prev_mat.size > 0:
        delta["material_delta"] = np.mean(np.abs(curr_mat - prev_mat))
    else:
        delta["material_delta"] = 0.0

    # Lights: intensity/position changes
    prev_light = np.asarray(prev_graph.get("lights", np.array([])))
    curr_light = np.asarray(curr_graph.get("lights", np.array([])))
    if prev_light.shape != curr_light.shape:
        delta["has_structural_change"] = True
        delta["light_birth"] = True
    elif prev_light.size > 0:
        delta["light_delta"] = np.mean(np.abs(curr_light - prev_light))
    else:
        delta["light_delta"] = 0.0

    # Camera: translation + rotation
    prev_cam = np.asarray(prev_graph.get("camera", np.array([])))
    curr_cam = np.asarray(curr_graph.get("camera", np.array([])))
    if prev_cam.size > 0 and curr_cam.size > 0:
        cam_diff = curr_cam - prev_cam
        delta["camera_translation"] = float(np.linalg.norm(cam_diff[:3]))
        if cam_diff.size >= 6:
            delta["camera_rotation"] = float(np.linalg.norm(cam_diff[3:6]))
        else:
            delta["camera_rotation"] = 0.0
    else:
        delta["camera_translation"] = 0.0
        delta["camera_rotation"] = 0.0

    return delta


def detect_surprise(
    predicted_latent: np.ndarray,
    actual_latent: np.ndarray,
    running_mean: float,
    running_std: float,
) -> tuple:
    """Detect surprise via MSE z-score on latent comparison.

    Returns (is_surprise, mse, z_score, updated_mean, updated_std).
    """
    mse = float(np.mean((predicted_latent - actual_latent) ** 2))
    alpha = 0.1  # EMA smoothing
    updated_mean = (1 - alpha) * running_mean + alpha * mse
    updated_var = (1 - alpha) * (running_std ** 2) + alpha * (mse - running_mean) ** 2
    updated_std = max(updated_var ** 0.5, 1e-6)

    z_score = (mse - running_mean) / max(running_std, 1e-6) if running_std > 1e-6 else 0.0
    is_surprise = z_score > SURPRISE_THRESHOLD

    return is_surprise, mse, z_score, updated_mean, updated_std


def detect_jump_cut(delta: dict) -> bool:
    """Detect jump cut from scene delta: large camera motion."""
    translation = delta.get("camera_translation", 0.0)
    rotation = delta.get("camera_rotation", 0.0)
    return translation > JUMP_CUT_TRANSLATION or rotation > JUMP_CUT_ROTATION


def detect_auto_surprise(delta: dict) -> bool:
    """Detect auto-surprise: structural changes in scene graph.

    Triggers on: new objects, material type changes, light additions.
    """
    if delta.get("has_structural_change", False):
        logger.info("Auto-surprise: structural change detected")
        return True
    # Large geometry delta = object moving fast
    geom_delta = delta.get("geometry_delta", 0.0)
    if geom_delta > 0.5:
        logger.info("Auto-surprise: large geometry delta %.3f", geom_delta)
        return True
    return False


def flatten_delta(delta: dict) -> np.ndarray:
    """Flatten delta dict into fixed-size vector for SceneDeltaEncoder."""
    values = []
    for key in sorted(delta.keys()):
        if key == "has_structural_change":
            values.append(float(delta[key]))
            continue
        val = delta[key]
        if isinstance(val, bool):
            values.append(float(val))
        elif isinstance(val, (int, float)):
            values.append(float(val))
        elif isinstance(val, np.ndarray):
            values.extend(val.flatten()[:10].tolist())
    while len(values) < DELTA_VECTOR_SIZE:
        values.append(0.0)
    return np.array(values[:DELTA_VECTOR_SIZE], dtype=np.float32)
