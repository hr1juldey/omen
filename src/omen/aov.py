"""AOV pass reader with graceful degradation.

Reads auxiliary render passes from Mitsuba or Blender:
- albedo (3 ch): diffuse color
- normal (3 ch): world-space normals
- depth (1 ch): camera depth
- material_id (1 ch): cryptomatte object ID
- motion_vectors (2 ch): optical flow

Gracefully degrades when passes are missing — zero-fills and logs warnings.
"""

import logging

import numpy as np

try:
    import mitsuba as mi  # noqa: F401 — used for Bitmap type checks
    import drjit as dr  # noqa: F401 — used for tensor conversion

    MITSUBA_AVAILABLE = True
except ImportError:
    MITSUBA_AVAILABLE = False

logger = logging.getLogger("omen.aov")

# AOV pass definitions: name -> (channels, description)
AOV_PASSES = {
    "albedo": (3, "Diffuse albedo color"),
    "normal": (3, "World-space surface normals"),
    "depth": (1, "Camera-space depth"),
    "material_id": (1, "Cryptomatte material ID"),
    "motion_vectors": (2, "Optical flow / motion vectors"),
}


def read_aov_mitsuba(bitmap, pass_name: str) -> np.ndarray:
    """Extract a single AOV pass from Mitsuba Bitmap.

    Args:
        bitmap: Mitsuba Bitmap object with AOV channels
        pass_name: one of AOV_PASSES keys

    Returns:
        numpy array (H, W, channels) or zeros if unavailable
    """
    expected_ch, desc = AOV_PASSES.get(pass_name, (0, "unknown"))

    if not MITSUBA_AVAILABLE:
        logger.warning("Mitsuba unavailable — zero-filling %s", pass_name)
        return np.zeros((1, 1, expected_ch), dtype=np.float32)

    try:
        # Convert bitmap to numpy (H, W, C) in NHWC layout
        data = np.array(bitmap, copy=False)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        ch_offset = _channel_offset(pass_name)
        if ch_offset is not None and data.shape[-1] >= ch_offset + expected_ch:
            return data[:, :, ch_offset : ch_offset + expected_ch].astype(np.float32)

        logger.warning(
            "AOV '%s' not found in bitmap channels — zero-filling", pass_name
        )
        return np.zeros((data.shape[0], data.shape[1], expected_ch), dtype=np.float32)

    except Exception as exc:
        logger.warning("Failed to read %s: %s — zero-filling", pass_name, exc)
        return np.zeros((1, 1, expected_ch), dtype=np.float32)


def _channel_offset(pass_name: str):
    """Map pass name to channel offset in multi-channel bitmap."""
    offsets = {"albedo": 0, "normal": 3, "depth": 6}
    return offsets.get(pass_name)


def read_all_aov(bitmap) -> dict:
    """Read all available AOV passes with graceful degradation.

    Args:
        bitmap: Mitsuba Bitmap with AOV channels

    Returns:
        dict mapping pass_name -> (H, W, C) numpy arrays
    """
    result = {}
    available = []
    missing = []

    for name, (channels, _) in AOV_PASSES.items():
        data = read_aov_mitsuba(bitmap, name)
        result[name] = data
        if np.any(data != 0):
            available.append(name)
        else:
            missing.append(name)

    logger.info(
        "AOV available: %s — missing: %s (using degraded mode)",
        ", ".join(f"{n}=yes" for n in available) or "none",
        ", ".join(missing) or "none",
    )
    return result


def pack_aux_buffer(aov_dict: dict, height: int, width: int) -> np.ndarray:
    """Pack AOV passes into (H, W, 10) tensor for tile fingerprint.

    Channels: albedo(3) + normal(3) + depth(1) + material_id(1) + motion(2) = 10
    """
    parts = []
    for name in ["albedo", "normal", "depth", "material_id", "motion_vectors"]:
        data = aov_dict.get(name)
        if data is not None and data.shape[0] == height and data.shape[1] == width:
            parts.append(data)
        else:
            ch = AOV_PASSES[name][0]
            parts.append(np.zeros((height, width, ch), dtype=np.float32))

    return np.concatenate(parts, axis=-1)
