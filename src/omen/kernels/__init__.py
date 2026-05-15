"""Python bridge for Nabla custom Mojo GPU kernels.

Mojo GPU kernels (hardware-level):
  - tile_fingerprint: 23-dim tile stats from aux buffer
  - aov_pack: Pack Mitsuba AOV channels into (H,W,10)
  - moe_dispatch: Fused expert routing + weighted combination
  - mla_compress / mla_reconstruct: 16x skip projection + SiLU
  - ssim_compute: Per-pixel SSIM with 7x7 window

Uses Nabla's call_custom_kernel API with numpy fallbacks.
"""

from omen.kernels.aov_pack import compute_aov_pack_gpu, compute_aov_pack_numpy
from omen.kernels.conv2d import conv2d_safe
from omen.kernels.mla_compress import (
    compute_mla_compress_gpu,
    compute_mla_reconstruct_gpu,
)
from omen.kernels.moe_dispatch import (
    compute_moe_dispatch_gpu,
    compute_moe_dispatch_numpy,
)
from omen.kernels.ssim_kernel import compute_ssim_gpu, compute_ssim_map_gpu

__all__ = [
    "compute_aov_pack_gpu",
    "compute_aov_pack_numpy",
    "compute_mla_compress_gpu",
    "compute_mla_reconstruct_gpu",
    "compute_moe_dispatch_gpu",
    "compute_moe_dispatch_numpy",
    "compute_ssim_gpu",
    "compute_ssim_map_gpu",
    "compute_tile_fingerprint_gpu",
    "conv2d_safe",
    "compute_tile_fingerprint_numpy",
]

import logging
from pathlib import Path

import numpy as np

try:
    import nabla as nb
    from nabla.ops import UnaryOperation, call_custom_kernel

    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False
    UnaryOperation = object
    call_custom_kernel = None

logger = logging.getLogger("omen.kernels")

TILE_SIZE = 8
FINGERPRINT_DIM = 23
KERNEL_DIR = Path(__file__).parent


class TileFingerprintOp(UnaryOperation):
    """Nabla operation wrapping the Mojo tile fingerprint kernel."""

    name = "tile_fingerprint"

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        """Required: infer output shapes without building graph nodes.

        Input: aux (H, W, 10) -> Output: (H//8, W//8, 23)
        """
        x = args[0]
        h, w = x.shape[0], x.shape[1]
        out_shape = (h // TILE_SIZE, w // TILE_SIZE, FINGERPRINT_DIM)
        return [out_shape], [x.dtype], [x.device]

    def kernel(self, x, **kwargs):
        return call_custom_kernel(
            "tile_fingerprint",
            str(KERNEL_DIR),
            x,
            x.type,
        )


def compute_tile_fingerprint_gpu(aux_np: np.ndarray) -> np.ndarray:
    """Compute tile fingerprints via Nabla + Mojo GPU kernel.

    Args:
        aux_np: (H, W, 10) packed AOV buffer as numpy float32

    Returns:
        (H//8, W//8, 23) tile fingerprints as numpy float32
    """
    if not NABLA_AVAILABLE:
        logger.warning("Nabla unavailable — falling back to numpy")
        return compute_tile_fingerprint_numpy(aux_np)

    try:
        aux_tensor = nb.Tensor.from_dlpack(aux_np.astype(np.float32))
        op = TileFingerprintOp()
        result = op(aux_tensor, {})
        return result.to_numpy()
    except Exception as exc:
        logger.warning("Mojo kernel failed (%s) — falling back to numpy", exc)
        return compute_tile_fingerprint_numpy(aux_np)


def compute_tile_fingerprint_numpy(aux: np.ndarray) -> np.ndarray:
    """Pure numpy fallback for 23-dim tile fingerprint.

    Args:
        aux: (H, W, 10) packed AOV buffer

    Returns:
        (H//8, W//8, 23) tile fingerprints
    """
    h, w, _ = aux.shape
    ny, nx = h // TILE_SIZE, w // TILE_SIZE
    fp = np.zeros((ny, nx, FINGERPRINT_DIM), dtype=np.float32)

    for ty in range(ny):
        for tx in range(nx):
            y0, x0 = ty * TILE_SIZE, tx * TILE_SIZE
            y1, x1 = y0 + TILE_SIZE, x0 + TILE_SIZE
            tile = aux[y0:y1, x0:x1]

            # Material histogram (ch 7) -> fp[0:8]
            mat_ids = tile[:, :, 7].flatten().astype(int)
            hist, _ = np.histogram(mat_ids, bins=8, range=(0, 8))
            fp[ty, tx, :8] = hist / max(hist.sum(), 1)
            # Normal variance (ch 3-5) -> fp[8:11]
            fp[ty, tx, 8:11] = np.var(tile[:, :, 3:6], axis=(0, 1))
            # Depth variance (ch 6) -> fp[11]
            fp[ty, tx, 11] = np.var(tile[:, :, 6])
            # Edge density -> fp[12]
            dx = np.abs(np.diff(tile[:, :, 6], axis=1))
            dy = np.abs(np.diff(tile[:, :, 6], axis=0))
            fp[ty, tx, 12] = (np.mean(dx > 0.01) + np.mean(dy > 0.01)) / 2
            # Dominant material -> fp[13]
            fp[ty, tx, 13] = np.argmax(hist) / 8.0
            # Mean albedo (ch 0-2) -> fp[14:17]
            fp[ty, tx, 14:17] = np.mean(tile[:, :, :3], axis=(0, 1))
            # Motion stats (ch 8-9) -> fp[17:23]
            motion = tile[:, :, 8:10]
            fp[ty, tx, 17:19] = np.mean(motion, axis=(0, 1))
            fp[ty, tx, 19:21] = np.var(motion, axis=(0, 1))
            fp[ty, tx, 21] = float(np.max(np.abs(motion)))
            fp[ty, tx, 22] = float(np.mean(np.abs(motion) > 0.1))

    return fp
