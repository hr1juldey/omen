"""Python bridge for ssim_compute Mojo GPU kernel.

Computes per-pixel SSIM map using 7x7 uniform window on GPU.
Packs both images into a single (2, H+6, W+6) tensor to avoid the
MAX framework multi-input custom kernel data transfer bug.
Output is original (unpadded) image size (H, W).
"""

import logging
from pathlib import Path

import numpy as np

try:
    from nabla.ops import UnaryOperation, call_custom_kernel

    NABLA_AVAILABLE = True
except ImportError:
    UnaryOperation = object
    NABLA_AVAILABLE = False

logger = logging.getLogger("omen.kernels.ssim_kernel")

KERNEL_DIR = Path(__file__).parent
_PAD = 3  # (WIN-1)//2 = 3 for symmetric 7x7 window


class SSIMComputeOp(UnaryOperation):
    """Nabla op wrapping Mojo ssim_compute kernel.

    Receives a single (2, H+6, W+6) stacked tensor containing both images.
    Output is original (H, W) size.
    """

    @property
    def name(self) -> str:
        return "ssim_compute"

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        source = args[0]
        # source is (2, H+6, W+6) — output is (H, W) = original size
        h = int(source.shape[1]) - 2 * _PAD
        w = int(source.shape[2]) - 2 * _PAD
        return [(h, w)], [source.dtype], [source.device]

    def kernel(self, args, kwargs):
        from max.graph import TensorType

        source = args[0]
        h = int(source.shape[1]) - 2 * _PAD
        w = int(source.shape[2]) - 2 * _PAD
        out_type = TensorType(dtype=source.dtype, shape=(h, w), device=source.device)
        result = call_custom_kernel("ssim_compute", str(KERNEL_DIR), source, out_type)
        return [result]


def compute_ssim_gpu(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute mean SSIM via Mojo GPU kernel.

    Args:
        img1: (H, W) float32 image
        img2: (H, W) float32 image

    Returns:
        Mean SSIM score (0.0 to 1.0)
    """
    if not NABLA_AVAILABLE:
        return _ssim_numpy_fallback(img1, img2)

    try:
        import nabla as nb

        # Pad images for boundary handling (reflect pad, size 3)
        i1 = np.pad(img1.astype(np.float32), _PAD, mode="reflect")
        i2 = np.pad(img2.astype(np.float32), _PAD, mode="reflect")

        # Stack into single (2, H+6, W+6) tensor — avoids multi-input bug
        stacked_np = np.stack([i1, i2], axis=0)
        stacked = nb.Tensor.from_dlpack(stacked_np)

        op = SSIMComputeOp()
        result = op([stacked], {})[0]
        ssim_map = result.to_numpy()
        return float(np.mean(ssim_map))
    except Exception as exc:
        logger.warning("SSIM Mojo kernel failed (%s) — numpy fallback", exc)
        return _ssim_numpy_fallback(img1, img2)


def compute_ssim_map_gpu(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Compute per-pixel SSIM map via Mojo GPU kernel.

    Returns:
        (H, W) SSIM map
    """
    if not NABLA_AVAILABLE:
        raise RuntimeError("Nabla unavailable for SSIM map computation")

    import nabla as nb

    i1 = np.pad(img1.astype(np.float32), _PAD, mode="reflect")
    i2 = np.pad(img2.astype(np.float32), _PAD, mode="reflect")

    stacked_np = np.stack([i1, i2], axis=0)
    stacked = nb.Tensor.from_dlpack(stacked_np)

    op = SSIMComputeOp()
    result = op([stacked], {})[0]
    return result.to_numpy()


def _ssim_numpy_fallback(img1: np.ndarray, img2: np.ndarray) -> float:
    """Minimal numpy SSIM fallback for when Nabla unavailable."""
    from omen.modes.quality import ssim

    return ssim(img1, img2)
