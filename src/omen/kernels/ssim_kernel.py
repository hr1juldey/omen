"""Python bridge for ssim_compute Mojo GPU kernel.

Computes per-pixel SSIM map using 7x7 uniform window on GPU.
Replaces the cumsum-based numpy SSIM in omen.modes.quality.
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


class SSIMComputeOp(UnaryOperation):
    """Nabla op wrapping Mojo ssim_compute kernel."""

    @property
    def name(self) -> str:
        return "ssim_compute"

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        img1 = args[0]
        h, w = img1.shape
        return [(h, w)], [img1.dtype], [img1.device]

    def kernel(self, args, kwargs):
        from max.graph import TensorType

        img1, img2 = args[0], args[1]
        h, w = int(img1.shape[0]), int(img1.shape[1])
        out_type = TensorType(dtype=img1.dtype, shape=(h, w), device=img1.device)
        result = call_custom_kernel("ssim_compute", str(KERNEL_DIR), [img1, img2], out_type)
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

        # Pad images for boundary handling (reflect pad, size 2)
        pad = 2
        i1 = np.pad(img1.astype(np.float32), pad, mode="reflect")
        i2 = np.pad(img2.astype(np.float32), pad, mode="reflect")

        t1 = nb.Tensor.from_dlpack(i1)
        t2 = nb.Tensor.from_dlpack(i2)
        op = SSIMComputeOp()
        result = op([t1, t2], {})[0]
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

    pad = 2
    i1 = np.pad(img1.astype(np.float32), pad, mode="reflect")
    i2 = np.pad(img2.astype(np.float32), pad, mode="reflect")
    t1 = nb.Tensor.from_dlpack(i1)
    t2 = nb.Tensor.from_dlpack(i2)
    op = SSIMComputeOp()
    result = op([t1, t2], {})[0]
    return result.to_numpy()


def _ssim_numpy_fallback(img1: np.ndarray, img2: np.ndarray) -> float:
    """Minimal numpy SSIM fallback for when Nabla unavailable."""
    from omen.modes.quality import ssim

    return ssim(img1, img2)
