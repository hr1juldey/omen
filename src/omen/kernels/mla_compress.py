"""Python bridge for mla_compress / mla_reconstruct Mojo GPU kernels.

Fuses Linear projection + SiLU activation for U-Net skip compression.
Reduces skip memory 16x: (N, C) -> (N, C//16).

Uses Nabla's call_custom_kernel API, falls back to numpy matmul.
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

logger = logging.getLogger("omen.kernels.mla_compress")

KERNEL_DIR = Path(__file__).parent
MAX_CHANNELS = 512


class MLACompressOp(UnaryOperation):
    """Nabla op wrapping Mojo mla_compress kernel."""

    name = "mla_compress"

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        features = args[0]
        weights = args[1]
        n = features.shape[0]
        c_out = weights.shape[1]
        return [(n, c_out)], [features.dtype], [features.device]

    def kernel(self, features, weights, **kwargs):
        return call_custom_kernel("mla_compress", str(KERNEL_DIR), features, weights)


class MLAReconstructOp(UnaryOperation):
    """Nabla op wrapping Mojo mla_reconstruct kernel."""

    name = "mla_reconstruct"

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        compressed = args[0]
        weights = args[1]
        n = compressed.shape[0]
        c_out = weights.shape[1]
        return [(n, c_out)], [compressed.dtype], [compressed.device]

    def kernel(self, compressed, weights, **kwargs):
        return call_custom_kernel(
            "mla_reconstruct", str(KERNEL_DIR), compressed, weights
        )


def compute_mla_compress_gpu(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compress skip features via Mojo GPU kernel.

    Args:
        features: (N, C_in) encoder features
        weights: (C_in, C_latent) projection matrix

    Returns:
        (N, C_latent) compressed features with SiLU activation
    """
    if not NABLA_AVAILABLE:
        return _mla_compress_numpy(features, weights)

    try:
        import nabla as nb

        f_tensor = nb.Tensor.from_dlpack(features.astype(np.float32))
        w_tensor = nb.Tensor.from_dlpack(weights.astype(np.float32))
        op = MLACompressOp()
        result = op([f_tensor, w_tensor], {})
        return result.to_numpy()
    except Exception as exc:
        logger.warning("MLA compress Mojo failed (%s) — numpy fallback", exc)
        return _mla_compress_numpy(features, weights)


def compute_mla_reconstruct_gpu(
    compressed: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Reconstruct skip features via Mojo GPU kernel.

    Args:
        compressed: (N, C_latent) compressed latent
        weights: (C_latent, C_in) up-projection matrix

    Returns:
        (N, C_in) reconstructed features
    """
    if not NABLA_AVAILABLE:
        return compressed @ weights

    try:
        import nabla as nb

        c_tensor = nb.Tensor.from_dlpack(compressed.astype(np.float32))
        w_tensor = nb.Tensor.from_dlpack(weights.astype(np.float32))
        op = MLAReconstructOp()
        result = op([c_tensor, w_tensor], {})
        return result.to_numpy()
    except Exception as exc:
        logger.warning("MLA reconstruct Mojo failed (%s) — numpy fallback", exc)
        return compressed @ weights


def _mla_compress_numpy(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Numpy fallback: matmul + SiLU."""
    projected = features @ weights
    return projected * (1.0 / (1.0 + np.exp(-projected)))  # SiLU
