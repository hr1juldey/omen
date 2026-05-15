"""Python bridge for mla_compress / mla_reconstruct Mojo GPU kernels.

Fuses Linear projection + SiLU activation for U-Net skip compression.
Reduces skip memory 16x: (N, C) -> (N, C//16).

Packs features + weights into a single flat tensor to avoid the
MAX framework multi-input custom kernel data transfer bug.
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


def _pack_tensors(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pack data + weights into single flat tensor with metadata header.

    Layout: [N, C_in, C_latent, data_flat, weights_flat]
    """
    n, c_in = data.shape
    _, c_lat = weights.shape
    header = np.array([n, c_in, c_lat], dtype=np.float32)
    return np.concatenate([header, data.flatten(), weights.flatten()])


class MLACompressOp(UnaryOperation):
    """Nabla op wrapping Mojo mla_compress kernel (single packed input)."""

    @property
    def name(self) -> str:
        return "mla_compress"

    def __init__(self, n: int, c_latent: int):
        self.n = n
        self.c_latent = c_latent

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        return [(self.n, self.c_latent)], [args[0].dtype], [args[0].device]

    def kernel(self, args, kwargs):
        from max.graph import TensorType

        source = args[0]
        out_type = TensorType(
            dtype=source.dtype, shape=(self.n, self.c_latent), device=source.device
        )
        result = call_custom_kernel("mla_compress", str(KERNEL_DIR), source, out_type)
        return [result]


class MLAReconstructOp(UnaryOperation):
    """Nabla op wrapping Mojo mla_reconstruct kernel (single packed input)."""

    @property
    def name(self) -> str:
        return "mla_reconstruct"

    def __init__(self, n: int, c_in: int):
        self.n = n
        self.c_in = c_in

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        return [(self.n, self.c_in)], [args[0].dtype], [args[0].device]

    def kernel(self, args, kwargs):
        from max.graph import TensorType

        source = args[0]
        out_type = TensorType(
            dtype=source.dtype, shape=(self.n, self.c_in), device=source.device
        )
        result = call_custom_kernel(
            "mla_reconstruct", str(KERNEL_DIR), source, out_type
        )
        return [result]


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

        f32 = features.astype(np.float32)
        w32 = weights.astype(np.float32)
        n, c_lat = f32.shape[0], w32.shape[1]

        packed = _pack_tensors(f32, w32)
        tensor = nb.Tensor.from_dlpack(packed)

        op = MLACompressOp(n=n, c_latent=c_lat)
        result = op([tensor], {})[0]
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

        c32 = compressed.astype(np.float32)
        w32 = weights.astype(np.float32)
        n, c_in = c32.shape[0], w32.shape[1]

        packed = _pack_tensors(c32, w32)
        tensor = nb.Tensor.from_dlpack(packed)

        op = MLAReconstructOp(n=n, c_in=c_in)
        result = op([tensor], {})[0]
        return result.to_numpy()
    except Exception as exc:
        logger.warning("MLA reconstruct Mojo failed (%s) — numpy fallback", exc)
        return compressed @ weights


def _mla_compress_numpy(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Numpy fallback: matmul + hard-swish (matches Mojo kernel)."""
    projected = features @ weights
    return projected * np.clip(projected / 6.0 + 0.5, 0.0, 1.0)
