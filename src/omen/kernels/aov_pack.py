"""Python bridge for aov_pack Mojo GPU kernel.

Packs Mitsuba multi-channel AOV render into (H, W, 10) aux buffer.
Uses Nabla's call_custom_kernel API, falls back to numpy.
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

logger = logging.getLogger("omen.kernels.aov_pack")

PACKED_CH = 10
KERNEL_DIR = Path(__file__).parent


class AOVPackOp(UnaryOperation):
    """Nabla operation wrapping the Mojo AOV pack kernel."""

    name = "aov_pack"

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        x = args[0]
        h, w = x.shape[0], x.shape[1]
        return [(h, w, PACKED_CH)], [x.dtype], [x.device]

    def kernel(self, x, **kwargs):
        return call_custom_kernel("aov_pack", str(KERNEL_DIR), x, x.type)


def compute_aov_pack_gpu(source_np: np.ndarray) -> np.ndarray:
    """Pack Mitsuba AOV channels via Nabla + Mojo GPU kernel.

    Args:
        source_np: (H, W, C) multi-channel Mitsuba render, C >= 10

    Returns:
        (H, W, 10) packed aux buffer
    """
    if not NABLA_AVAILABLE:
        return compute_aov_pack_numpy(source_np)

    try:
        import nabla as nb

        tensor = nb.Tensor.from_dlpack(source_np.astype(np.float32))
        op = AOVPackOp()
        result = op(tensor)
        return result.to_numpy()
    except Exception as exc:
        logger.warning("AOV pack Mojo kernel failed (%s) — numpy fallback", exc)
        return compute_aov_pack_numpy(source_np)


def compute_aov_pack_numpy(source: np.ndarray) -> np.ndarray:
    """Pure numpy fallback: pack Mitsuba AOV into (H, W, 10)."""
    h, w, _ = source.shape
    packed = np.zeros((h, w, PACKED_CH), dtype=np.float32)
    # ch 0-2 -> albedo (source ch 3-5)
    packed[:, :, :3] = source[:, :, 3:6]
    # ch 3-5 -> normal (source ch 6-8)
    packed[:, :, 3:6] = source[:, :, 6:9]
    # ch 6 -> depth (source ch 9)
    packed[:, :, 6] = source[:, :, 9]
    return packed
