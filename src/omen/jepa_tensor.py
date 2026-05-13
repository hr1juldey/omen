"""Tensor conversion utilities for Nabla/Mitsuba interop.

DLPack zero-copy when both on GPU, numpy fallback for CPU.
"""

import logging
import os

import numpy as np

logger = logging.getLogger("omen.jepa_tensor")

CHECKPOINT_DIR = os.path.expanduser("~/.omen/checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.omen")

try:
    import nabla as nb

    NABLA_AVAILABLE = True
except ImportError:
    nb = None
    NABLA_AVAILABLE = False


def to_nabla(array, is_gpu: bool = False):
    """Convert numpy or Dr.Jit array to Nabla tensor via DLPack."""
    if not NABLA_AVAILABLE:
        raise RuntimeError("Nabla not available")

    if hasattr(array, "__dlpack__"):
        try:
            return nb.Tensor.from_dlpack(array)
        except Exception:
            pass

    if not isinstance(array, np.ndarray):
        array = np.array(array)

    tensor = nb.ndarray(array)

    if is_gpu:
        try:
            tensor = tensor.cuda()
        except Exception:
            pass

    return tensor


def to_numpy(tensor) -> np.ndarray:
    """Convert Nabla tensor to numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.to_numpy() if hasattr(tensor, "to_numpy") else np.array(tensor)


def add_alpha(render: np.ndarray) -> np.ndarray:
    """Add alpha channel to RGB render -> RGBA."""
    if render.ndim == 3 and render.shape[-1] == 3:
        alpha = np.ones((*render.shape[:2], 1), dtype=render.dtype)
        return np.concatenate([render, alpha], axis=-1)
    return render
