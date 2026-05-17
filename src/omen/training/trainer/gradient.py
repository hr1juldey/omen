"""Gradient clipping on pytree dicts for functional Nabla training."""

import numpy as np
import nabla as nb


def clip_grad_norm_pytree(grads, max_norm, return_norm=False):
    """Clip gradient norm on a ``{name: Tensor}`` pytree.

    Operates on numpy arrays (grads are already realized) to avoid
    creating hundreds of lazy tensor nodes for the norm computation.
    """
    total_sq = 0.0
    numpy_grads = {}
    for name, g in grads.items():
        arr = g.to_numpy()
        # Sanitize: replace NaN/Inf with zero to prevent spreading
        arr = np.where(np.isfinite(arr), arr, 0.0)
        numpy_grads[name] = arr
        total_sq += float(np.sum(arr**2))

    total_norm = total_sq**0.5
    if total_norm <= max_norm:
        if return_norm:
            return grads, total_norm
        return grads

    scale = max_norm / (total_norm + 1e-6)
    clipped = {
        name: nb.Tensor.from_dlpack(arr * scale) for name, arr in numpy_grads.items()
    }
    if return_norm:
        return clipped, total_norm
    return clipped
