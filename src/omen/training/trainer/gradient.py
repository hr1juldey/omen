"""Gradient clipping on pytree dicts for functional Nabla training."""

from nabla import tree_map


def clip_grad_norm_pytree(grads, max_norm):
    """Clip gradient norm on a ``{name: Tensor}`` pytree.

    Unlike the imperative version that reads ``p.grad``, this operates
    directly on the gradient dict returned by ``nb.value_and_grad``.

    Args:
        grads: ``{name: Tensor}`` gradient pytree.
        max_norm: Maximum allowed global L2 norm.

    Returns:
        Clipped gradient pytree with the same structure.
    """
    total_sq = 0.0
    for g in grads.values():
        total_sq = total_sq + (g**2).sum().to_numpy().item()
    total_norm = total_sq**0.5

    if total_norm <= max_norm:
        return grads

    scale = max_norm / (total_norm + 1e-6)
    return tree_map(lambda g: g * scale, grads)
