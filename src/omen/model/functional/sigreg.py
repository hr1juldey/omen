"""Functional SIGReg loss — 0 params, pure nabla ops."""

import nabla as nb

from omen.kernels.activations import square


def sigreg_fn(predicted_latent, config=None):
    """Compute variance regularization loss (pure ops, 0 params).

    Prevents representation collapse by penalizing low variance.

    Args:
        predicted_latent: (batch, dim) latent embeddings.
        config: OmenConfig (unused, kept for API consistency).

    Returns:
        scalar loss value (higher when variance is low).
    """
    eps = 1e-6
    mean = nb.mean(predicted_latent, axis=0)
    var = nb.mean(square(predicted_latent - mean), axis=0)
    std = nb.sqrt(var + eps)
    return -nb.mean(nb.log(std + eps))
