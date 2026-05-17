"""Functional confidence head — JAX-style forward pass.

Mirrors ConfidenceHead.forward but takes params dict directly.
"""

import nabla as nb

from omen.kernels.activations import sigmoid_gpu, silu_gpu


def confidence_fn(p, latent, h, w):
    """Predict per-pixel confidence map using params dict.

    Args:
        p: prefix-stripped params with net.0/2/4.{weight,bias}.
        latent: (batch, latent_dim)
        h: height of output map
        w: width of output map

    Returns:
        (batch, h, w, 1) confidence map.
    """
    # Sequential: Linear(1024,96) -> SiLU -> Linear(96,48) -> SiLU -> Linear(48,1) -> Sigmoid
    x = latent @ p["net.0.weight"] + p["net.0.bias"]
    x = silu_gpu(x)
    x = x @ p["net.2.weight"] + p["net.2.bias"]
    x = silu_gpu(x)
    x = x @ p["net.4.weight"] + p["net.4.bias"]
    conf = sigmoid_gpu(x)

    # Expand (B, 1) -> (B, H*W) -> (B, H, W, 1)
    conf = conf.expand(int(conf.shape[0]), h * w)
    return nb.reshape(conf, (int(conf.shape[0]), h, w, 1))
