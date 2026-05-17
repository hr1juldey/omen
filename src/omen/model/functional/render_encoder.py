"""Functional render feature encoder — JAX-style forward pass.

Mirrors RenderFeatureEncoder.forward but takes params dict directly.
"""

from omen.kernels.activations import silu_gpu
from omen.kernels.conv2d import conv2d_safe


def render_encoder_fn(p, rgba):
    """Encode RGBA render image to latent vector using params dict.

    Args:
        p: prefix-stripped params with conv1/2/3_filter, conv1/2/3_bias, proj.
        rgba: (batch, H, W, 4) RGBA render tensor.

    Returns:
        (batch, latent_dim) render latent.
    """
    x = silu_gpu(
        conv2d_safe(rgba, p["conv1_filter"], stride=2, padding=1, bias=p["conv1_bias"])
    )
    x = silu_gpu(
        conv2d_safe(x, p["conv2_filter"], stride=2, padding=1, bias=p["conv2_bias"])
    )
    x = silu_gpu(
        conv2d_safe(x, p["conv3_filter"], stride=2, padding=1, bias=p["conv3_bias"])
    )

    # Global average pool over spatial dims -> (B, 128)
    x = x.mean(axis=(1, 2))

    return x @ p["proj.weight"] + p["proj.bias"]
