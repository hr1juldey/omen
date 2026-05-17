"""Functional cross-attention fusion — JAX-style forward pass.

Mirrors CrossAttentionFusion.forward but takes params dict directly.
"""

import nabla as nb

from omen.kernels.activations import sigmoid_gpu, square


def _linear(x, weight, bias):
    """Functional linear: x @ W + b."""
    return x @ weight + bias


def _layer_norm(x, weight, bias, eps=1e-5):
    """Functional layer norm — uses square() not **2 for GPU safety."""
    mean = x.mean(axis=-1, keepdims=True)
    var = square(x - mean).mean(axis=-1, keepdims=True)
    return (x - mean) / nb.sqrt(var + eps) * weight + bias


def cross_attn_fn(p, render_latent, scene_latent):
    """Gated fusion of render and scene latents.

    Args:
        p: prefix-stripped params with gate.{weight,bias}, norm.{weight,bias}.
        render_latent: (batch, latent_dim)
        scene_latent: (batch, latent_dim)

    Returns:
        (batch, latent_dim) fused latent.
    """
    g = sigmoid_gpu(_linear(render_latent, p["gate.weight"], p["gate.bias"]))
    fused = _layer_norm(
        render_latent + g * scene_latent,
        p["norm.weight"],
        p["norm.bias"],
    )
    return fused
