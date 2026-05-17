"""Functional episodic correction — JAX-style forward pass.

Mirrors EpisodicCorrection.forward but takes params dict directly.
"""

import nabla as nb

from omen.kernels.activations import silu_gpu


def episodic_fn(p, main_output, scene_context):
    """Apply episodic correction using params dict.

    Args:
        p: prefix-stripped params with net.0/2.{weight,bias}.
        main_output: (batch, dim) main model output.
        scene_context: (batch, dim) scene context embedding.

    Returns:
        (batch, dim) corrected output.
    """
    combined = nb.concatenate([main_output, scene_context], axis=-1)

    # Sequential: Linear(dim*2, hidden) -> SiLU -> Linear(hidden, dim)
    x = combined @ p["net.0.weight"] + p["net.0.bias"]
    x = silu_gpu(x)
    correction = x @ p["net.2.weight"] + p["net.2.bias"]

    return main_output + correction
