"""GPU-safe activation functions — pure nabla decomposition.

Two nabla bugs block GPU backward passes:
1. nb.silu VJP creates `sub(1.0, sig_x)` — Python scalar → CPU tensor → device mismatch
2. UnaryOperation subclass hits `TensorValue.num_shards` AttributeError during graph tracing

Fix: decompose silu into pure nabla ops (multiply + sigmoid). Nabla's autodiff
computes the backward via chain rule through each op individually, bypassing both bugs.
"""

import logging

logger = logging.getLogger("omen.kernels.activations")

try:
    import nabla as nb

    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False


def silu_gpu(x):
    """GPU-safe SiLU activation via pure nabla decomposition.

    x * sigmoid(x) — no custom op, no UnaryOperation, no scalar constants.
    Drop-in replacement for nb.silu that works on GPU backward passes.
    """
    if not NABLA_AVAILABLE:
        raise ImportError("Nabla required for silu_gpu")
    return x * nb.sigmoid(x)
