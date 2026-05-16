"""GPU-safe activation functions — pure nabla ops with scalar-free VJPs.

Nabla ops whose VJPs create `sub(1.0, ...)` fail on GPU backward because
`ensure_tensor(1.0)` creates a CPU constant → MAX compiler rejects mixed devices.
Affected: sigmoid, tanh, silu, gelu, relu.

Safe ops (verified GPU backward): exp, mul, add, div, neg, softmax.

GPU-safe sigmoid:  1.0 / (1.0 + exp(-x))  — numerically stable, no overflow
  - neg VJP:   -cotangent                 — no scalar
  - exp VJP:   cotangent * exp(x)         — no scalar
  - add VJP:   pass-through               — no scalar
  - div VJP:   mul + neg                   — no scalar
"""

import logging

logger = logging.getLogger("omen.kernels.activations")

try:
    import nabla as nb

    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False


def sigmoid_gpu(x):
    """GPU-safe sigmoid via exp — scalar-free backward, numerically stable.

    sigmoid(x) = 1.0 / (1.0 + exp(-x))
    Stable: no overflow for large positive x (exp(-x) → 0).
    Avoids nb.sigmoid whose VJP creates sub(1.0, output) → CPU scalar.
    """
    if not NABLA_AVAILABLE:
        raise ImportError("Nabla required for sigmoid_gpu")
    return 1.0 / (1.0 + nb.exp(nb.neg(x)))


def silu_gpu(x):
    """GPU-safe SiLU via exp-based sigmoid — scalar-free backward.

    silu(x) = x * sigmoid_gpu(x)
    Drop-in replacement for nb.silu that works on GPU backward passes.
    """
    if not NABLA_AVAILABLE:
        raise ImportError("Nabla required for silu_gpu")
    return x * sigmoid_gpu(x)
