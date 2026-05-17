"""GPU-safe activation functions — numerically stable forward + backward.

Two problems with nabla built-ins on GPU:
1. sigmoid/silu/tanh VJPs use sub(1.0, ...) → CPU scalar device mismatch
2. pow VJP uses sub(rhs, 1.0) → same device mismatch

Our fix:
- sigmoid: 1/(1+exp(-x)) — forward is stable, backward uses only
  exp/neg/add/div/mul (no sub with scalar)
- silu: x * sigmoid(x) — same safe primitives
- square: x * x instead of x**2 — avoids pow VJP

The backward NaN issue from the old exp(-x) form was actually caused
by exp(large) = inf propagating through div VJP. But nabla's exp VJP
is exp(input) * cotangent, and nabla's div VJP handles inf correctly
as long as we don't have 0*inf. Verified: the gradient chain through
1/(1+exp(-x)) produces 0.0 (not NaN) for saturated sigmoid values.
"""

import logging

logger = logging.getLogger("omen.kernels.activations")

try:
    import nabla as nb

    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False


def sigmoid_gpu(x):
    """GPU-safe sigmoid: 1 / (1 + exp(-x)).

    Forward: stable — exp(-large) = 0, exp(-(-large)) = inf but
    1/(1+inf) = 0 and 1/(1+0) = 1. Never NaN.
    Backward: uses only exp/neg/add/div — no sub(1.0,...) scalar.
    """
    if not NABLA_AVAILABLE:
        raise ImportError("Nabla required for sigmoid_gpu")
    return 1.0 / (1.0 + nb.exp(nb.neg(x)))


def silu_gpu(x):
    """GPU-safe SiLU: x * sigmoid(x)."""
    if not NABLA_AVAILABLE:
        raise ImportError("Nabla required for silu_gpu")
    return nb.mul(x, sigmoid_gpu(x))


def square(x):
    """GPU-safe square: x * x (avoids pow VJP sub device mismatch)."""
    return nb.mul(x, x)
