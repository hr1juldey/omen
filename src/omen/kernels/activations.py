"""GPU-safe activation functions — numerically stable forward + backward.

Problems with nabla built-ins on GPU:
1. sigmoid/silu/tanh VJPs use sub(1.0, ...) → CPU scalar device mismatch
2. pow VJP uses sub(rhs, 1.0) → same device mismatch
3. sqrt VJP uses mul(2.0, ...) / div(1.0, ...) → same device mismatch
4. log VJP uses div(1.0, ...) → same device mismatch

Our fixes use only ops with GPU-safe VJPs: exp, mul, add, sub, neg, div.
- sigmoid: 1/(1+exp(-x))
- silu: x * sigmoid(x)
- square: x * x (avoids pow VJP)
- rsqrt_gpu: Newton-Raphson 1/sqrt(x) — only mul/add/sub/div
- sqrt_gpu: x * rsqrt_gpu(x)
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


def rsqrt_gpu(x, n_iter=6):
    """GPU-safe 1/sqrt(x) via Newton-Raphson.

    Avoids nb.sqrt's VJP which creates mul(2.0, output) with CPU scalar.
    Newton iteration for y = rsqrt(x): y = y * (3 - x*y*y) / 2.
    Constants built from x/x=1 (same device), avoiding CPU scalars.
    """
    if not NABLA_AVAILABLE:
        raise ImportError("Nabla required for rsqrt_gpu")
    # Constants on same device as x (gradient = 0 w.r.t. x for x/x)
    one = nb.div(x, x)
    three = nb.add(nb.add(one, one), one)
    two = nb.add(one, one)
    half = nb.div(one, two)
    # Newton-Raphson: y = y * (3 - x*y^2) / 2
    y = half  # initial guess ≈ 0.5, reasonable for var in [0, 10]
    for _ in range(n_iter):
        ysq = nb.mul(y, y)
        y = nb.mul(y, nb.mul(nb.sub(three, nb.mul(x, ysq)), half))
    return y


def sqrt_gpu(x, n_iter=6):
    """GPU-safe sqrt(x) = x * rsqrt(x). Avoids nb.sqrt CPU scalar VJP."""
    if not NABLA_AVAILABLE:
        raise ImportError("Nabla required for sqrt_gpu")
    return nb.mul(x, rsqrt_gpu(x, n_iter=n_iter))
