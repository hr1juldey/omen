"""GPU-safe activation functions with scalar-free backward passes.

Nabla's built-in silu/sigmoid VJP rules create Python scalar constants
(`sub(1.0, sig_x)`) that become CPU tensors during GPU backward — causing
device mismatch. These custom ops avoid that by using tensor-only identities.

Key identity: σ(-x) = 1 - σ(x) — computed without any Python scalar.
"""

import logging

logger = logging.getLogger("omen.kernels.activations")

try:
    import nabla as nb
    from nabla.ops import UnaryOperation

    NABLA_AVAILABLE = True
except ImportError:
    UnaryOperation = object
    NABLA_AVAILABLE = False


class SiluGPU(UnaryOperation):
    """GPU-safe SiLU: x * σ(x) with scalar-free derivative.

    Forward: x * sigmoid(x)
    Backward: cotangent * (σ(x) + x * σ(x) * σ(-x))

    The σ(-x) = 1 - σ(x) identity avoids the `sub(1.0, sig_x)` that
    nabla's built-in silu VJP uses — no Python scalar constants.
    """

    @property
    def name(self) -> str:
        return "silu_gpu"

    def kernel(self, args, kwargs):
        x = args[0]
        sig = nb.sigmoid(x)
        return [x * sig]

    def _derivative(self, primal, output):
        sig = nb.sigmoid(primal)
        sig_neg = nb.sigmoid(-primal)  # = 1 - σ(x), no scalar
        return sig + primal * sig * sig_neg


def silu_gpu(x):
    """GPU-safe SiLU activation (scalar-free backward).

    Drop-in replacement for nb.silu that works on GPU backward passes.
    Falls back to nb.silu when nabla is unavailable.
    """
    if not NABLA_AVAILABLE:
        raise ImportError("Nabla required for silu_gpu")
    op = SiluGPU()
    return op([x], {})[0]
