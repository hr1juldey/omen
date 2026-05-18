"""Python bridge for Mojo GPU sigmoid/silu kernels.

Single-op activations bypassing MAX graph — reduces LLVM JIT compilation RAM.

Forward: Mojo GPU kernel (1 graph node vs 4-5 for pure-nabla)
Backward: _derivative recomputes from primal input via pure-nabla ops

Pattern: official Nabla custom kernels tutorial
https://www.nablaml.com/examples/12_custom_mojo_kernels.html
"""

import logging
from pathlib import Path

try:
    from nabla.ops import UnaryOperation, call_custom_kernel

    import nabla as nb

    NABLA_AVAILABLE = True
except ImportError:
    UnaryOperation = object
    NABLA_AVAILABLE = False

from omen.kernels.activations import sigmoid_gpu, silu_gpu, square

logger = logging.getLogger("omen.kernels.activations_gpu")

KERNEL_DIR = str(Path(__file__).parent)


class _SigmoidMojoOp(UnaryOperation):
    """Mojo sigmoid kernel — forward via Mojo, backward via pure-nabla."""

    @property
    def name(self) -> str:
        return "sigmoid_kernel"

    def kernel(self, args, kwargs):
        x = args[0]
        result = call_custom_kernel(
            "sigmoid_kernel", KERNEL_DIR, x, x.type, device=x.device
        )
        return [result]

    def _derivative(self, primals, output):
        # dσ/dx = σ(x) - σ(x)²
        # Recompute from primal (pure-nabla) — never touch Mojo output on GPU
        sig = sigmoid_gpu(primals[0])
        return sig - square(sig)


class _SiluMojoOp(UnaryOperation):
    """Mojo silu kernel — forward via Mojo, backward via pure-nabla."""

    @property
    def name(self) -> str:
        return "silu_kernel"

    def kernel(self, args, kwargs):
        x = args[0]
        result = call_custom_kernel(
            "silu_kernel", KERNEL_DIR, x, x.type, device=x.device
        )
        return [result]

    def _derivative(self, primals, output):
        # d(silu)/dx = σ(x) + x*σ(x)*(1-σ(x))
        # = σ(x) + x*σ(x) - x*σ(x)²
        # = σ(x) + x*(σ(x) - σ(x)²)
        # Recompute sigmoid from primal (pure-nabla)
        x = primals[0]
        sig = sigmoid_gpu(x)
        return sig + x * (sig - square(sig))


# Module-level op instances
_sigmoid_op = _SigmoidMojoOp()
_silu_op = _SiluMojoOp()


def sigmoid_mojo(x):
    """GPU sigmoid via Mojo kernel. Falls back to pure-nabla on failure."""
    if not NABLA_AVAILABLE:
        return sigmoid_gpu(x)
    try:
        return _sigmoid_op([x], {})[0]
    except Exception as exc:
        logger.warning("Mojo sigmoid kernel failed (%s) — nabla fallback", exc)
        return sigmoid_gpu(x)


def silu_mojo(x):
    """GPU silu via Mojo kernel. Falls back to pure-nabla on failure."""
    if not NABLA_AVAILABLE:
        return silu_gpu(x)
    try:
        return _silu_op([x], {})[0]
    except Exception as exc:
        logger.warning("Mojo silu kernel failed (%s) — nabla fallback", exc)
        return silu_gpu(x)
