"""Python bridge for Mojo GPU sigmoid/silu kernels.

Single-op activations bypassing MAX graph — reduces LLVM JIT compilation RAM.

Forward: Mojo GPU kernel (1 graph node vs 4-5 for pure-nabla)
Backward: pure-nabla VJP rules (compiled once, cached by @nb.compile)
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

KERNEL_DIR = Path(__file__).parent


class _SigmoidMojoOp(UnaryOperation):
    """Nabla operation wrapping the Mojo sigmoid kernel."""

    @property
    def name(self) -> str:
        return "sigmoid_kernel"

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        x = args[0]
        shape = tuple(int(d) for d in x.shape)
        return [shape], [x.dtype], [x.device]

    def kernel(self, args, kwargs):
        from max.graph import TensorType

        x = args[0]
        shape = tuple(int(d) for d in x.shape)
        out_type = TensorType(dtype=x.dtype, shape=shape, device=x.device)
        result = call_custom_kernel("sigmoid_kernel", str(KERNEL_DIR), x, out_type)
        return [result]

    def vjp_rule(self, primals, cotangents, outputs, kwargs):
        # sigmoid'(x) = σ(x)(1-σ(x)) = σ(x) - σ(x)²
        # Recompute from primal — never touch Mojo output tensor (causes SIGSEGV on GPU)
        ct = cotangents[0]
        sig = sigmoid_gpu(primals[0])
        return [ct * nb.sub(sig, square(sig))]


class _SiluMojoOp(UnaryOperation):
    """Nabla operation wrapping the Mojo silu kernel."""

    @property
    def name(self) -> str:
        return "silu_kernel"

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        x = args[0]
        shape = tuple(int(d) for d in x.shape)
        return [shape], [x.dtype], [x.device]

    def kernel(self, args, kwargs):
        from max.graph import TensorType

        x = args[0]
        shape = tuple(int(d) for d in x.shape)
        out_type = TensorType(dtype=x.dtype, shape=shape, device=x.device)
        result = call_custom_kernel("silu_kernel", str(KERNEL_DIR), x, out_type)
        return [result]

    def vjp_rule(self, primals, cotangents, outputs, kwargs):
        # silu'(x) = σ(x) + silu(x)*(1-σ(x)) = σ(x) + silu(x) - silu(x)*σ(x)
        # Recompute σ(x) for backward (~4 nabla nodes, compiled once)
        ct = cotangents[0]
        silu_out = outputs[0]
        sig = sigmoid_gpu(primals[0])
        return [ct * (sig + silu_out - nb.mul(silu_out, sig))]


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
