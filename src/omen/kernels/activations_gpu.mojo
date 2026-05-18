"""GPU sigmoid and silu kernels — single-op activations bypassing MAX graph.

Element-wise: each activation = 1 Mojo kernel call instead of 4-5 nabla ops.

Uses rational Padé approximation of tanh to avoid exp() (which needs
floating-point trait evidence unavailable inside foreach callbacks).

sigmoid(x) = 0.5 + 0.5 * tanh(x/2)
silu(x)    = x * sigmoid(x)

tanh(y) approximated by Padé [2/2]: y(15 + y^2) / (15 + 6y^2)
Accurate to <1% for |x| < 5, exact at x=0.
"""

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList


def _sigmoid[dtype: DType, W: Int](val: SIMD[dtype, W]) -> SIMD[dtype, W]:
    """Sigmoid via tanh Pade [2/2] approximation. No exp needed."""
    var half = val * 0.5
    var sq = half * half
    var tanh_a = half * (15.0 + sq) / (15.0 + 6.0 * sq)
    return 0.5 + 0.5 * tanh_a


@compiler.register("sigmoid_kernel")
struct SigmoidKernel:
    """Element-wise sigmoid via Padé approximation."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank, static_spec = _],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        def compute[W: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, W]:
            return _sigmoid(x.load[W](idx))

        foreach[compute, target=target](output, ctx)


@compiler.register("silu_kernel")
struct SiluKernel:
    """Element-wise SiLU: x * sigmoid(x) via Padé approximation."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank, static_spec = _],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        def compute[W: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, W]:
            var val = x.load[W](idx)
            return val * _sigmoid(val)

        foreach[compute, target=target](output, ctx)
