"""GPU-accelerated MoE expert dispatch for Nabla.

Fuses top-k expert selection + weighted combination into single kernel.
Packs expert_outputs + routing_weights into single flat tensor to avoid
the MAX framework multi-input custom kernel data transfer bug.

Input:  combined flat tensor layout:
          [0:3] = [T, C, E] metadata
          [3 : 3+T*C*E] = expert_outputs flattened (T, C, E)
          [3+T*C*E : end] = routing_weights flattened (T, E)
Output: (T, C) — weighted sum of expert outputs
"""

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList

comptime MAX_EXPERTS = 8


@compiler.register("moe_dispatch")
struct MoEDispatch:
    """Weighted combination of expert outputs via routing weights.

    For each (token, channel), accumulates weighted expert outputs.
    """

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        combined: InputTensor[dtype = output.dtype, rank = 1, static_spec = _],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        def combine[W: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, W]:
            var T = Int(combined.load[1](IndexList[1](0)))
            var C = Int(combined.load[1](IndexList[1](1)))
            var E = Int(combined.load[1](IndexList[1](2)))
            var eo_off = 3
            var rw_off = 3 + T * C * E

            var token = Int(idx[0])
            var ch_start = Int(idx[1])

            var result = SIMD[output.dtype, W](0.0)
            comptime for lane in range(W):
                var ch = ch_start + lane
                var acc = SIMD[output.dtype, 1](0.0)

                comptime for e in range(MAX_EXPERTS):
                    if e < E:
                        var w = combined.load[1](
                            IndexList[1](rw_off + token * E + e)
                        )
                        var e_out = combined.load[1](
                            IndexList[1](
                                eo_off + token * C * E + ch * E + e
                            )
                        )
                        acc = acc + w * e_out

                result[lane] = acc

            return result

        foreach[combine, target=target](output, ctx)
