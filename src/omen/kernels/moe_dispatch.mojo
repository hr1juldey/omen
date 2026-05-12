"""GPU-accelerated MoE expert dispatch for Nabla.

Fuses top-k expert selection + weighted combination into single kernel.
Replaces Python for-loops over experts with parallel GPU scatter-add.

Inputs:
  expert_outputs (tokens, channels, num_experts) - all expert outputs stacked
  routing_weights (tokens, num_experts) - sparse weight matrix (top-k nonzero)
Output:
  combined (tokens, channels) - weighted sum of expert outputs

Replaces the double Python loop in omen.model.moe ExpertGroup.forward().
"""

comptime MAX_EXPERTS = 8


@compiler.register("moe_dispatch")
struct MoEDispatch:
    """Weighted combination of expert outputs via routing weights.

    For each (token, channel), accumulates weighted expert outputs.
    routing_weights is sparse: nonzero only for top-k selected experts.
    """

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        expert_outputs: InputTensor[dtype = output.dtype, rank = 3],
        routing_weights: InputTensor[dtype = output.dtype, rank = 2],
        ctx: DeviceContextPtr,
    ):
        @parameter
        def combine[W: Int](idx: IndexList[2]) -> SIMD[output.dtype, W]:
            var token = idx[0]
            var ch = idx[1]
            var acc = SIMD[output.dtype, W](0.0)

            # Iterate all experts — unselected ones have weight=0, so
            # multiply-accumulate is safe and avoids dynamic indexing
            comptime for e in range(MAX_EXPERTS):
                var w = routing_weights.load[1]([token, e])
                var e_out = expert_outputs.load[1]([token, ch, e])
                acc = acc + w * e_out

            return acc

        foreach[combine, target=target](output, ctx)
