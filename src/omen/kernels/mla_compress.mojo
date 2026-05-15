"""GPU-accelerated MLA skip compression for Nabla.

Fuses Linear projection + SiLU activation into single kernel for
U-Net skip connection compression (DeepSeek-V2/V3 MLA pattern).

Reduces skip memory 16x: (N, C) -> (N, C//16) via learned projection.

Inputs:
  features (N, C_in) - encoder skip features
  weights_down (C_in, C_latent) - projection matrix
Output:
  compressed (N, C_latent) - silu(features @ weights_down)

C_in varies by tier: fast=192, medium=256, high=512
C_latent = C_in // 16 (min 4)
"""

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList
comptime TILE = 16


@compiler.register("mla_compress")
struct MLACompress:
    """Fused linear projection + SiLU for skip compression."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        features: InputTensor[dtype = output.dtype, rank = 2, static_spec = _],
        weights_down: InputTensor[dtype = output.dtype, rank = 2, static_spec = _],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        def compress[W: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, W]:
            var n = Int(idx[0])
            var c_out = Int(idx[1])
            var acc = SIMD[output.dtype, W](0.0)

            # Accumulate: output[n, c_out] = sum_c(features[n, c] * W[c, c_out])
            # Process in tiles of 16 for better memory access patterns
            comptime for tile in range(32):  # 32 * 16 = 512 max channels
                var c_base = tile * TILE
                comptime for d in range(TILE):
                    var c = c_base + d
                    var feat = features.load[1](IndexList[2](n, c))
                    var w = weights_down.load[1](IndexList[2](c, c_out))
                    acc = acc + feat * w

            # Fused hard-swish activation: x * clamp(x/6 + 0.5, 0, 1)
            # Arithmetic-only (no exp/sigmoid — unavailable in nabla foreach)
            var six = SIMD[output.dtype, W](6.0)
            var half = SIMD[output.dtype, W](0.5)
            var zero_val = SIMD[output.dtype, W](0.0)
            var one_val = SIMD[output.dtype, W](1.0)
            var gate = (acc / six + half).clamp(zero_val, one_val)
            return acc * gate

        foreach[compress, target=target](output, ctx)


@compiler.register("mla_reconstruct")
struct MLAReconstruct:
    """Reconstruct skip features from compressed latent.

    Input: compressed (N, C_latent)
    Weights: weights_up (C_latent, C_in)
    Output: reconstructed (N, C_in)
    """

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        compressed: InputTensor[dtype = output.dtype, rank = 2, static_spec = _],
        weights_up: InputTensor[dtype = output.dtype, rank = 2, static_spec = _],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        def reconstruct[W: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, W]:
            var n = Int(idx[0])
            var c_out = Int(idx[1])
            var acc = SIMD[output.dtype, W](0.0)

            # C_latent is small (C_in // 16), so full unroll is fine
            comptime for tile in range(2):  # 2 * 16 = 32 max latent dim
                var c_base = tile * TILE
                comptime for d in range(TILE):
                    var c = c_base + d
                    var z = compressed.load[1](IndexList[2](n, c))
                    var w = weights_up.load[1](IndexList[2](c, c_out))
                    acc = acc + z * w

            return acc

        foreach[reconstruct, target=target](output, ctx)
