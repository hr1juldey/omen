"""GPU-accelerated MLA skip compression for Nabla.

Fuses Linear projection + SiLU activation into single kernel for
U-Net skip connection compression (DeepSeek-V2/V3 MLA pattern).

Input:  combined (3 + N*dim0 + dim0*dim1,) flat tensor:
          [0:3]   = [N, dim0, dim1] metadata
          [3 : 3+N*dim0] = data flattened (features or compressed)
          [3+N*dim0 : end] = weights flattened
Output: (N, dim1) — matmul result (with optional activation)

dim0 = inner loop dimension (C_in for compress, C_latent for reconstruct)
dim1 = output dimension (C_latent for compress, C_in for reconstruct)

Packed into single tensor to avoid MAX framework multi-input data transfer bug.
"""

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList

comptime TILE = 16
comptime MAX_DIM0_TILES = 32  # 32 * 16 = 512 max dim0


@compiler.register("mla_compress")
struct MLACompress:
    """Fused linear projection + hard-swish for skip compression."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        combined: InputTensor[dtype = output.dtype, rank = 1, static_spec = _],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        def compress[W: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, W]:
            var N = Int(combined.load[1](IndexList[1](0)))
            var dim0 = Int(combined.load[1](IndexList[1](1)))
            var dim1 = Int(combined.load[1](IndexList[1](2)))
            var data_off = 3
            var wt_off = 3 + N * dim0

            var n = Int(idx[0])
            var c_out_start = Int(idx[1])

            var result = SIMD[output.dtype, W](0.0)
            comptime for lane in range(W):
                var c_out = c_out_start + lane
                var acc = SIMD[output.dtype, 1](0.0)

                comptime for tile in range(MAX_DIM0_TILES):
                    var c_base = tile * TILE
                    comptime for d in range(TILE):
                        var c = c_base + d
                        if c < dim0:
                            var feat = combined.load[1](
                                IndexList[1](data_off + n * dim0 + c)
                            )
                            var w = combined.load[1](
                                IndexList[1](wt_off + c * dim1 + c_out)
                            )
                            acc = acc + feat * w

                # Hard-swish: x * clamp(x/6 + 0.5, 0, 1)
                var six = SIMD[output.dtype, 1](6.0)
                var half = SIMD[output.dtype, 1](0.5)
                var zero_val = SIMD[output.dtype, 1](0.0)
                var one_val = SIMD[output.dtype, 1](1.0)
                var gate = (acc / six + half).clamp(zero_val, one_val)

                result[lane] = acc * gate

            return result

        foreach[compress, target=target](output, ctx)


@compiler.register("mla_reconstruct")
struct MLAReconstruct:
    """Reconstruct skip features from compressed latent."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        combined: InputTensor[dtype = output.dtype, rank = 1, static_spec = _],
        ctx: DeviceContextPtr,
    ) raises:
        # dim0 is small (C_latent, max ~32), so 2 tiles of 16
        comptime MAX_TILES = 2

        @parameter
        def reconstruct[W: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, W]:
            var N = Int(combined.load[1](IndexList[1](0)))
            var dim0 = Int(combined.load[1](IndexList[1](1)))
            var dim1 = Int(combined.load[1](IndexList[1](2)))
            var data_off = 3
            var wt_off = 3 + N * dim0

            var n = Int(idx[0])
            var c_out_start = Int(idx[1])

            var result = SIMD[output.dtype, W](0.0)
            comptime for lane in range(W):
                var c_out = c_out_start + lane
                var acc = SIMD[output.dtype, 1](0.0)

                comptime for tile in range(MAX_TILES):
                    var c_base = tile * TILE
                    comptime for d in range(TILE):
                        var c = c_base + d
                        if c < dim0:
                            var z = combined.load[1](
                                IndexList[1](data_off + n * dim0 + c)
                            )
                            var w = combined.load[1](
                                IndexList[1](wt_off + c * dim1 + c_out)
                            )
                            acc = acc + z * w

                result[lane] = acc

            return result

        foreach[reconstruct, target=target](output, ctx)
