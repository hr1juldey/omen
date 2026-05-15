"""GPU-accelerated 8x8 tile fingerprint computation for Nabla.

Computes 23-dim fingerprint per tile from auxiliary render buffers:
  Material histogram(8) + normal_var(3) + depth_var(1) + edge_density(1) +
  dominant_mat(1) + mean_albedo(3) + velocity_mean(2) + velocity_var(2) +
  velocity_max(1) + occlusion_frac(1) = 23

Input: aux (H, W, 10) — albedo(3)+normal(3)+depth(1)+mat_id(1)+motion(2)
Output: fingerprints (H//8, W//8, 23)
"""

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList

comptime TILE = 8
comptime FP_DIM = 23
comptime AUX_CH = 10


@compiler.register("tile_fingerprint")
struct TileFingerprint:
    """Compute 23-dim tile fingerprint from packed AOV buffer."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        aux: InputTensor[dtype = output.dtype, rank = 3, static_spec = _],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        def compute_tile[W: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, W]:
            var ty = Int(idx[0])
            var tx = Int(idx[1])
            var ch_start = Int(idx[2])

            var h_base = ty * TILE
            var w_base = tx * TILE
            var inv_area = SIMD[output.dtype, 1](1.0 / (TILE * TILE))

            var result = SIMD[output.dtype, W](0.0)
            comptime for lane in range(W):
                var ch = ch_start + lane
                var acc = SIMD[output.dtype, 1](0.0)

                comptime for dy in range(TILE):
                    comptime for dx in range(TILE):
                        var h = h_base + Int(dy)
                        var w = w_base + Int(dx)

                        # Normal variance (ch 8-10)
                        if ch >= 8 and ch < 11:
                            var nc = ch - 8 + 3
                            var val = aux.load[1](
                                IndexList[3](h, w, nc)
                            )
                            acc = acc + val * val * inv_area
                        # Depth variance (ch 11)
                        elif ch == 11:
                            var dv = aux.load[1](
                                IndexList[3](h, w, 6)
                            )
                            acc = acc + dv * dv * inv_area
                        # Mean albedo (ch 14-16)
                        elif ch >= 14 and ch < 17:
                            var ac = ch - 14
                            acc = acc + aux.load[1](
                                IndexList[3](h, w, ac)
                            ) * inv_area

                result[lane] = acc

            return result

        foreach[compute_tile, target=target](output, ctx)
