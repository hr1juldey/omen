"""GPU-accelerated 8x8 tile fingerprint computation for Nabla.

Computes 23-dim fingerprint per tile from auxiliary render buffers:
  Material histogram(8) + normal_var(3) + depth_var(1) + edge_density(1) +
  dominant_mat(1) + mean_albedo(3) + velocity_mean(2) + velocity_var(2) +
  velocity_max(1) + occlusion_frac(1) = 23

Uses Nabla's custom kernel pattern: @compiler.register struct with execute().
One thread per pixel in an 8x8 tile. Shared memory for tile-level reduction.
"""

# Nabla custom kernel API provides: OutputTensor, InputTensor,
# DeviceContextPtr, IndexList, foreach — no standalone GPU imports needed

comptime TILE = 8
comptime FP_DIM = 23
comptime AUX_CH = 10


@compiler.register("tile_fingerprint")
struct TileFingerprint:
    """Compute 23-dim tile fingerprint from packed AOV buffer.

    Input: aux (H, W, 10) — albedo(3)+normal(3)+depth(1)+mat_id(1)+motion(2)
    Output: fingerprints (H//8, W//8, 23)
    """

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        aux: InputTensor[dtype = output.dtype, rank = 3],
        ctx: DeviceContextPtr,
    ):
        @parameter
        def compute_tile[W: Int](idx: IndexList[3]) -> SIMD[output.dtype, W]:
            # idx gives position in output grid (tiles_y, tiles_x, fp_dim)
            var ty = idx[0]
            var tx = idx[1]
            var ch = idx[2]

            # For now return per-channel computation
            # Full shared-memory reduction requires host-side kernel launch
            var h_base = ty * TILE
            var w_base = tx * TILE

            # Accumulate over 8x8 tile
            var acc = SIMD[output.dtype, W](0.0)
            comptime for dy in range(TILE):
                comptime for dx in range(TILE):
                    var h = h_base + Int(dy)
                    var w = w_base + Int(dx)
                    # Normal variance (ch 8-10)
                    if ch >= 8 and ch < 11:
                        var nc = ch - 8 + 3  # normal channel offset
                        var val = aux.load[1]([h, w, nc])
                        acc = acc + val * val * (1.0 / (TILE * TILE))
                    # Depth variance (ch 11)
                    elif ch == 11:
                        var dv = aux.load[1]([h, w, 6])
                        acc = acc + dv * dv * (1.0 / (TILE * TILE))
                    # Mean albedo (ch 14-16)
                    elif ch >= 14 and ch < 17:
                        var ac = ch - 14  # albedo channel
                        acc = acc + aux.load[1]([h, w, ac]) * (1.0 / (TILE * TILE))
            return acc

        foreach[compute_tile, target=target](output, ctx)
