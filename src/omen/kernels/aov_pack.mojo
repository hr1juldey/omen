"""GPU-accelerated AOV channel packing for Nabla.

Interleaves Mitsuba AOV channels into contiguous (H, W, 10) buffer:
  albedo(3) + normal(3) + depth(1) + material_id(1) + motion_vectors(2)

Input: source (H, W, C) Mitsuba AOV render result (C >= 10)
Output: packed (H, W, 10) contiguous aux buffer

Mitsuba AOV layout (typical with _AOV_SPEC):
  ch 0-2: RGB color
  ch 3-5: albedo
  ch 6-8: normal (sh_normal)
  ch 9:   depth (dd.y)

Output layout (matches aov.py AOV_PASSES):
  ch 0-2: albedo
  ch 3-5: normal
  ch 6:   depth
  ch 7:   material_id (zero if unavailable)
  ch 8-9: motion vectors (zero if unavailable)
"""

comptime PACKED_CH = 10


@compiler.register("aov_pack")
struct AOVPack:
    """Pack Mitsuba multi-channel render into 10-ch aux buffer."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        source: InputTensor[dtype = output.dtype, rank = 3],
        ctx: DeviceContextPtr,
    ):
        @parameter
        def pack_channel[W: Int](idx: IndexList[3]) -> SIMD[output.dtype, W]:
            var h = idx[0]
            var w = idx[1]
            var ch = idx[2]
            var zero = SIMD[output.dtype, W](0.0)

            # Output ch 0-2 -> albedo from source ch 3-5
            if ch < 3:
                return source.load[1]([h, w, ch + 3])

            # Output ch 3-5 -> normal from source ch 6-8
            elif ch < 6:
                return source.load[1]([h, w, ch + 3])

            # Output ch 6 -> depth from source ch 9
            elif ch == 6:
                return source.load[1]([h, w, 9])

            # Output ch 7, 8, 9 -> not in standard Mitsuba AOV
            return zero

        foreach[pack_channel, target=target](output, ctx)
