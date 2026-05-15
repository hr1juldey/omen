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

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList

comptime PACKED_CH = 10


@compiler.register("aov_pack")
struct AOVPack:
    """Pack Mitsuba multi-channel render into 10-ch aux buffer."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        source: InputTensor[dtype = output.dtype, rank = 3, static_spec = _],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        def pack_channel[W: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, W]:
            var h = Int(idx[0])
            var w = Int(idx[1])
            var ch_start = Int(idx[2])

            var result = SIMD[output.dtype, W](0.0)
            comptime for lane in range(W):
                var ch = ch_start + lane
                if ch < 3:
                    result[lane] = source.load[1](
                        IndexList[3](h, w, ch + 3)
                    )
                elif ch < 6:
                    result[lane] = source.load[1](
                        IndexList[3](h, w, ch + 3)
                    )
                elif ch == 6:
                    result[lane] = source.load[1](
                        IndexList[3](h, w, 9)
                    )

            return result

        foreach[pack_channel, target=target](output, ctx)
