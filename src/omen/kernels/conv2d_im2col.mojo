"""GPU im2col for conv2d: extract patches from NHWC input into flat matrix.

Input:
  x      (B, H, W, C_in)  — 4D NHWC tensor
  params (11,) float32    — [Hout, Wout, Kh, Kw, Cin, sh, sw, ph, pw, H, W]

Output:
  patches (B*Hout*Wout, Kh*Kw*Cin) — 2D flat patch matrix
"""

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList


@compiler.register("conv2d_im2col")
struct Conv2dIm2col:
    """Extract conv2d patches via im2col transformation."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = 4, static_spec = _],
        params: InputTensor[dtype = output.dtype, rank = 1, static_spec = _],
        ctx: DeviceContextPtr,
    ) raises:
        var Hout = Int(params.load[1](IndexList[1](0)))
        var Wout = Int(params.load[1](IndexList[1](1)))
        var Kh = Int(params.load[1](IndexList[1](2)))
        var Kw = Int(params.load[1](IndexList[1](3)))
        var Cin = Int(params.load[1](IndexList[1](4)))
        var sh = Int(params.load[1](IndexList[1](5)))
        var sw = Int(params.load[1](IndexList[1](6)))
        var ph = Int(params.load[1](IndexList[1](7)))
        var pw = Int(params.load[1](IndexList[1](8)))
        var H = Int(params.load[1](IndexList[1](9)))
        var W = Int(params.load[1](IndexList[1](10)))

        var Hout_Wout = Hout * Wout
        var Kw_Cin = Kw * Cin

        @parameter
        def extract_patch[width: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
            var spatial_idx = Int(idx[0])
            var kernel_idx_start = Int(idx[1])

            var b = spatial_idx // Hout_Wout
            var rem = spatial_idx - b * Hout_Wout
            var oh = rem // Wout
            var ow = rem - oh * Wout

            var result = SIMD[output.dtype, width](0.0)
            comptime for lane in range(width):
                var ki = kernel_idx_start + lane
                var kh_local = ki // Kw_Cin
                var rem2 = ki - kh_local * Kw_Cin
                var kw_local = rem2 // Cin
                var ci = rem2 - kw_local * Cin

                var ih = oh * sh + kh_local - ph
                var iw = ow * sw + kw_local - pw

                if ih >= 0 and ih < H and iw >= 0 and iw < W:
                    result[lane] = x.load[1](IndexList[4](b, ih, iw, ci))
            return result

        foreach[extract_patch, target=target](output, ctx)
