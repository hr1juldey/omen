"""GPU col2im for conv2d backward: scatter column matrix back to NHWC tensor.

Input:
  col    (B*H_out*W_out, Kh*Kw*C_in) — column matrix from matmul
  params (11,) float32               — [H_out, W_out, Kh, Kw, Cin, stride_h,
                                        stride_w, pad_h, pad_w, H, W]

Output:
  grad_x (B, H, W, C_in) — accumulated gradient for input
"""

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList


@compiler.register("conv2im_col2im")
struct Conv2dCol2im:
    """Scatter column matrix back to image via col2im."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        col: InputTensor[dtype = output.dtype, rank = 2, static_spec = _],
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

        var Kw_Cin = Kw * Cin
        var Hout_Wout = Hout * Wout

        @parameter
        def scatter_gather[width: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
            var b = Int(idx[0])
            var ih = Int(idx[1])
            var iw = Int(idx[2])
            var ci_start = Int(idx[3])

            var result = SIMD[output.dtype, width](0.0)
            comptime for lane in range(width):
                var ci = ci_start + lane

                var acc = SIMD[output.dtype, 1](0.0)
                for oh in range(Hout):
                    var kh = ih + ph - oh * sh
                    if kh < 0 or kh >= Kh:
                        continue

                    for ow in range(Wout):
                        var kw = iw + pw - ow * sw
                        if kw < 0 or kw >= Kw:
                            continue

                        var spatial_idx = b * Hout_Wout + oh * Wout + ow
                        var kernel_idx = kh * Kw_Cin + kw * Cin + ci

                        acc = acc + col.load[1](IndexList[2](spatial_idx, kernel_idx))

                result[lane] = acc
            return result

        foreach[scatter_gather, target=target](output, ctx)
