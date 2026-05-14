"""GPU col2im for conv2d backward: scatter column matrix back to NHWC tensor.

Used by Conv2dOp.vjp_rule for computing grad_input (transposed convolution
scatter-back).

Input:
  col    (B*H_out*W_out, Kh*Kw*C_in) — column matrix from matmul
  params (11,) float32               — [H_out, W_out, Kh, Kw, Cin, stride_h,
                                        stride_w, pad_h, pad_w, H, W]

Output:
  grad_x (B, H, W, C_in) — accumulated gradient for input

Gather pattern: iterate over OUTPUT positions (b, ih, iw, ci).
Each thread accumulates contributions from all (oh, ow, kh, kw) column entries
where oh*stride + kh - pad == ih. Sequential inner loop, no atomics.
"""

@compiler.register("conv2im_col2im")
struct Conv2dCol2im:
    """Scatter column matrix back to image via col2im (gather pattern)."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        col: InputTensor[dtype = output.dtype, rank = 2],
        params: InputTensor[dtype = output.dtype, rank = 1],
        ctx: DeviceContextPtr,
    ):
        # Read runtime parameters
        var Hout = Int(params.load[1]([0]))
        var Wout = Int(params.load[1]([1]))
        var Kh = Int(params.load[1]([2]))
        var Kw = Int(params.load[1]([3]))
        var Cin = Int(params.load[1]([4]))
        var sh = Int(params.load[1]([5]))
        var sw = Int(params.load[1]([6]))
        var ph = Int(params.load[1]([7]))
        var pw = Int(params.load[1]([8]))
        var H = Int(params.load[1]([9]))
        var W = Int(params.load[1]([10]))

        var Kw_Cin = Kw * Cin
        var Hout_Wout = Hout * Wout

        @parameter
        def scatter_gather[width: Int](idx: IndexList[4]) -> SIMD[output.dtype, width]:
            var b = Int(idx[0])
            var ih = Int(idx[1])
            var iw = Int(idx[2])
            var ci = Int(idx[3])

            var acc = SIMD[output.dtype, width](0.0)

            # Iterate over all kernel positions and output positions that
            # contribute to this (b, ih, iw, ci)
            # Condition: oh*sh + kh - ph == ih AND ow*sw + kw - pw == iw
            # So: oh == (ih + ph - kh) / sh, must be integer and in [0, Hout)
            # And: ow == (iw + pw - kw) / sw, must be integer and in [0, Wout)
            for oh in range(Hout):
                # Check row: ih + ph must equal oh*sh + kh for some kh in [0, Kh)
                # kh = ih + ph - oh*sh, must be in [0, Kh)
                var kh = ih + ph - oh * sh
                if kh < 0 or kh >= Kh:
                    continue

                for ow in range(Wout):
                    var kw = iw + pw - ow * sw
                    if kw < 0 or kw >= Kw:
                        continue

                    # Compute flat column index
                    var spatial_idx = b * Hout_Wout + oh * Wout + ow
                    var kernel_idx = kh * Kw_Cin + kw * Cin + ci

                    acc = acc + col.load[1]([spatial_idx, kernel_idx])

            return acc

        foreach[scatter_gather, target=target](output, ctx)
