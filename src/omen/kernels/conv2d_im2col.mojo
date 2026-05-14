"""GPU im2col for conv2d: extract patches from NHWC input into flat matrix.

Used by Conv2dOp for the forward pass of conv2d via im2col + matmul.

Input:
  x      (B, H, W, C_in)  — 4D NHWC tensor
  params (11,) float32    — [Hout, Wout, Kh, Kw, Cin, sh, sw, ph, pw, H, W]

Output:
  patches (B*Hout*Wout, Kh*Kw*Cin) — 2D flat patch matrix

Gather pattern: each output element reads one value from x.
Zero-padding for out-of-bounds positions.
"""


@compiler.register("conv2d_im2col")
struct Conv2dIm2col:
    """Extract conv2d patches via im2col transformation."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = 4],
        params: InputTensor[dtype = output.dtype, rank = 1],
        ctx: DeviceContextPtr,
    ):
        # Read runtime parameters from params tensor
        # Layout: [Hout, Wout, Kh, Kw, Cin, sh, sw, ph, pw, H, W]
        var Hout = Int(params.load[1]([0]))
        var Wout = Int(params.load[1]([1]))
        var Kh = Int(params.load[1]([2]))
        # Kh not used in im2col decomposition but included for param layout
        # consistency with col2im
        var Kw = Int(params.load[1]([3]))
        var Cin = Int(params.load[1]([4]))
        var sh = Int(params.load[1]([5]))
        var sw = Int(params.load[1]([6]))
        var ph = Int(params.load[1]([7]))
        var pw = Int(params.load[1]([8]))
        var H = Int(params.load[1]([9]))
        var W = Int(params.load[1]([10]))

        var Hout_Wout = Hout * Wout
        var Kw_Cin = Kw * Cin

        @parameter
        def extract_patch[width: Int](idx: IndexList[2]) -> SIMD[output.dtype, width]:
            var spatial_idx = Int(idx[0])
            var kernel_idx = Int(idx[1])

            # Decompose spatial index -> (b, oh, ow)
            var b = spatial_idx // Hout_Wout
            var rem = spatial_idx - b * Hout_Wout
            var oh = rem // Wout
            var ow = rem - oh * Wout

            # Decompose kernel index -> (kh, kw, ci)
            var kh = kernel_idx // Kw_Cin
            var rem2 = kernel_idx - kh * Kw_Cin
            var kw = rem2 // Cin
            var ci = rem2 - kw * Cin

            # Compute input position with stride and padding
            var ih = oh * sh + kh - ph
            var iw = ow * sw + kw - pw

            # Boundary check - zero-padding for out-of-bounds
            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                return x.load[1]([b, ih, iw, ci])
            else:
                return SIMD[output.dtype, width](0.0)

        foreach[extract_patch, target=target](output, ctx)
