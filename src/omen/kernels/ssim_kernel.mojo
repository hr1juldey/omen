"""GPU-accelerated SSIM computation for Nabla.

Computes Structural Similarity Index map between two images
using sliding window statistics on GPU.

Input:  source (2, H+6, W+6) — img1 and img2 stacked, reflect-padded by 3
Output: ssim_map (H, W) — per-pixel SSIM score for original image

Window size: 7x7 with reflect padding (handled by Python bridge).
SSIM constants: C1 = 0.01^2, C2 = 0.03^2 (standard values).

The two input images are packed into a single (2, H+6, W+6) tensor to
avoid the MAX framework multi-input custom kernel data transfer bug.
"""

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList

comptime WIN = 7
comptime PAD = WIN // 2  # 3 — symmetric window centering
comptime FP_C1 = 0.0001  # (0.01)^2
comptime FP_C2 = 0.0009  # (0.03)^2


@compiler.register("ssim_compute")
struct SSIMCompute:
    """Compute per-pixel SSIM map using 7x7 uniform window.

    Source tensor layout: (2, H+6, W+6) where source[0] = img1, source[1] = img2.
    Output tensor layout: (H, W) — original (unpadded) image size.
    """

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        source: InputTensor[dtype = output.dtype, rank = 3, static_spec = _],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        def compute_ssim[W: Int](idx: IndexList[output.rank]) -> SIMD[output.dtype, W]:
            var h = Int(idx[0])
            var w_start = Int(idx[1])

            var result = SIMD[output.dtype, W](0.0)
            comptime for lane in range(W):
                var w = w_start + lane

                # Per-lane scalar accumulators for 7x7 window
                var s1 = SIMD[output.dtype, 1](0.0)
                var s2 = SIMD[output.dtype, 1](0.0)
                var s11 = SIMD[output.dtype, 1](0.0)
                var s22 = SIMD[output.dtype, 1](0.0)
                var s12 = SIMD[output.dtype, 1](0.0)
                var n = SIMD[output.dtype, 1](0.0)

                comptime for dy in range(WIN):
                    comptime for dx in range(WIN):
                        var y = h + Int(dy) - PAD
                        var x = w + Int(dx) - PAD
                        # Lower-bound check only — upper bounds are
                        # guaranteed by PAD=3 padding on source.
                        if y >= 0 and x >= 0:
                            var v1 = source.load[1](
                                IndexList[3](0, y, x)
                            )
                            var v2 = source.load[1](
                                IndexList[3](1, y, x)
                            )
                            s1 = s1 + v1
                            s2 = s2 + v2
                            s11 = s11 + v1 * v1
                            s22 = s22 + v2 * v2
                            s12 = s12 + v1 * v2
                            n = n + SIMD[output.dtype, 1](1.0)

                # SSIM formula
                var safe_n = n + SIMD[output.dtype, 1](1e-8)
                var mu1 = s1 / safe_n
                var mu2 = s2 / safe_n
                var sig1 = s11 / safe_n - mu1 * mu1
                var sig2 = s22 / safe_n - mu2 * mu2
                var sig12 = s12 / safe_n - mu1 * mu2

                var c1 = SIMD[output.dtype, 1](FP_C1)
                var c2 = SIMD[output.dtype, 1](FP_C2)
                var num = (2.0 * mu1 * mu1 + c1) * (2.0 * sig12 + c2)
                var den = (mu1 * mu1 + mu2 * mu2 + c1) * (sig1 + sig2 + c2)

                result[lane] = num / den

            return result

        foreach[compute_ssim, target=target](output, ctx)
