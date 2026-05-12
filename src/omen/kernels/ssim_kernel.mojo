"""GPU-accelerated SSIM computation for Nabla.

Computes Structural Similarity Index map between two images
using sliding window statistics on GPU.

Input: img1 (H, W), img2 (H, W) — single-channel float32
Output: ssim_map (H, W) — per-pixel SSIM score

Window size: 7x7 with reflect padding (handled by Python bridge).
SSIM constants: C1 = 0.01^2, C2 = 0.03^2 (standard values).

Based on Wang et al. (2004) SSIM with uniform window weighting.
"""

comptime WIN = 7
comptime PAD = WIN // 3  # 2 — reflect pad applied by Python bridge
comptime FP_C1 = 0.0001  # (0.01)^2
comptime FP_C2 = 0.0009  # (0.03)^2


@compiler.register("ssim_compute")
struct SSIMCompute:
    """Compute per-pixel SSIM map using 7x7 uniform window."""

    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        img1: InputTensor[dtype = output.dtype, rank = 2],
        img2: InputTensor[dtype = output.dtype, rank = 2],
        ctx: DeviceContextPtr,
    ):
        @parameter
        def compute_ssim[W: Int](idx: IndexList[2]) -> SIMD[output.dtype, W]:
            var h = idx[0]
            var w = idx[1]
            var c1 = SIMD[output.dtype, W](FP_C1)
            var c2 = SIMD[output.dtype, W](FP_C2)

            # Accumulate window statistics
            var sum1 = SIMD[output.dtype, W](0.0)
            var sum2 = SIMD[output.dtype, W](0.0)
            var sum11 = SIMD[output.dtype, W](0.0)
            var sum22 = SIMD[output.dtype, W](0.0)
            var sum12 = SIMD[output.dtype, W](0.0)
            var n = SIMD[output.dtype, W](0.0)

            comptime for dy in range(WIN):
                comptime for dx in range(WIN):
                    var y = h + Int(dy) - PAD
                    var x = w + Int(dx) - PAD
                    # Boundary: skip out-of-range pixels
                    if y >= 0 and x >= 0:
                        var v1 = img1.load[1]([y, x])
                        var v2 = img2.load[1]([y, x])
                        sum1 = sum1 + v1
                        sum2 = sum2 + v2
                        sum11 = sum11 + v1 * v1
                        sum22 = sum22 + v2 * v2
                        sum12 = sum12 + v1 * v2
                        n = n + SIMD[output.dtype, W](1.0)

            # Avoid division by zero at image corners
            var safe_n = n + SIMD[output.dtype, W](1e-8)
            var mu1 = sum1 / safe_n
            var mu2 = sum2 / safe_n
            var sig1 = sum11 / safe_n - mu1 * mu1
            var sig2 = sum22 / safe_n - mu2 * mu2
            var sig12 = sum12 / safe_n - mu1 * mu2

            var num = (2.0 * mu1 * mu2 + c1) * (2.0 * sig12 + c2)
            var den = (mu1 * mu1 + mu2 * mu2 + c1) * (sig1 + sig2 + c2)

            return num / den

        foreach[compute_ssim, target=target](output, ctx)
