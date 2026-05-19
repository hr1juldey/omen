"""Pixel Shuffle upsampling kernel — sub-pixel convolution for decoder.

Rearranges (H, W, C*R*R) -> (H*R, W*R, C) by reshaping channels
into spatial dimensions. R=2 hardcoded (comptime).

Forward: output[oh,ow,c] = input[oh//R, ow//R, c*R*R + (oh%R)*R + (ow%R)]
Backward: bijection — gradient maps 1:1.

Usage: mojo run src/omen/kernels/pixel_shuffle.mojo
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

comptime dtype = DType.float32
comptime BLOCK_SIZE = 512


# ════════════════════════════════════════════════════════════════════
# FORWARD KERNEL
# ════════════════════════════════════════════════════════════════════

def pixel_shuffle_forward_kernel[LT: TensorLayout, LT1: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT1, MutAnyOrigin],
    H: Int, W: Int, C: Int, R: Int,
):
    comptime assert input.flat_rank == 3 and output.flat_rank == 3
    var idx = global_idx.x
    var H_out = H * R
    var W_out = W * R
    var total = H_out * W_out * C
    if idx < total:
        var oh = idx // (W_out * C)
        var rem = idx - oh * W_out * C
        var ow = rem // C
        var c = rem - ow * C
        var ih = oh // R
        var iw = ow // R
        var rh = oh - ih * R
        var rw = ow - iw * R
        var c_in = c * R * R + rh * R + rw
        var val = rebind[Scalar[dtype]](input[ih, iw, c_in])
        output[oh, ow, c] = rebind[output.ElementType](val)


# ════════════════════════════════════════════════════════════════════
# BACKWARD KERNEL
# ════════════════════════════════════════════════════════════════════

def pixel_shuffle_backward_kernel[LT: TensorLayout, LT1: TensorLayout](
    d_output: TileTensor[dtype, LT, MutAnyOrigin],
    d_input: TileTensor[dtype, LT1, MutAnyOrigin],
    H: Int, W: Int, C: Int, R: Int,
):
    comptime assert d_output.flat_rank == 3 and d_input.flat_rank == 3
    var idx = global_idx.x
    var H_out = H * R
    var W_out = W * R
    var total = H_out * W_out * C
    if idx < total:
        var oh = idx // (W_out * C)
        var rem = idx - oh * W_out * C
        var ow = rem // C
        var c = rem - ow * C
        var ih = oh // R
        var iw = ow // R
        var rh = oh - ih * R
        var rw = ow - iw * R
        var c_in = c * R * R + rh * R + rw
        var g = rebind[Scalar[dtype]](d_output[oh, ow, c])
        d_input[ih, iw, c_in] = rebind[d_input.ElementType](g)


# ════════════════════════════════════════════════════════════════════
# TEST
# ════════════════════════════════════════════════════════════════════

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    print("Pixel Shuffle Kernel Test")
    print("=" * 40)

    var ctx = DeviceContext()

    # Test: (4, 4, 8) -> (8, 8, 2) with R=2
    comptime TH = 4
    comptime TW = 4
    comptime TC = 2
    comptime TR = 2
    comptime TC_IN = TC * TR * TR
    comptime TH_OUT = TH * TR
    comptime TW_OUT = TW * TR

    # Allocate
    var in_buf = ctx.enqueue_create_buffer[dtype](TH * TW * TC_IN)
    var out_buf = ctx.enqueue_create_buffer[dtype](TH_OUT * TW_OUT * TC)

    # Initialize: input[ih, iw, c_in] = ih*100 + iw*10 + c_in
    var host_in = ctx.enqueue_create_host_buffer[dtype](TH * TW * TC_IN)
    for ih in range(TH):
        for iw in range(TW):
            for ci in range(TC_IN):
                host_in[ih * TW * TC_IN + iw * TC_IN + ci] = Float32(
                    ih * 100 + iw * 10 + ci
                )
    ctx.enqueue_copy(dst_buf=in_buf, src_buf=host_in)

    # Run forward
    comptime in_3d = row_major[TH, TW, TC_IN]()
    comptime out_3d = row_major[TH_OUT, TW_OUT, TC]()
    comptime fwd = pixel_shuffle_forward_kernel[type_of(in_3d), type_of(out_3d)]

    ctx.enqueue_function[fwd](
        TileTensor(in_buf, in_3d),
        TileTensor(out_buf, out_3d),
        TH, TW, TC, TR,
        grid_dim=ceildiv(TH_OUT * TW_OUT * TC, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()

    # Verify: output[oh,ow,c] = input[oh//2, ow//2, c*4 + (oh%2)*2 + (ow%2)]
    with out_buf.map_to_host() as host_out:
        var result = TileTensor(host_out, out_3d)
        var ok = True
        # output[0,0,0] = input[0,0, 0*4+0*2+0] = input[0,0,0] = 0.0
        var v = rebind[Scalar[dtype]](result[0, 0, 0])
        if abs(v - 0.0) > 0.001:
            print("FAIL: output[0,0,0] = {} expected 0.0".format(v))
            ok = False
        # output[0,1,0] = input[0,0, 0*4+0*2+1] = input[0,0,1] = 1.0
        v = rebind[Scalar[dtype]](result[0, 1, 0])
        if abs(v - 1.0) > 0.001:
            print("FAIL: output[0,1,0] = {} expected 1.0".format(v))
            ok = False
        # output[1,0,0] = input[0,0, 0*4+1*2+0] = input[0,0,2] = 2.0
        v = rebind[Scalar[dtype]](result[1, 0, 0])
        if abs(v - 2.0) > 0.001:
            print("FAIL: output[1,0,0] = {} expected 2.0".format(v))
            ok = False
        # output[1,1,0] = input[0,0, 0*4+1*2+1] = input[0,0,3] = 3.0
        v = rebind[Scalar[dtype]](result[1, 1, 0])
        if abs(v - 3.0) > 0.001:
            print("FAIL: output[1,1,0] = {} expected 3.0".format(v))
            ok = False
        # output[0,0,1] = input[0,0, 1*4+0*2+0] = input[0,0,4] = 4.0
        v = rebind[Scalar[dtype]](result[0, 0, 1])
        if abs(v - 4.0) > 0.001:
            print("FAIL: output[0,0,1] = {} expected 4.0".format(v))
            ok = False
        # output[2,0,0] = input[1,0, 0*4+0*2+0] = input[1,0,0] = 100.0
        v = rebind[Scalar[dtype]](result[2, 0, 0])
        if abs(v - 100.0) > 0.001:
            print("FAIL: output[2,0,0] = {} expected 100.0".format(v))
            ok = False

        if ok:
            print("Forward: PASS")

    # Test backward
    var dout_buf = ctx.enqueue_create_buffer[dtype](TH_OUT * TW_OUT * TC)
    var din_buf = ctx.enqueue_create_buffer[dtype](TH * TW * TC_IN)
    var host_dout = ctx.enqueue_create_host_buffer[dtype](TH_OUT * TW_OUT * TC)
    for i in range(TH_OUT * TW_OUT * TC):
        host_dout[i] = Float32(i + 1)
    ctx.enqueue_copy(dst_buf=dout_buf, src_buf=host_dout)
    din_buf.enqueue_fill(0.0)

    comptime bwd = pixel_shuffle_backward_kernel[
        type_of(out_3d), type_of(in_3d)
    ]
    ctx.enqueue_function[bwd](
        TileTensor(dout_buf, out_3d),
        TileTensor(din_buf, in_3d),
        TH, TW, TC, TR,
        grid_dim=ceildiv(TH_OUT * TW_OUT * TC, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()

    # Verify: d_input[oh//R, ow//R, c*R*R + (oh%R)*R + (ow%R)] = d_output[oh,ow,c]
    with din_buf.map_to_host() as host_din:
        var d_result = TileTensor(host_din, in_3d)
        var bwd_ok = True
        # d_output[0,0,0] = 1.0 -> d_input[0,0,0] should be 1.0
        var dv = rebind[Scalar[dtype]](d_result[0, 0, 0])
        if abs(dv - 1.0) > 0.001:
            print("FAIL: d_input[0,0,0] = {} expected 1.0".format(dv))
            bwd_ok = False
        # d_output[0,1,0]=3.0 -> d_input[0,0, c=0*R*R+0*R+1=1]
        dv = rebind[Scalar[dtype]](d_result[0, 0, 1])
        if abs(dv - 3.0) > 0.001:
            print("FAIL: d_input[0,0,1] = {} expected 3.0".format(dv))
            bwd_ok = False
        # d_output[1,0,0]=17.0 (HWC: idx=1*8*2+0*2+0=16, val=17)
        # -> d_input[0,0, c=0*R*R+1*R+0=2]
        dv = rebind[Scalar[dtype]](d_result[0, 0, 2])
        if abs(dv - 17.0) > 0.001:
            print("FAIL: d_input[0,0,2] = {} expected 17.0".format(dv))
            bwd_ok = False
        # d_output[1,1,0]=19.0 (HWC: idx=1*8*2+1*2+0=18, val=19)
        # -> d_input[0,0, c=0*R*R+1*R+1=3]
        dv = rebind[Scalar[dtype]](d_result[0, 0, 3])
        if abs(dv - 19.0) > 0.001:
            print("FAIL: d_input[0,0,3] = {} expected 19.0".format(dv))
            bwd_ok = False

        if bwd_ok:
            print("Backward: PASS")

    print("Pixel Shuffle: ALL TESTS PASSED")
