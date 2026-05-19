"""Mamba U-Net Decoder — self-contained Mojo GPU denoiser decoder.

Takes encoder outputs (fused_latent + skip features) and produces
denoised RGB via: bottleneck inject → upsample → skip → upsample → sigmoid.

Uses 1x1 convolutions (per-pixel linear) for channel mixing,
pixel shuffle for spatial upsampling, and SiLU activations.

Standalone test: synthetic data → train decoder → loss decreases.

Usage: mojo run tests/mojo_decoder.mojo
"""

from std.math import ceildiv, sqrt, exp
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from std.python import Python
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime dtype = DType.float32
comptime BLOCK = 512

# Decoder dimensions (test scale — can be scaled up)
comptime TILE = 32
comptime CH = 16
comptime LATENT = 32
comptime H1 = TILE // 2
comptime W1 = TILE // 2
comptime H2 = TILE // 4
comptime W2 = TILE // 4
comptime UP1 = CH * 4       # pixel shuffle expands: CH*4 → CH at 2x res
comptime MID = CH // 2      # 8 channels after mid-projection
comptime UP2 = 3 * 4        # 12 → 3ch at 2x res
comptime OUT = 3


# ════════════════════════════════════════════════════════════════════
# SHARED FORWARD KERNELS
# ════════════════════════════════════════════════════════════════════

def silu_kernel[LT: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin], size: Int,
):
    comptime assert input.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var x = rebind[Scalar[dtype]](input[tid])
        var sig = 1.0 / (1.0 + exp(-x))
        output[tid] = rebind[output.ElementType](x * sig)


def sigmoid_kernel[LT: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin], size: Int,
):
    comptime assert input.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var x = rebind[Scalar[dtype]](input[tid])
        output[tid] = rebind[output.ElementType](1.0 / (1.0 + exp(-x)))


def bias_add_kernel[LT: TensorLayout, LT1: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    bias: TileTensor[dtype, LT1, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin], M: Int, N: Int,
):
    comptime assert input.flat_rank == 2 and bias.flat_rank == 1
    var tid = global_idx.x
    if tid < M * N:
        var row = tid // N
        var col = tid - row * N
        var x = rebind[Scalar[dtype]](input[row, col])
        var b = rebind[Scalar[dtype]](bias[col])
        output[row, col] = rebind[output.ElementType](x + b)


def matmul_1d_kernel[LT1: TensorLayout, LT2: TensorLayout, LT3: TensorLayout](
    A: TileTensor[dtype, LT1, MutAnyOrigin],
    B: TileTensor[dtype, LT2, MutAnyOrigin],
    C: TileTensor[dtype, LT3, MutAnyOrigin], M: Int, K: Int, N: Int,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    var tid = global_idx.x
    if tid < M * N:
        var row = tid // N
        var col = tid - row * N
        var acc: C.ElementType = 0.0
        for k in range(K):
            var a = rebind[Scalar[dtype]](A[row, k])
            var b = rebind[Scalar[dtype]](B[k, col])
            acc += a * b
        C[row, col] = acc


def residual_add_kernel[LT: TensorLayout](
    x: TileTensor[dtype, LT, MutAnyOrigin],
    y: TileTensor[dtype, LT, MutAnyOrigin],
    dst: TileTensor[dtype, LT, MutAnyOrigin], size: Int,
):
    comptime assert x.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var a = rebind[Scalar[dtype]](x[tid])
        var b = rebind[Scalar[dtype]](y[tid])
        dst[tid] = rebind[dst.ElementType](a + b)


def scalar_mul_kernel[LT: TensorLayout](
    a: TileTensor[dtype, LT, MutAnyOrigin],
    b: TileTensor[dtype, LT, MutAnyOrigin],
    dst: TileTensor[dtype, LT, MutAnyOrigin], size: Int,
):
    comptime assert a.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var av = rebind[Scalar[dtype]](a[tid])
        var bv = rebind[Scalar[dtype]](b[tid])
        dst[tid] = rebind[dst.ElementType](av * bv)


def sub_kernel[LT: TensorLayout](
    a: TileTensor[dtype, LT, MutAnyOrigin],
    b: TileTensor[dtype, LT, MutAnyOrigin],
    dst: TileTensor[dtype, LT, MutAnyOrigin], size: Int,
):
    comptime assert a.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var av = rebind[Scalar[dtype]](a[tid])
        var bv = rebind[Scalar[dtype]](b[tid])
        dst[tid] = rebind[dst.ElementType](av - bv)


def square_kernel[LT: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin], size: Int,
):
    comptime assert input.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var x = rebind[Scalar[dtype]](input[tid])
        output[tid] = rebind[output.ElementType](x * x)


# Broadcast (1, C) → (N, C) by repeating row 0
def broadcast_kernel[LT: TensorLayout, LT1: TensorLayout](
    src: TileTensor[dtype, LT, MutAnyOrigin],
    dst: TileTensor[dtype, LT1, MutAnyOrigin], N: Int, C: Int,
):
    comptime assert src.flat_rank == 2 and dst.flat_rank == 2
    var idx = global_idx.x
    if idx < N * C:
        var row = idx // C
        var col = idx - row * C
        var val = rebind[Scalar[dtype]](src[0, col])
        dst[row, col] = rebind[dst.ElementType](val)


# Bottleneck inject: gate = sigmoid(latent), out = conv_out + gate * latent
def bottleneck_inject_kernel[
    LT: TensorLayout, LT1: TensorLayout, LT2: TensorLayout,
](
    latent_2d: TileTensor[dtype, LT, MutAnyOrigin],   # (1, C)
    conv_3d: TileTensor[dtype, LT1, MutAnyOrigin],     # (H, W, C)
    output: TileTensor[dtype, LT2, MutAnyOrigin],      # (H, W, C)
    H: Int, W: Int, C: Int,
):
    comptime assert latent_2d.flat_rank == 2 and conv_3d.flat_rank == 3
    comptime assert output.flat_rank == 3
    var idx = global_idx.x
    if idx < H * W * C:
        var h = idx // (W * C)
        var rem = idx - h * W * C
        var w = rem // C
        var c = rem - w * C
        var lv = rebind[Scalar[dtype]](latent_2d[0, c])
        var gate = 1.0 / (1.0 + exp(-lv))
        var cv = rebind[Scalar[dtype]](conv_3d[h, w, c])
        output[h, w, c] = rebind[output.ElementType](cv + gate * lv)


# Pixel shuffle: (H,W,C*R*R) → (H*R,W*R,C)
def pixel_shuffle_fwd_kernel[LT: TensorLayout, LT1: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT1, MutAnyOrigin],
    H: Int, W: Int, C: Int, R: Int,
):
    comptime assert input.flat_rank == 3 and output.flat_rank == 3
    var idx = global_idx.x
    var Ho = H * R
    var Wo = W * R
    if idx < Ho * Wo * C:
        var oh = idx // (Wo * C)
        var rem = idx - oh * Wo * C
        var ow = rem // C
        var c = rem - ow * C
        var ih = oh // R
        var iw = ow // R
        var ci = c * R * R + (oh - ih * R) * R + (ow - iw * R)
        var val = rebind[Scalar[dtype]](input[ih, iw, ci])
        output[oh, ow, c] = rebind[output.ElementType](val)


def adamw_kernel[LT: TensorLayout](
    param: TileTensor[dtype, LT, MutAnyOrigin],
    grad: TileTensor[dtype, LT, MutAnyOrigin],
    m: TileTensor[dtype, LT, MutAnyOrigin],
    v: TileTensor[dtype, LT, MutAnyOrigin], size: Int,
    lr: Float32, b1: Float32, b2: Float32, eps: Float32, wd: Float32,
):
    comptime assert param.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var p = rebind[Scalar[dtype]](param[tid])
        var g = rebind[Scalar[dtype]](grad[tid])
        var mv = rebind[Scalar[dtype]](m[tid])
        var vv = rebind[Scalar[dtype]](v[tid])
        var nm = b1 * mv + (1.0 - b1) * g
        var nv = b2 * vv + (1.0 - b2) * g * g
        m[tid] = rebind[m.ElementType](nm)
        v[tid] = rebind[v.ElementType](nv)
        var upd = lr * nm / (sqrt(nv) + eps) + wd * lr * p
        param[tid] = rebind[param.ElementType](p - upd)


# ════════════════════════════════════════════════════════════════════
# BACKWARD KERNELS
# ════════════════════════════════════════════════════════════════════

def silu_backward_kernel[LT: TensorLayout](
    x_saved: TileTensor[dtype, LT, MutAnyOrigin],
    grad_out: TileTensor[dtype, LT, MutAnyOrigin],
    grad_in: TileTensor[dtype, LT, MutAnyOrigin], size: Int,
):
    comptime assert x_saved.flat_rank == 1 and grad_in.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var x = rebind[Scalar[dtype]](x_saved[tid])
        var sig = 1.0 / (1.0 + exp(-x))
        var ds = sig * (1.0 + x * (1.0 - sig))
        var g = rebind[Scalar[dtype]](grad_out[tid])
        grad_in[tid] = rebind[grad_in.ElementType](g * ds)


def sigmoid_backward_kernel[LT: TensorLayout](
    x_saved: TileTensor[dtype, LT, MutAnyOrigin],
    grad_out: TileTensor[dtype, LT, MutAnyOrigin],
    grad_in: TileTensor[dtype, LT, MutAnyOrigin], size: Int,
):
    comptime assert x_saved.flat_rank == 1 and grad_in.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var x = rebind[Scalar[dtype]](x_saved[tid])
        var sig = 1.0 / (1.0 + exp(-x))
        var g = rebind[Scalar[dtype]](grad_out[tid])
        grad_in[tid] = rebind[grad_in.ElementType](g * sig * (1.0 - sig))


def matmul_backward_A[
    LT1: TensorLayout, LT2: TensorLayout, LT3: TensorLayout,
    LT4: TensorLayout, LT5: TensorLayout,
](
    A: TileTensor[dtype, LT1, MutAnyOrigin],
    B: TileTensor[dtype, LT2, MutAnyOrigin],
    dC: TileTensor[dtype, LT3, MutAnyOrigin],
    dA: TileTensor[dtype, LT4, MutAnyOrigin],
    dB: TileTensor[dtype, LT5, MutAnyOrigin], M: Int, K: Int, N: Int,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and dC.flat_rank == 2
    comptime assert dA.flat_rank == 2 and dB.flat_rank == 2
    var tid = global_idx.x
    if tid < M * K:
        var i = tid // K
        var k = tid - i * K
        var acc: dA.ElementType = 0.0
        for j in range(N):
            acc += rebind[Scalar[dtype]](dC[i, j]) * rebind[Scalar[dtype]](B[k, j])
        dA[i, k] = acc


def matmul_backward_B[
    LT1: TensorLayout, LT2: TensorLayout, LT3: TensorLayout, LT5: TensorLayout,
](
    A: TileTensor[dtype, LT1, MutAnyOrigin],
    dC: TileTensor[dtype, LT3, MutAnyOrigin],
    dB: TileTensor[dtype, LT5, MutAnyOrigin], M: Int, K: Int, N: Int,
):
    comptime assert A.flat_rank == 2 and dC.flat_rank == 2 and dB.flat_rank == 2
    var tid = global_idx.x
    if tid < K * N:
        var k = tid // N
        var j = tid - k * N
        var acc: dB.ElementType = 0.0
        for i in range(M):
            acc += rebind[Scalar[dtype]](A[i, k]) * rebind[Scalar[dtype]](dC[i, j])
        dB[k, j] = acc


def bias_backward_kernel[LT: TensorLayout, LT1: TensorLayout](
    dout: TileTensor[dtype, LT, MutAnyOrigin],
    db: TileTensor[dtype, LT1, MutAnyOrigin], M: Int, N: Int,
):
    comptime assert dout.flat_rank == 2 and db.flat_rank == 1
    var tid = global_idx.x
    if tid < N:
        var acc: Scalar[dtype] = 0.0
        for i in range(M):
            acc += rebind[Scalar[dtype]](dout[i, tid])
        db[tid] = rebind[db.ElementType](acc)


def residual_add_backward_kernel[LT: TensorLayout](
    dout: TileTensor[dtype, LT, MutAnyOrigin],
    da: TileTensor[dtype, LT, MutAnyOrigin],
    db: TileTensor[dtype, LT, MutAnyOrigin], size: Int,
):
    comptime assert dout.flat_rank == 1 and da.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var g = rebind[Scalar[dtype]](dout[tid])
        da[tid] = rebind[da.ElementType](g)
        db[tid] = rebind[db.ElementType](g)


def scalar_mul_backward_kernel[LT: TensorLayout](
    a: TileTensor[dtype, LT, MutAnyOrigin],
    b: TileTensor[dtype, LT, MutAnyOrigin],
    dout: TileTensor[dtype, LT, MutAnyOrigin],
    da: TileTensor[dtype, LT, MutAnyOrigin],
    db: TileTensor[dtype, LT, MutAnyOrigin], size: Int,
):
    comptime assert a.flat_rank == 1 and da.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var av = rebind[Scalar[dtype]](a[tid])
        var bv = rebind[Scalar[dtype]](b[tid])
        var g = rebind[Scalar[dtype]](dout[tid])
        da[tid] = rebind[da.ElementType](g * bv)
        db[tid] = rebind[db.ElementType](g * av)


def sub_backward_kernel[LT: TensorLayout](
    dout: TileTensor[dtype, LT, MutAnyOrigin],
    da: TileTensor[dtype, LT, MutAnyOrigin],
    db: TileTensor[dtype, LT, MutAnyOrigin], size: Int,
):
    comptime assert dout.flat_rank == 1 and da.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var g = rebind[Scalar[dtype]](dout[tid])
        da[tid] = rebind[da.ElementType](g)
        db[tid] = rebind[db.ElementType](0.0 - g)


# Pixel shuffle backward: d_output(H*R,W*R,C) → d_input(H,W,C*R*R)
def pixel_shuffle_bwd_kernel[LT: TensorLayout, LT1: TensorLayout](
    d_output: TileTensor[dtype, LT, MutAnyOrigin],
    d_input: TileTensor[dtype, LT1, MutAnyOrigin],
    H: Int, W: Int, C: Int, R: Int,
):
    comptime assert d_output.flat_rank == 3 and d_input.flat_rank == 3
    var idx = global_idx.x
    var Ho = H * R
    var Wo = W * R
    if idx < Ho * Wo * C:
        var oh = idx // (Wo * C)
        var rem = idx - oh * Wo * C
        var ow = rem // C
        var c = rem - ow * C
        var ih = oh // R
        var iw = ow // R
        var ci = c * R * R + (oh - ih * R) * R + (ow - iw * R)
        var g = rebind[Scalar[dtype]](d_output[oh, ow, c])
        d_input[ih, iw, ci] = rebind[d_input.ElementType](g)


# Bottleneck inject backward: d_latent_2d, d_conv_3d from d_output
# out = conv + sigmoid(latent) * latent
# d_conv = dout (from residual)
# d_latent[c] = sum_{h,w}( dout[h,w,c] * (sigmoid'(latent[c])*latent[c] + sigmoid(latent[c])) )
def bottleneck_inject_backward_kernel[
    LT: TensorLayout, LT1: TensorLayout, LT2: TensorLayout, LT3: TensorLayout,
](
    latent_2d: TileTensor[dtype, LT, MutAnyOrigin],
    dout_3d: TileTensor[dtype, LT1, MutAnyOrigin],
    d_latent: TileTensor[dtype, LT2, MutAnyOrigin],
    d_conv: TileTensor[dtype, LT3, MutAnyOrigin],
    H: Int, W: Int, C: Int,
):
    comptime assert latent_2d.flat_rank == 2 and dout_3d.flat_rank == 3
    comptime assert d_latent.flat_rank == 2 and d_conv.flat_rank == 3
    var c = global_idx.x
    if c < C:
        var lv = rebind[Scalar[dtype]](latent_2d[0, c])
        var sig = 1.0 / (1.0 + exp(-lv))
        var deriv = sig * (1.0 - sig) * lv + sig
        var dl_acc: Scalar[dtype] = 0.0
        for h in range(H):
            for w in range(W):
                var dv = rebind[Scalar[dtype]](dout_3d[h, w, c])
                dl_acc += dv * deriv
                d_conv[h, w, c] = rebind[d_conv.ElementType](dv)
        d_latent[0, c] = rebind[d_latent.ElementType](dl_acc)


def reduce_sum_kernel[LT: TensorLayout, LT1: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT1, MutAnyOrigin], size: Int,
):
    comptime assert input.flat_rank == 1 and output.flat_rank == 1
    var tid = global_idx.x
    var partial: Scalar[dtype] = 0.0
    for i in range(tid, size, BLOCK):
        partial += rebind[Scalar[dtype]](input[i])
    var shared = stack_allocation[dtype,
        address_space=AddressSpace.SHARED](row_major[BLOCK]()).fill(0)
    shared[tid] = rebind[shared.ElementType](partial)
    barrier()
    var active = BLOCK
    comptime for _ in range(9):
        active //= 2
        if tid < active:
            var a = rebind[Scalar[dtype]](shared[tid])
            var b = rebind[Scalar[dtype]](shared[tid + active])
            shared[tid] = rebind[shared.ElementType](a + b)
        barrier()
    if tid == 0:
        output[0] = rebind[output.ElementType](rebind[Scalar[dtype]](shared[0]))


# ════════════════════════════════════════════════════════════════════
# MAIN — TRAINING LOOP
# ════════════════════════════════════════════════════════════════════

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    print("Mamba U-Net Decoder — Training Test")
    print("=" * 50)
    print("Tile={} CH={} LAT={} H1={} H2={}".format(TILE, CH, LATENT, H1, H2))

    var ctx = DeviceContext()
    var np = Python.import_module("numpy")
    var rng = np.random.RandomState(42)

    # ── Allocate decoder parameters ──────────────────────────────────
    # Bottleneck: Linear(LATENT → CH)
    var bn_w = ctx.enqueue_create_buffer[dtype](LATENT * CH)
    var bn_b = ctx.enqueue_create_buffer[dtype](CH)
    # Upsample1: Linear(CH → UP1)
    var u1_w = ctx.enqueue_create_buffer[dtype](CH * UP1)
    var u1_b = ctx.enqueue_create_buffer[dtype](UP1)
    # Mid: Linear(CH → MID)
    var md_w = ctx.enqueue_create_buffer[dtype](CH * MID)
    var md_b = ctx.enqueue_create_buffer[dtype](MID)
    # Upsample2: Linear(MID → UP2)
    var u2_w = ctx.enqueue_create_buffer[dtype](MID * UP2)
    var u2_b = ctx.enqueue_create_buffer[dtype](UP2)

    # Xavier init
    var bn_w_np = rng.randn(LATENT, CH).astype("float32") * np.sqrt(2.0 / LATENT)
    for i in range(LATENT * CH):
        var host = ctx.enqueue_create_host_buffer[dtype](1)
        host[0] = Float32(py=bn_w_np.ravel()[i])
        ctx.enqueue_copy(dst_buf=bn_w, src_buf=host)
    bn_b.enqueue_fill(0.0)
    var u1_np = rng.randn(CH, UP1).astype("float32") * np.sqrt(2.0 / CH)
    for i in range(CH * UP1):
        var host = ctx.enqueue_create_host_buffer[dtype](1)
        host[0] = Float32(py=u1_np.ravel()[i])
        ctx.enqueue_copy(dst_buf=u1_w, src_buf=host)
    u1_b.enqueue_fill(0.0)
    var md_np = rng.randn(CH, MID).astype("float32") * np.sqrt(2.0 / CH)
    for i in range(CH * MID):
        var host = ctx.enqueue_create_host_buffer[dtype](1)
        host[0] = Float32(py=md_np.ravel()[i])
        ctx.enqueue_copy(dst_buf=md_w, src_buf=host)
    md_b.enqueue_fill(0.0)
    var u2_np = rng.randn(MID, UP2).astype("float32") * np.sqrt(2.0 / MID)
    for i in range(MID * UP2):
        var host = ctx.enqueue_create_host_buffer[dtype](1)
        host[0] = Float32(py=u2_np.ravel()[i])
        ctx.enqueue_copy(dst_buf=u2_w, src_buf=host)
    u2_b.enqueue_fill(0.0)
    ctx.synchronize()

    # ── AdamW state (m, v per param) ─────────────────────────────────
    var bn_w_m = ctx.enqueue_create_buffer[dtype](LATENT * CH)
    var bn_w_v = ctx.enqueue_create_buffer[dtype](LATENT * CH)
    var bn_b_m = ctx.enqueue_create_buffer[dtype](CH)
    var bn_b_v = ctx.enqueue_create_buffer[dtype](CH)
    var u1_w_m = ctx.enqueue_create_buffer[dtype](CH * UP1)
    var u1_w_v = ctx.enqueue_create_buffer[dtype](CH * UP1)
    var u1_b_m = ctx.enqueue_create_buffer[dtype](UP1)
    var u1_b_v = ctx.enqueue_create_buffer[dtype](UP1)
    var md_w_m = ctx.enqueue_create_buffer[dtype](CH * MID)
    var md_w_v = ctx.enqueue_create_buffer[dtype](CH * MID)
    var md_b_m = ctx.enqueue_create_buffer[dtype](MID)
    var md_b_v = ctx.enqueue_create_buffer[dtype](MID)
    var u2_w_m = ctx.enqueue_create_buffer[dtype](MID * UP2)
    var u2_w_v = ctx.enqueue_create_buffer[dtype](MID * UP2)
    var u2_b_m = ctx.enqueue_create_buffer[dtype](UP2)
    var u2_b_v = ctx.enqueue_create_buffer[dtype](UP2)
    for buf in [bn_w_m, bn_w_v, bn_b_m, bn_b_v, u1_w_m, u1_w_v, u1_b_m, u1_b_v, md_w_m, md_w_v, md_b_m, md_b_v, u2_w_m, u2_w_v, u2_b_m, u2_b_v]:
        buf.enqueue_fill(0.0)

    # ── Gradient buffers ──────────────────────────────────────────────
    var d_bn_w = ctx.enqueue_create_buffer[dtype](LATENT * CH)
    var d_bn_b = ctx.enqueue_create_buffer[dtype](CH)
    var d_u1_w = ctx.enqueue_create_buffer[dtype](CH * UP1)
    var d_u1_b = ctx.enqueue_create_buffer[dtype](UP1)
    var d_md_w = ctx.enqueue_create_buffer[dtype](CH * MID)
    var d_md_b = ctx.enqueue_create_buffer[dtype](MID)
    var d_u2_w = ctx.enqueue_create_buffer[dtype](MID * UP2)
    var d_u2_b = ctx.enqueue_create_buffer[dtype](UP2)

    # ── Activation buffers (forward) ──────────────────────────────────
    var bn_out_2d = ctx.enqueue_create_buffer[dtype](1 * CH)
    var injected = ctx.enqueue_create_buffer[dtype](H2 * W2 * CH)
    var up1_mm = ctx.enqueue_create_buffer[dtype](H2 * W2 * UP1)
    var up1_bias = ctx.enqueue_create_buffer[dtype](H2 * W2 * UP1)
    var up1_ps = ctx.enqueue_create_buffer[dtype](H1 * W1 * CH)
    var skip_out = ctx.enqueue_create_buffer[dtype](H1 * W1 * CH)
    var md_mm = ctx.enqueue_create_buffer[dtype](H1 * W1 * MID)
    var md_pre_silu = ctx.enqueue_create_buffer[dtype](H1 * W1 * MID)
    var md_silu = ctx.enqueue_create_buffer[dtype](H1 * W1 * MID)
    var u2_mm = ctx.enqueue_create_buffer[dtype](H1 * W1 * UP2)
    var u2_bias = ctx.enqueue_create_buffer[dtype](H1 * W1 * UP2)
    var u2_ps = ctx.enqueue_create_buffer[dtype](TILE * TILE * OUT)
    var output = ctx.enqueue_create_buffer[dtype](TILE * TILE * OUT)
    var loss_diff = ctx.enqueue_create_buffer[dtype](TILE * TILE * OUT)
    var loss_sq = ctx.enqueue_create_buffer[dtype](TILE * TILE * OUT)
    var loss_buf = ctx.enqueue_create_buffer[dtype](1)

    # Synthetic encoder outputs (fixed for test)
    var fused = ctx.enqueue_create_buffer[dtype](LATENT)
    var conv1_silu = ctx.enqueue_create_buffer[dtype](H1 * W1 * CH)
    var conv2_silu = ctx.enqueue_create_buffer[dtype](H2 * W2 * CH)
    var target = ctx.enqueue_create_buffer[dtype](TILE * TILE * OUT)
    var host_f = ctx.enqueue_create_host_buffer[dtype](LATENT)
    for i in range(LATENT):
        host_f[i] = Float32(py=rng.randn())
    ctx.enqueue_copy(dst_buf=fused, src_buf=host_f)
    var host_c1 = ctx.enqueue_create_host_buffer[dtype](H1 * W1 * CH)
    for i in range(H1 * W1 * CH):
        host_c1[i] = Float32(py=rng.randn() * 0.1)
    ctx.enqueue_copy(dst_buf=conv1_silu, src_buf=host_c1)
    var host_c2 = ctx.enqueue_create_host_buffer[dtype](H2 * W2 * CH)
    for i in range(H2 * W2 * CH):
        host_c2[i] = Float32(py=rng.randn() * 0.1)
    ctx.enqueue_copy(dst_buf=conv2_silu, src_buf=host_c2)
    var host_t = ctx.enqueue_create_host_buffer[dtype](TILE * TILE * OUT)
    for i in range(TILE * TILE * OUT):
        host_t[i] = Float32(py=rng.uniform(0.0, 1.0))
    ctx.enqueue_copy(dst_buf=target, src_buf=host_t)

    # ── Pre-bind kernels ──────────────────────────────────────────────
    comptime layout_1 = row_major[1]()
    comptime layout_ch = row_major[CH]()
    comptime layout_up1 = row_major[UP1]()
    comptime layout_mid = row_major[MID]()
    comptime layout_up2 = row_major[UP2]()
    comptime layout_out = row_major[OUT]()
    comptime layout_1_ch = row_major[1, CH]()
    comptime layout_1_lat = row_major[1, LATENT]()
    comptime layout_hw2_ch = row_major[H2 * W2, CH]()
    comptime layout_hw2_up1 = row_major[H2 * W2, UP1]()
    comptime layout_hw1_ch = row_major[H1 * W1, CH]()
    comptime layout_hw1_mid = row_major[H1 * W1, MID]()
    comptime layout_hw1_up2 = row_major[H1 * W1, UP2]()
    comptime flat_hw1_ch = row_major[H1 * W1 * CH]()
    comptime flat_hw1_mid = row_major[H1 * W1 * MID]()
    comptime layout_tout = row_major[TILE * TILE * OUT]()
    comptime layout_3d_h2 = row_major[H2, W2, CH]()
    comptime layout_3d_h1_ch = row_major[H1, W1, CH]()
    comptime layout_3d_h2_up1 = row_major[H2, W2, UP1]()
    comptime layout_3d_h1_up2 = row_major[H1, W1, UP2]()
    comptime layout_3d_tile = row_major[TILE, TILE, OUT]()

    # Matmul kernels (bind comptime layouts)
    comptime mm_bn = matmul_1d_kernel[type_of(layout_1_lat), type_of(row_major[LATENT, CH]()), type_of(layout_1_ch)]
    comptime mm_u1 = matmul_1d_kernel[type_of(layout_hw2_ch), type_of(row_major[CH, UP1]()), type_of(layout_hw2_up1)]
    comptime mm_md = matmul_1d_kernel[type_of(layout_hw1_ch), type_of(row_major[CH, MID]()), type_of(layout_hw1_mid)]
    comptime mm_u2 = matmul_1d_kernel[type_of(layout_hw1_mid), type_of(row_major[MID, UP2]()), type_of(layout_hw1_up2)]

    comptime bias_bn = bias_add_kernel[type_of(layout_1_ch), type_of(layout_ch)]
    comptime bias_u1 = bias_add_kernel[type_of(layout_hw2_up1), type_of(layout_up1)]
    comptime bias_md = bias_add_kernel[type_of(layout_hw1_mid), type_of(layout_mid)]
    comptime bias_u2 = bias_add_kernel[type_of(layout_hw1_up2), type_of(layout_up2)]

    comptime inj_fwd = bottleneck_inject_kernel[type_of(layout_1_ch), type_of(layout_3d_h2), type_of(layout_3d_h2)]
    comptime ps1_fwd = pixel_shuffle_fwd_kernel[type_of(layout_3d_h2_up1), type_of(layout_3d_h1_ch)]
    comptime ps2_fwd = pixel_shuffle_fwd_kernel[type_of(layout_3d_h1_up2), type_of(layout_3d_tile)]

    comptime silu_1d = silu_kernel[type_of(flat_hw1_mid)]
    comptime sig_1d = sigmoid_kernel[type_of(layout_tout)]
    comptime res_flat = residual_add_kernel[type_of(flat_hw1_ch)]
    comptime sub_1d = sub_kernel[type_of(layout_tout)]
    comptime sq_1d = square_kernel[type_of(layout_tout)]
    comptime red_1d = reduce_sum_kernel[type_of(layout_tout), type_of(layout_1)]

    # Backward kernels
    comptime silu_bwd = silu_backward_kernel[type_of(flat_hw1_mid)]
    comptime sig_bwd = sigmoid_backward_kernel[type_of(layout_tout)]
    comptime ps1_bwd = pixel_shuffle_bwd_kernel[type_of(layout_3d_h1_ch), type_of(layout_3d_h2_up1)]
    comptime ps2_bwd = pixel_shuffle_bwd_kernel[type_of(layout_3d_tile), type_of(layout_3d_h1_up2)]
    comptime inj_bwd = bottleneck_inject_backward_kernel[type_of(layout_1_ch), type_of(layout_3d_h2), type_of(layout_1_ch), type_of(layout_3d_h2)]
    comptime res_bwd = residual_add_backward_kernel[type_of(flat_hw1_ch)]

    comptime bias_bwd_bn = bias_backward_kernel[type_of(layout_1_ch), type_of(layout_ch)]
    comptime bias_bwd_u1 = bias_backward_kernel[type_of(layout_hw2_up1), type_of(layout_up1)]
    comptime bias_bwd_md = bias_backward_kernel[type_of(layout_hw1_mid), type_of(layout_mid)]
    comptime bias_bwd_u2 = bias_backward_kernel[type_of(layout_hw1_up2), type_of(layout_up2)]

    comptime mm_bwd_A_bn = matmul_backward_A[type_of(layout_1_lat), type_of(row_major[LATENT, CH]()), type_of(layout_1_ch), type_of(layout_1_lat), type_of(row_major[LATENT, CH]())]
    comptime mm_bwd_B_bn = matmul_backward_B[type_of(layout_1_lat), type_of(row_major[LATENT, CH]()), type_of(layout_1_ch), type_of(row_major[LATENT, CH]())]

    comptime mm_bwd_A_u1 = matmul_backward_A[type_of(layout_hw2_ch), type_of(row_major[CH, UP1]()), type_of(layout_hw2_up1), type_of(layout_hw2_ch), type_of(row_major[CH, UP1]())]
    comptime mm_bwd_B_u1 = matmul_backward_B[type_of(layout_hw2_ch), type_of(row_major[CH, UP1]()), type_of(layout_hw2_up1), type_of(row_major[CH, UP1]())]

    comptime mm_bwd_A_md = matmul_backward_A[type_of(layout_hw1_ch), type_of(row_major[CH, MID]()), type_of(layout_hw1_mid), type_of(layout_hw1_ch), type_of(row_major[CH, MID]())]
    comptime mm_bwd_B_md = matmul_backward_B[type_of(layout_hw1_ch), type_of(row_major[CH, MID]()), type_of(layout_hw1_mid), type_of(row_major[CH, MID]())]

    comptime mm_bwd_A_u2 = matmul_backward_A[type_of(layout_hw1_mid), type_of(row_major[MID, UP2]()), type_of(layout_hw1_up2), type_of(layout_hw1_mid), type_of(row_major[MID, UP2]())]
    comptime mm_bwd_B_u2 = matmul_backward_B[type_of(layout_hw1_mid), type_of(row_major[MID, UP2]()), type_of(layout_hw1_up2), type_of(row_major[MID, UP2]())]

    # AdamW kernels
    comptime aw_lat = adamw_kernel[type_of(row_major[LATENT * CH]())]
    comptime aw_ch = adamw_kernel[type_of(layout_ch)]
    comptime aw_up1 = adamw_kernel[type_of(row_major[CH * UP1]())]
    comptime aw_up1b = adamw_kernel[type_of(layout_up1)]
    comptime aw_mid = adamw_kernel[type_of(row_major[CH * MID]())]
    comptime aw_midb = adamw_kernel[type_of(layout_mid)]
    comptime aw_up2 = adamw_kernel[type_of(row_major[MID * UP2]())]
    comptime aw_up2b = adamw_kernel[type_of(layout_up2)]

    comptime STEPS = 200
    comptime LR: Float32 = 1e-3
    print("Training {} steps...".format(STEPS))

    # Temp backward buffers
    var d_output = ctx.enqueue_create_buffer[dtype](TILE * TILE * OUT)
    var d_u2_ps = ctx.enqueue_create_buffer[dtype](H1 * W1 * UP2)
    var d_u2_bias = ctx.enqueue_create_buffer[dtype](H1 * W1 * UP2)
    var d_u2_mm = ctx.enqueue_create_buffer[dtype](H1 * W1 * UP2)
    var d_md_silu = ctx.enqueue_create_buffer[dtype](H1 * W1 * MID)
    var d_md_mm = ctx.enqueue_create_buffer[dtype](H1 * W1 * MID)
    var d_skip = ctx.enqueue_create_buffer[dtype](H1 * W1 * CH)
    var d_up1_ps = ctx.enqueue_create_buffer[dtype](H1 * W1 * CH)
    var d_up1_bias = ctx.enqueue_create_buffer[dtype](H2 * W2 * UP1)
    var d_up1_mm = ctx.enqueue_create_buffer[dtype](H2 * W2 * UP1)
    var d_injected = ctx.enqueue_create_buffer[dtype](H2 * W2 * CH)
    var d_bn_out = ctx.enqueue_create_buffer[dtype](1 * CH)

    # Trash buffers for aliasing avoidance (encoder-side grads not needed)
    var bn_bias_tmp = ctx.enqueue_create_buffer[dtype](1 * CH)
    var d_skip_trash = ctx.enqueue_create_buffer[dtype](H1 * W1 * CH)
    var d_conv_trash = ctx.enqueue_create_buffer[dtype](H2 * W2 * CH)
    var d_fused_trash = ctx.enqueue_create_buffer[dtype](LATENT)

    for step in range(1, STEPS + 1):
        # ════════════════════════════════════════════════════════════
        # FORWARD
        # ════════════════════════════════════════════════════════════
        # 1. Bottleneck: fused(1,LATENT) → Linear → (1,CH)
        ctx.enqueue_function[mm_bn](
            TileTensor(fused, layout_1_lat), TileTensor(bn_w, row_major[LATENT, CH]()), TileTensor(bn_out_2d, layout_1_ch),
            1, LATENT, CH, grid_dim=ceildiv(CH, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[bias_bn](
            TileTensor(bn_out_2d, layout_1_ch), TileTensor(bn_b, layout_ch), TileTensor(bn_bias_tmp, layout_1_ch),
            1, CH, grid_dim=ceildiv(CH, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_copy(dst_buf=bn_out_2d, src_buf=bn_bias_tmp)
        # 2. Inject: gate = sigmoid(latent), out = conv2 + gate*latent
        ctx.enqueue_function[inj_fwd](
            TileTensor(bn_out_2d, layout_1_ch), TileTensor(conv2_silu, layout_3d_h2),
            TileTensor(injected, layout_3d_h2), H2, W2, CH,
            grid_dim=ceildiv(H2 * W2 * CH, BLOCK), block_dim=BLOCK,
        )
        # 3. Upsample1: (H2*W2,CH) → Linear → (H2*W2,UP1) → PS → (H1,W1,CH)
        ctx.enqueue_function[mm_u1](
            TileTensor(injected, layout_hw2_ch), TileTensor(u1_w, row_major[CH, UP1]()), TileTensor(up1_mm, layout_hw2_up1),
            H2 * W2, CH, UP1, grid_dim=ceildiv(H2 * W2 * UP1, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[bias_u1](
            TileTensor(up1_mm, layout_hw2_up1), TileTensor(u1_b, layout_up1), TileTensor(up1_bias, layout_hw2_up1),
            H2 * W2, UP1, grid_dim=ceildiv(H2 * W2 * UP1, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[ps1_fwd](
            TileTensor(up1_bias, layout_3d_h2_up1), TileTensor(up1_ps, layout_3d_h1_ch),
            H2, W2, CH, 2, grid_dim=ceildiv(H1 * W1 * CH, BLOCK), block_dim=BLOCK,
        )
        # 4. Skip add: up1_ps + conv1_silu
        ctx.enqueue_function[res_flat](
            TileTensor(up1_ps, flat_hw1_ch), TileTensor(conv1_silu, flat_hw1_ch),
            TileTensor(skip_out, flat_hw1_ch), H1 * W1 * CH,
            grid_dim=ceildiv(H1 * W1 * CH, BLOCK), block_dim=BLOCK,
        )
        # 5. Mid: (H1*W1,CH) → Linear → (H1*W1,MID) → SiLU
        ctx.enqueue_function[mm_md](
            TileTensor(skip_out, layout_hw1_ch), TileTensor(md_w, row_major[CH, MID]()), TileTensor(md_mm, layout_hw1_mid),
            H1 * W1, CH, MID, grid_dim=ceildiv(H1 * W1 * MID, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[bias_md](
            TileTensor(md_mm, layout_hw1_mid), TileTensor(md_b, layout_mid), TileTensor(md_pre_silu, layout_hw1_mid),
            H1 * W1, MID, grid_dim=ceildiv(H1 * W1 * MID, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[silu_1d](
            TileTensor(md_pre_silu, flat_hw1_mid), TileTensor(md_silu, flat_hw1_mid), H1 * W1 * MID,
            grid_dim=ceildiv(H1 * W1 * MID, BLOCK), block_dim=BLOCK,
        )
        # 6. Upsample2: (H1*W1,MID) → Linear → (H1*W1,UP2) → PS → (TILE,TILE,OUT)
        ctx.enqueue_function[mm_u2](
            TileTensor(md_silu, layout_hw1_mid), TileTensor(u2_w, row_major[MID, UP2]()), TileTensor(u2_mm, layout_hw1_up2),
            H1 * W1, MID, UP2, grid_dim=ceildiv(H1 * W1 * UP2, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[bias_u2](
            TileTensor(u2_mm, layout_hw1_up2), TileTensor(u2_b, layout_up2), TileTensor(u2_bias, layout_hw1_up2),
            H1 * W1, UP2, grid_dim=ceildiv(H1 * W1 * UP2, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[ps2_fwd](
            TileTensor(u2_bias, layout_3d_h1_up2), TileTensor(u2_ps, layout_3d_tile),
            H1, W1, OUT, 2, grid_dim=ceildiv(TILE * TILE * OUT, BLOCK), block_dim=BLOCK,
        )
        # 7. Sigmoid output
        ctx.enqueue_function[sig_1d](
            TileTensor(u2_ps, layout_tout), TileTensor(output, layout_tout), TILE * TILE * OUT,
            grid_dim=ceildiv(TILE * TILE * OUT, BLOCK), block_dim=BLOCK,
        )

        # ════════════════════════════════════════════════════════════
        # LOSS: MSE
        # ════════════════════════════════════════════════════════════
        ctx.enqueue_function[sub_1d](
            TileTensor(output, layout_tout), TileTensor(target, layout_tout),
            TileTensor(loss_diff, layout_tout), TILE * TILE * OUT,
            grid_dim=ceildiv(TILE * TILE * OUT, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[sq_1d](
            TileTensor(loss_diff, layout_tout), TileTensor(loss_sq, layout_tout), TILE * TILE * OUT,
            grid_dim=ceildiv(TILE * TILE * OUT, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[red_1d](
            TileTensor(loss_sq, layout_tout), TileTensor(loss_buf, layout_1), TILE * TILE * OUT,
            grid_dim=1, block_dim=BLOCK,
        )

        if step % 20 == 0 or step == 1:
            ctx.synchronize()
            with loss_buf.map_to_host() as lh:
                var lt = TileTensor(lh, layout_1)
                var lv = rebind[Scalar[dtype]](lt[0])
                print("  step {}: loss = {}".format(step, lv))

        # ════════════════════════════════════════════════════════════
        # BACKWARD
        # ════════════════════════════════════════════════════════════
        # d_loss = 2 * (output - target) / N  (but skip /N, just use 2*diff)
        # Using scalar_mul for scaling (reuse loss_diff as d_loss base)
        # d_loss/d_output = 2 * (output - target) = 2 * loss_diff
        # We approximate by using loss_diff directly as grad (skip factor 2)

        # 7. Sigmoid backward: d_u2_ps
        ctx.enqueue_function[sig_bwd](
            TileTensor(u2_ps, layout_tout), TileTensor(loss_diff, layout_tout),
            TileTensor(d_output, layout_tout), TILE * TILE * OUT,
            grid_dim=ceildiv(TILE * TILE * OUT, BLOCK), block_dim=BLOCK,
        )
        # 6b. Pixel shuffle backward: d_u2_ps → d_u2_bias
        ctx.enqueue_function[ps2_bwd](
            TileTensor(d_output, layout_3d_tile), TileTensor(d_u2_ps, layout_3d_h1_up2),
            H1, W1, OUT, 2, grid_dim=ceildiv(TILE * TILE * OUT, BLOCK), block_dim=BLOCK,
        )
        # 6a. Bias backward → d_u2_b; matmul backward → d_u2_w, d_md_silu
        ctx.enqueue_function[bias_bwd_u2](
            TileTensor(d_u2_ps, layout_hw1_up2), TileTensor(d_u2_b, layout_up2),
            H1 * W1, UP2, grid_dim=ceildiv(UP2, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[mm_bwd_A_u2](
            TileTensor(md_silu, layout_hw1_mid), TileTensor(u2_w, row_major[MID, UP2]()),
            TileTensor(d_u2_ps, layout_hw1_up2), TileTensor(d_md_silu, layout_hw1_mid),
            TileTensor(d_u2_w, row_major[MID, UP2]()), H1 * W1, MID, UP2,
            grid_dim=ceildiv(H1 * W1 * MID, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[mm_bwd_B_u2](
            TileTensor(md_silu, layout_hw1_mid), TileTensor(d_u2_ps, layout_hw1_up2),
            TileTensor(d_u2_w, row_major[MID, UP2]()), H1 * W1, MID, UP2,
            grid_dim=ceildiv(MID * UP2, BLOCK), block_dim=BLOCK,
        )
        # 5. SiLU backward → d_md_mm; bias backward → d_md_b; matmul → d_md_w, d_skip
        ctx.enqueue_function[silu_bwd](
            TileTensor(md_pre_silu, flat_hw1_mid), TileTensor(d_md_silu, flat_hw1_mid),
            TileTensor(d_md_mm, flat_hw1_mid), H1 * W1 * MID,
            grid_dim=ceildiv(H1 * W1 * MID, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[bias_bwd_md](
            TileTensor(d_md_mm, layout_hw1_mid), TileTensor(d_md_b, layout_mid),
            H1 * W1, MID, grid_dim=ceildiv(MID, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[mm_bwd_A_md](
            TileTensor(skip_out, layout_hw1_ch), TileTensor(md_w, row_major[CH, MID]()),
            TileTensor(d_md_mm, layout_hw1_mid), TileTensor(d_skip, layout_hw1_ch),
            TileTensor(d_md_w, row_major[CH, MID]()), H1 * W1, CH, MID,
            grid_dim=ceildiv(H1 * W1 * CH, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[mm_bwd_B_md](
            TileTensor(skip_out, layout_hw1_ch), TileTensor(d_md_mm, layout_hw1_mid),
            TileTensor(d_md_w, row_major[CH, MID]()), H1 * W1, CH, MID,
            grid_dim=ceildiv(CH * MID, BLOCK), block_dim=BLOCK,
        )
        # 4. Residual backward: d_skip → d_up1_ps (ignore d_conv1_silu, encoder side)
        ctx.enqueue_function[res_bwd](
            TileTensor(d_skip, flat_hw1_ch), TileTensor(d_up1_ps, flat_hw1_ch),
            TileTensor(d_skip_trash, flat_hw1_ch), H1 * W1 * CH,
            grid_dim=ceildiv(H1 * W1 * CH, BLOCK), block_dim=BLOCK,
        )
        # 3b. Pixel shuffle backward: d_up1_ps → d_up1_bias
        ctx.enqueue_function[ps1_bwd](
            TileTensor(d_up1_ps, layout_3d_h1_ch), TileTensor(d_up1_bias, layout_3d_h2_up1),
            H2, W2, CH, 2, grid_dim=ceildiv(H1 * W1 * CH, BLOCK), block_dim=BLOCK,
        )
        # 3a. Bias backward → d_u1_b; matmul backward → d_u1_w, d_injected
        ctx.enqueue_function[bias_bwd_u1](
            TileTensor(d_up1_bias, layout_hw2_up1), TileTensor(d_u1_b, layout_up1),
            H2 * W2, UP1, grid_dim=ceildiv(UP1, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[mm_bwd_A_u1](
            TileTensor(injected, layout_hw2_ch), TileTensor(u1_w, row_major[CH, UP1]()),
            TileTensor(d_up1_bias, layout_hw2_up1), TileTensor(d_injected, layout_hw2_ch),
            TileTensor(d_u1_w, row_major[CH, UP1]()), H2 * W2, CH, UP1,
            grid_dim=ceildiv(H2 * W2 * CH, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[mm_bwd_B_u1](
            TileTensor(injected, layout_hw2_ch), TileTensor(d_up1_bias, layout_hw2_up1),
            TileTensor(d_u1_w, row_major[CH, UP1]()), H2 * W2, CH, UP1,
            grid_dim=ceildiv(CH * UP1, BLOCK), block_dim=BLOCK,
        )
        # 2. Bottleneck inject backward: d_injected → d_bn_out, d_conv2
        ctx.enqueue_function[inj_bwd](
            TileTensor(bn_out_2d, layout_1_ch), TileTensor(d_injected, layout_3d_h2),
            TileTensor(d_bn_out, layout_1_ch), TileTensor(d_conv_trash, layout_3d_h2),
            H2, W2, CH, grid_dim=ceildiv(CH, BLOCK), block_dim=BLOCK,
        )
        # 1. Bias backward → d_bn_b; matmul backward → d_bn_w
        ctx.enqueue_function[bias_bwd_bn](
            TileTensor(d_bn_out, layout_1_ch), TileTensor(d_bn_b, layout_ch),
            1, CH, grid_dim=ceildiv(CH, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[mm_bwd_A_bn](
            TileTensor(fused, layout_1_lat), TileTensor(bn_w, row_major[LATENT, CH]()),
            TileTensor(d_bn_out, layout_1_ch), TileTensor(d_fused_trash, layout_1_lat),
            TileTensor(d_bn_w, row_major[LATENT, CH]()), 1, LATENT, CH,
            grid_dim=ceildiv(LATENT, BLOCK), block_dim=BLOCK,
        )
        ctx.enqueue_function[mm_bwd_B_bn](
            TileTensor(fused, layout_1_lat), TileTensor(d_bn_out, layout_1_ch),
            TileTensor(d_bn_w, row_major[LATENT, CH]()), 1, LATENT, CH,
            grid_dim=ceildiv(LATENT * CH, BLOCK), block_dim=BLOCK,
        )

        # ════════════════════════════════════════════════════════════
        # ADAMW UPDATE
        # ════════════════════════════════════════════════════════════
        ctx.enqueue_function[aw_lat](TileTensor(bn_w, row_major[LATENT * CH]()), TileTensor(d_bn_w, row_major[LATENT * CH]()), TileTensor(bn_w_m, row_major[LATENT * CH]()), TileTensor(bn_w_v, row_major[LATENT * CH]()), LATENT * CH, LR, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(1e-4), grid_dim=ceildiv(LATENT * CH, BLOCK), block_dim=BLOCK)
        ctx.enqueue_function[aw_ch](TileTensor(bn_b, layout_ch), TileTensor(d_bn_b, layout_ch), TileTensor(bn_b_m, layout_ch), TileTensor(bn_b_v, layout_ch), CH, LR, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(1e-4), grid_dim=ceildiv(CH, BLOCK), block_dim=BLOCK)
        ctx.enqueue_function[aw_up1](TileTensor(u1_w, row_major[CH * UP1]()), TileTensor(d_u1_w, row_major[CH * UP1]()), TileTensor(u1_w_m, row_major[CH * UP1]()), TileTensor(u1_w_v, row_major[CH * UP1]()), CH * UP1, LR, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(1e-4), grid_dim=ceildiv(CH * UP1, BLOCK), block_dim=BLOCK)
        ctx.enqueue_function[aw_up1b](TileTensor(u1_b, layout_up1), TileTensor(d_u1_b, layout_up1), TileTensor(u1_b_m, layout_up1), TileTensor(u1_b_v, layout_up1), UP1, LR, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(1e-4), grid_dim=ceildiv(UP1, BLOCK), block_dim=BLOCK)
        ctx.enqueue_function[aw_mid](TileTensor(md_w, row_major[CH * MID]()), TileTensor(d_md_w, row_major[CH * MID]()), TileTensor(md_w_m, row_major[CH * MID]()), TileTensor(md_w_v, row_major[CH * MID]()), CH * MID, LR, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(1e-4), grid_dim=ceildiv(CH * MID, BLOCK), block_dim=BLOCK)
        ctx.enqueue_function[aw_midb](TileTensor(md_b, layout_mid), TileTensor(d_md_b, layout_mid), TileTensor(md_b_m, layout_mid), TileTensor(md_b_v, layout_mid), MID, LR, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(1e-4), grid_dim=ceildiv(MID, BLOCK), block_dim=BLOCK)
        ctx.enqueue_function[aw_up2](TileTensor(u2_w, row_major[MID * UP2]()), TileTensor(d_u2_w, row_major[MID * UP2]()), TileTensor(u2_w_m, row_major[MID * UP2]()), TileTensor(u2_w_v, row_major[MID * UP2]()), MID * UP2, LR, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(1e-4), grid_dim=ceildiv(MID * UP2, BLOCK), block_dim=BLOCK)
        ctx.enqueue_function[aw_up2b](TileTensor(u2_b, layout_up2), TileTensor(d_u2_b, layout_up2), TileTensor(u2_b_m, layout_up2), TileTensor(u2_b_v, layout_up2), UP2, LR, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(1e-4), grid_dim=ceildiv(UP2, BLOCK), block_dim=BLOCK)

    ctx.synchronize()
    with loss_buf.map_to_host() as lh:
        var lt = TileTensor(lh, layout_1)
        var lv = rebind[Scalar[dtype]](lt[0])
        print("\nFinal loss: {}".format(lv))

    print("Decoder training complete!")
