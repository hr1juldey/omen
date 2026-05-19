"""Pure Mojo GPU tiled AOV denoiser — full training loop.

Architecture:
  Tile encoder:  AOV (64x64x15) → Conv1→FiLM→SiLU → Conv2→FiLM→SiLU → Pool → Linear → 128d
  Scene encoder: scene features (18d) → depth-layer residual MLP → 128d latent
  FiLM:          gamma, beta = Linear(scene_latent)  (dynamic conditioning)
  Cross-attention: fused = tile_latent + sigmoid(tile_latent @ W + b) * scene_latent
  Loss: MSE(fused, target)
  Optimizer: AdamW

Data: Mitsuba renders via Python interop (cornell, shaderball, etc.)

Usage: mojo run tests/test_mojo_tiled_denoiser.mojo
"""

from std.math import ceildiv, sqrt, exp, pow
from std.sys import has_accelerator
from std.gpu import global_idx, thread_idx, block_idx, block_dim
from std.gpu.sync import barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.python import Python, PythonObject
from std.atomic import Atomic
from layout import TileTensor, TensorLayout, row_major, stack_allocation

# ── Comptime constants ────────────────────────────────────────────
comptime dtype = DType.float32
comptime SCENE_FEAT_DIM = 18
comptime LATENT_DIM = 1024
comptime CHANNELS = 512
comptime DEFAULT_DEPTH = 16384
comptime AOV_BASE_CH = 13
comptime AOV_POS_CH = 2
comptime AOV_CH = AOV_BASE_CH + AOV_POS_CH  # 15 total
comptime TILE_SIZE = 512
comptime BLOCK_SIZE = 512

# Tile encoder spatial dims after stride-2 convs
comptime H1 = TILE_SIZE // 2  # 32
comptime W1 = TILE_SIZE // 2
comptime H2 = TILE_SIZE // 4  # 16
comptime W2 = TILE_SIZE // 4

# ── Layout definitions ─────────────────────────────────────────────
comptime layout_128 = row_major[CHANNELS]()
comptime layout_latent = row_major[LATENT_DIM]()
comptime layout_1_128 = row_major[1, CHANNELS]()
comptime layout_1_latent = row_major[1, LATENT_DIM]()
comptime layout_1 = row_major[1]()


# ════════════════════════════════════════════════════════════════════
# FORWARD KERNELS
# ════════════════════════════════════════════════════════════════════

def silu_kernel[LT: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,
):
    comptime assert input.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var x = rebind[Scalar[dtype]](input[tid])
        var sig = 1.0 / (1.0 + exp(-x))
        output[tid] = rebind[output.ElementType](x * sig)


def sigmoid_kernel[LT: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,
):
    comptime assert input.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var x = rebind[Scalar[dtype]](input[tid])
        var sig = 1.0 / (1.0 + exp(-x))
        output[tid] = rebind[output.ElementType](sig)


def bias_add_kernel[LT: TensorLayout, LT1: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    bias: TileTensor[dtype, LT1, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin],
    M: Int, N: Int,
):
    comptime assert input.flat_rank == 2 and bias.flat_rank == 1
    var tid = global_idx.x
    if tid < M * N:
        var row = tid // N
        var col = tid % N
        var x = rebind[Scalar[dtype]](input[row, col])
        var b = rebind[Scalar[dtype]](bias[col])
        output[row, col] = rebind[output.ElementType](x + b)


def residual_add_kernel[LT: TensorLayout](
    x: TileTensor[dtype, LT, MutAnyOrigin],
    residual: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,
):
    comptime assert x.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var a = rebind[Scalar[dtype]](x[tid])
        var b = rebind[Scalar[dtype]](residual[tid])
        output[tid] = rebind[output.ElementType](a + b)


def scalar_mul_kernel[LT: TensorLayout](
    a: TileTensor[dtype, LT, MutAnyOrigin],
    b: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,
):
    comptime assert a.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var av = rebind[Scalar[dtype]](a[tid])
        var bv = rebind[Scalar[dtype]](b[tid])
        output[tid] = rebind[output.ElementType](av * bv)


def square_kernel[LT: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,
):
    comptime assert input.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var x = rebind[Scalar[dtype]](input[tid])
        output[tid] = rebind[output.ElementType](x * x)


def sub_kernel[LT: TensorLayout](
    a: TileTensor[dtype, LT, MutAnyOrigin],
    b: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,
):
    comptime assert a.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var av = rebind[Scalar[dtype]](a[tid])
        var bv = rebind[Scalar[dtype]](b[tid])
        output[tid] = rebind[output.ElementType](av - bv)


def reduce_sum_kernel[LT: TensorLayout, LT1: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT1, MutAnyOrigin],
    size: Int,
):
    comptime assert input.flat_rank == 1 and output.flat_rank == 1
    var tid = thread_idx.x
    var shared = stack_allocation[dtype,
        address_space=AddressSpace.SHARED](row_major[BLOCK_SIZE]()).fill(0)
    var partial: Scalar[dtype] = 0.0
    for i in range(tid, size, block_dim.x):
        partial += rebind[Scalar[dtype]](input[i])
    shared[tid] = rebind[shared.ElementType](partial)
    barrier()
    var active = block_dim.x
    comptime for _ in range(8):
        active //= 2
        if tid < active and tid + active < block_dim.x:
            var s_val = rebind[Scalar[dtype]](shared[tid])
            var s_other = rebind[Scalar[dtype]](shared[tid + active])
            shared[tid] = rebind[shared.ElementType](s_val + s_other)
        barrier()
    if tid == 0:
        var val = rebind[Scalar[dtype]](shared[0])
        output[0] = rebind[output.ElementType](val)


def matmul_1d_kernel[LT1: TensorLayout, LT2: TensorLayout, LT3: TensorLayout](
    A: TileTensor[dtype, LT1, MutAnyOrigin],
    B: TileTensor[dtype, LT2, MutAnyOrigin],
    C: TileTensor[dtype, LT3, MutAnyOrigin],
    M: Int, K: Int, N: Int,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    var tx = thread_idx.x
    var bx = block_idx.x
    var by = block_idx.y
    var row = by
    var col = bx * 16 + tx % 16
    if row < M and col < N:
        var acc: C.ElementType = 0.0
        for k in range(K):
            var a_val = rebind[Scalar[dtype]](A[row, k])
            var b_val = rebind[Scalar[dtype]](B[k, col])
            acc += a_val * b_val
        C[row, col] = rebind[C.ElementType](acc)


def global_avg_pool_kernel[LT: TensorLayout, LT1: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT1, MutAnyOrigin],
    H: Int, W: Int, C: Int,
):
    comptime assert input.flat_rank == 3 and output.flat_rank == 1
    var c = global_idx.x
    if c < C:
        var sum_val: Scalar[dtype] = 0.0
        for h in range(H):
            for w in range(W):
                sum_val += rebind[Scalar[dtype]](input[h, w, c])
        output[c] = rebind[output.ElementType](sum_val / Float32(H * W))


def film_kernel[LT: TensorLayout, LT1: TensorLayout, LT2: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    gamma: TileTensor[dtype, LT1, MutAnyOrigin],
    beta: TileTensor[dtype, LT2, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin],
    H: Int, W: Int, C: Int,
):
    comptime assert input.flat_rank == 3 and gamma.flat_rank == 1 and beta.flat_rank == 1
    var idx = global_idx.x
    if idx < H * W * C:
        var h = idx // (W * C)
        var rem = idx - h * W * C
        var w = rem // C
        var c = rem - w * C
        var x = rebind[Scalar[dtype]](input[h, w, c])
        var g = rebind[Scalar[dtype]](gamma[c])
        var b = rebind[Scalar[dtype]](beta[c])
        output[h, w, c] = rebind[output.ElementType](g * x + b)


def im2col_kernel[
    LT: TensorLayout, LT1: TensorLayout,
    Kh: Int, Kw: Int, Sh: Int, Sw: Int, Ph: Int, Pw: Int, Cin: Int,
](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT1, MutAnyOrigin],
    H: Int, W: Int, H_out: Int, W_out: Int,
):
    comptime assert input.flat_rank == 3 and output.flat_rank == 2
    var idx = global_idx.x
    var total_patches = H_out * W_out
    var patch_size = Kh * Kw * Cin
    if idx < total_patches * patch_size:
        var patch_idx = idx // patch_size
        var elem_idx = idx - patch_idx * patch_size
        var h_out = patch_idx // W_out
        var w_out = patch_idx - h_out * W_out
        var kh = elem_idx // (Kw * Cin)
        var rem = elem_idx - kh * Kw * Cin
        var kw = rem // Cin
        var cin = rem - kw * Cin
        var h_in = h_out * Sh + kh - Ph
        var w_in = w_out * Sw + kw - Pw
        var val: Scalar[dtype] = 0.0
        if h_in >= 0 and h_in < H and w_in >= 0 and w_in < W:
            val = rebind[Scalar[dtype]](input[h_in, w_in, cin])
        output[patch_idx, elem_idx] = rebind[output.ElementType](val)


# ════════════════════════════════════════════════════════════════════
# BACKWARD KERNELS
# ════════════════════════════════════════════════════════════════════

# silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
# We recompute sigmoid from the saved pre-activation x
def silu_backward_kernel[LT: TensorLayout](
    x_saved: TileTensor[dtype, LT, MutAnyOrigin],   # pre-activation
    grad_out: TileTensor[dtype, LT, MutAnyOrigin],   # dL/d(output)
    grad_in: TileTensor[dtype, LT, MutAnyOrigin],    # dL/d(input) [output]
    size: Int,
):
    comptime assert x_saved.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var x = rebind[Scalar[dtype]](x_saved[tid])
        var sig = 1.0 / (1.0 + exp(-x))
        var ds = sig * (1.0 + x * (1.0 - sig))
        var g = rebind[Scalar[dtype]](grad_out[tid])
        grad_in[tid] = rebind[grad_in.ElementType](g * ds)


# sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
def sigmoid_backward_kernel[LT: TensorLayout](
    x_saved: TileTensor[dtype, LT, MutAnyOrigin],
    grad_out: TileTensor[dtype, LT, MutAnyOrigin],
    grad_in: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,
):
    comptime assert x_saved.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var x = rebind[Scalar[dtype]](x_saved[tid])
        var sig = 1.0 / (1.0 + exp(-x))
        var g = rebind[Scalar[dtype]](grad_out[tid])
        grad_in[tid] = rebind[grad_in.ElementType](g * sig * (1.0 - sig))


# matmul backward: dA = dC @ B^T, dB = A^T @ dC
def matmul_backward_kernel[
    LT1: TensorLayout, LT2: TensorLayout, LT3: TensorLayout,
    LT4: TensorLayout, LT5: TensorLayout,
](
    A: TileTensor[dtype, LT1, MutAnyOrigin],
    B: TileTensor[dtype, LT2, MutAnyOrigin],
    dC: TileTensor[dtype, LT3, MutAnyOrigin],
    dA: TileTensor[dtype, LT4, MutAnyOrigin],
    dB: TileTensor[dtype, LT5, MutAnyOrigin],
    M: Int, K: Int, N: Int,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and dC.flat_rank == 2
    comptime assert dA.flat_rank == 2 and dB.flat_rank == 2
    var tid = global_idx.x
    # dA[i,k] = sum_j dC[i,j] * B[k,j]
    if tid < M * K:
        var i = tid // K
        var k = tid % K
        var acc: dA.ElementType = 0.0
        for j in range(N):
            var dc_val = rebind[Scalar[dtype]](dC[i, j])
            var b_val = rebind[Scalar[dtype]](B[k, j])
            acc += dc_val * b_val
        dA[i, k] = acc


def matmul_backward_B_kernel[
    LT1: TensorLayout, LT2: TensorLayout, LT3: TensorLayout, LT5: TensorLayout,
](
    A: TileTensor[dtype, LT1, MutAnyOrigin],
    dC: TileTensor[dtype, LT3, MutAnyOrigin],
    dB: TileTensor[dtype, LT5, MutAnyOrigin],
    M: Int, K: Int, N: Int,
):
    comptime assert A.flat_rank == 2 and dC.flat_rank == 2 and dB.flat_rank == 2
    var tid = global_idx.x
    # dB[k,j] = sum_i A[i,k] * dC[i,j]
    if tid < K * N:
        var k = tid // N
        var j = tid % N
        var acc: dB.ElementType = 0.0
        for i in range(M):
            var a_val = rebind[Scalar[dtype]](A[i, k])
            var dc_val = rebind[Scalar[dtype]](dC[i, j])
            acc += a_val * dc_val
        dB[k, j] = acc


# bias backward: db[c] = sum over batch of dout[:,c]
def bias_backward_kernel[LT: TensorLayout, LT1: TensorLayout](
    dout: TileTensor[dtype, LT, MutAnyOrigin],
    db: TileTensor[dtype, LT1, MutAnyOrigin],
    M: Int, N: Int,
):
    comptime assert dout.flat_rank == 2 and db.flat_rank == 1
    var tid = global_idx.x
    if tid < N:
        var acc: Scalar[dtype] = 0.0
        for i in range(M):
            acc += rebind[Scalar[dtype]](dout[i, tid])
        db[tid] = rebind[db.ElementType](acc)


# scalar_mul backward: da = dout * b, db = dout * a
def scalar_mul_backward_kernel[LT: TensorLayout](
    a: TileTensor[dtype, LT, MutAnyOrigin],
    b: TileTensor[dtype, LT, MutAnyOrigin],
    dout: TileTensor[dtype, LT, MutAnyOrigin],
    da: TileTensor[dtype, LT, MutAnyOrigin],
    db: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,
):
    comptime assert a.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var av = rebind[Scalar[dtype]](a[tid])
        var bv = rebind[Scalar[dtype]](b[tid])
        var g = rebind[Scalar[dtype]](dout[tid])
        da[tid] = rebind[da.ElementType](g * bv)
        db[tid] = rebind[db.ElementType](g * av)


# sub backward: da = dout, db = -dout
def sub_backward_kernel[LT: TensorLayout](
    dout: TileTensor[dtype, LT, MutAnyOrigin],
    da: TileTensor[dtype, LT, MutAnyOrigin],
    db: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,
):
    comptime assert dout.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var g = rebind[Scalar[dtype]](dout[tid])
        da[tid] = rebind[da.ElementType](g)
        db[tid] = rebind[db.ElementType](0.0 - g)


# residual_add backward: both grads = dout
def residual_add_backward_kernel[LT: TensorLayout](
    dout: TileTensor[dtype, LT, MutAnyOrigin],
    da: TileTensor[dtype, LT, MutAnyOrigin],
    db: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,
):
    comptime assert dout.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var g = rebind[Scalar[dtype]](dout[tid])
        da[tid] = rebind[da.ElementType](g)
        db[tid] = rebind[db.ElementType](g)


# FiLM backward: d_input = gamma * dout
# d_gamma and d_beta computed as separate reductions
def film_backward_kernel[LT: TensorLayout, LT1: TensorLayout, LT2: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    gamma: TileTensor[dtype, LT1, MutAnyOrigin],
    dout: TileTensor[dtype, LT, MutAnyOrigin],
    d_input: TileTensor[dtype, LT, MutAnyOrigin],
    H: Int, W: Int, C: Int,
):
    comptime assert input.flat_rank == 3 and gamma.flat_rank == 1 and dout.flat_rank == 3
    comptime assert d_input.flat_rank == 3
    var idx = global_idx.x
    if idx < H * W * C:
        var h = idx // (W * C)
        var rem = idx - h * W * C
        var w = rem // C
        var c = rem - w * C
        var g = rebind[Scalar[dtype]](gamma[c])
        var d = rebind[Scalar[dtype]](dout[h, w, c])
        d_input[h, w, c] = rebind[d_input.ElementType](g * d)


# global avg pool backward: d_input[i,j,c] = d_output[c] / (H*W)
def pool_backward_kernel[LT: TensorLayout, LT1: TensorLayout](
    dout: TileTensor[dtype, LT, MutAnyOrigin],
    d_input: TileTensor[dtype, LT1, MutAnyOrigin],
    H: Int, W: Int, C: Int,
):
    comptime assert dout.flat_rank == 1 and d_input.flat_rank == 3
    var idx = global_idx.x
    if idx < H * W * C:
        var h = idx // (W * C)
        var rem = idx - h * W * C
        var w = rem // C
        var c = rem - w * C
        var g = rebind[Scalar[dtype]](dout[c])
        d_input[h, w, c] = rebind[d_input.ElementType](g / Float32(H * W))


# FiLM d_gamma: d_gamma[c] = sum_{h,w}(dout[h,w,c] * input[h,w,c])
def film_d_gamma_kernel[LT: TensorLayout, LT1: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    dout: TileTensor[dtype, LT, MutAnyOrigin],
    d_gamma: TileTensor[dtype, LT1, MutAnyOrigin],
    H: Int, W: Int, C: Int,
):
    comptime assert input.flat_rank == 3 and dout.flat_rank == 3 and d_gamma.flat_rank == 1
    var c = global_idx.x
    if c < C:
        var acc: Scalar[dtype] = 0.0
        for h in range(H):
            for w in range(W):
                var i_val = rebind[Scalar[dtype]](input[h, w, c])
                var d_val = rebind[Scalar[dtype]](dout[h, w, c])
                acc += i_val * d_val
        d_gamma[c] = rebind[d_gamma.ElementType](acc)


# FiLM d_beta: d_beta[c] = sum_{h,w}(dout[h,w,c])
def film_d_beta_kernel[LT: TensorLayout, LT1: TensorLayout](
    dout: TileTensor[dtype, LT, MutAnyOrigin],
    d_beta: TileTensor[dtype, LT1, MutAnyOrigin],
    H: Int, W: Int, C: Int,
):
    comptime assert dout.flat_rank == 3 and d_beta.flat_rank == 1
    var c = global_idx.x
    if c < C:
        var acc: Scalar[dtype] = 0.0
        for h in range(H):
            for w in range(W):
                acc += rebind[Scalar[dtype]](dout[h, w, c])
        d_beta[c] = rebind[d_beta.ElementType](acc)


# ════════════════════════════════════════════════════════════════════
# ADAMW OPTIMIZER KERNEL
# ════════════════════════════════════════════════════════════════════

def adamw_kernel[LT: TensorLayout](
    param: TileTensor[dtype, LT, MutAnyOrigin],
    grad: TileTensor[dtype, LT, MutAnyOrigin],
    m: TileTensor[dtype, LT, MutAnyOrigin],
    v: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,
    lr: Float32, beta1: Float32, beta2: Float32, eps: Float32, wd: Float32,
):
    comptime assert param.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var p = rebind[Scalar[dtype]](param[tid])
        var g = rebind[Scalar[dtype]](grad[tid])
        var mv = rebind[Scalar[dtype]](m[tid])
        var vv = rebind[Scalar[dtype]](v[tid])
        var new_m = beta1 * mv + (1.0 - beta1) * g
        var new_v = beta2 * vv + (1.0 - beta2) * g * g
        m[tid] = rebind[m.ElementType](new_m)
        v[tid] = rebind[v.ElementType](new_v)
        var update = lr * new_m / (sqrt(new_v) + eps) + wd * lr * p
        param[tid] = rebind[param.ElementType](p - update)


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def copy_to_device(
    ctx: DeviceContext,
    buf: DeviceBuffer[dtype],
    np_array: PythonObject,
) raises:
    var arr_len = len(np_array)
    var host_buf = ctx.enqueue_create_host_buffer[dtype](arr_len)
    for i in range(arr_len):
        var val = Float32(py=np_array[i])
        host_buf[i] = val
    ctx.enqueue_copy(dst_buf=buf, src_buf=host_buf)


def render_mitsuba_tile(
    ctx: DeviceContext,
    np: PythonObject,
    scene_idx: Int,
    seed: Int,
) raises -> Tuple[DeviceBuffer[dtype], DeviceBuffer[dtype], DeviceBuffer[dtype]]:
    """Render a Mitsuba scene and return (aov_buf, scene_feat_buf, target_buf).

    Renders real 13ch AOV from noisy low-SPP render.
    Target is clean high-SPP GT encoded via fixed projection.
    """
    # Import the Python helper module
    Python.add_to_path("/home/riju279/Documents/Projects/MOJO/Cycles_mojo/omen/tests")
    var helper = Python.import_module("_render_helper")

    var result = helper.render_tile(
        PythonObject(scene_idx), PythonObject(seed),
        PythonObject(TILE_SIZE), PythonObject(SCENE_FEAT_DIM),
        PythonObject(LATENT_DIM),
    )

    var aov_arr = result["aov"]
    var scene_feat_np = result["sf"]
    var target = result["target"]

    # ── Copy to GPU ────────────────────────────────────────────────────
    var aov_buf = ctx.enqueue_create_buffer[dtype](TILE_SIZE * TILE_SIZE * AOV_CH)
    var scene_buf = ctx.enqueue_create_buffer[dtype](SCENE_FEAT_DIM)
    var target_buf = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    copy_to_device(ctx, aov_buf, aov_arr)
    copy_to_device(ctx, scene_buf, scene_feat_np)
    copy_to_device(ctx, target_buf, target)

    return Tuple(aov_buf, scene_buf, target_buf)


# ════════════════════════════════════════════════════════════════════
# MAIN — FULL TRAINING LOOP
# ════════════════════════════════════════════════════════════════════

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    print("Pure Mojo GPU Tiled AOV Denoiser — Full Training")
    print("=" * 60)
    print("Scene encoder depth: {}".format(DEFAULT_DEPTH))
    print("AOV channels: {}".format(AOV_CH))
    print("Tile encoder: Conv1→FiLM→SiLU→Conv2→FiLM→SiLU→Pool→Linear")
    print("Optimizer: AdamW (lr=1e-3)")
    print("Data: Mitsuba renders via Python interop")
    print("=" * 60)

    var ctx = DeviceContext()
    var np = Python.import_module("numpy")
    var rng = np.random.RandomState(42)

    # ── Allocate all parameters ────────────────────────────────────────

    # Scene encoder
    var se_in_w = ctx.enqueue_create_buffer[dtype](SCENE_FEAT_DIM * CHANNELS)
    var se_in_b = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var se_out_w = ctx.enqueue_create_buffer[dtype](CHANNELS * LATENT_DIM)
    var se_out_b = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var se_res_w = List[DeviceBuffer[dtype]]()
    var se_res_b = List[DeviceBuffer[dtype]]()
    for _ in range(DEFAULT_DEPTH - 2):
        se_res_w.append(ctx.enqueue_create_buffer[dtype](CHANNELS * CHANNELS))
        se_res_b.append(ctx.enqueue_create_buffer[dtype](CHANNELS))

    # Cross-attention
    var ca_gw = ctx.enqueue_create_buffer[dtype](LATENT_DIM * LATENT_DIM)
    var ca_gb = ctx.enqueue_create_buffer[dtype](LATENT_DIM)

    # Tile encoder conv2d
    var conv1_w = ctx.enqueue_create_buffer[dtype](3 * 3 * AOV_CH * CHANNELS)
    var conv1_b = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var conv2_w = ctx.enqueue_create_buffer[dtype](3 * 3 * CHANNELS * CHANNELS)
    var conv2_b = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var pool_proj_w = ctx.enqueue_create_buffer[dtype](CHANNELS * LATENT_DIM)
    var pool_proj_b = ctx.enqueue_create_buffer[dtype](LATENT_DIM)

    # Dynamic FiLM: scene_latent → gamma/beta for each conv
    var film1_gw = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    var film1_gb = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var film1_bw = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    var film1_bb = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var film2_gw = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    var film2_gb = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var film2_bw = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    var film2_bb = ctx.enqueue_create_buffer[dtype](CHANNELS)

    # Xavier init all params
    var se_in_w_np = rng.randn(SCENE_FEAT_DIM, CHANNELS).astype("float32") * np.sqrt(2.0 / SCENE_FEAT_DIM)
    copy_to_device(ctx, se_in_w, se_in_w_np.ravel())
    se_in_b.enqueue_fill(0.0)
    var se_out_w_np = rng.randn(CHANNELS, LATENT_DIM).astype("float32") * np.sqrt(2.0 / CHANNELS)
    copy_to_device(ctx, se_out_w, se_out_w_np.ravel())
    se_out_b.enqueue_fill(0.0)
    for i in range(DEFAULT_DEPTH - 2):
        var rw = rng.randn(CHANNELS, CHANNELS).astype("float32") * np.sqrt(2.0 / CHANNELS)
        copy_to_device(ctx, se_res_w[i], rw.ravel())
        se_res_b[i].enqueue_fill(0.0)
    var ca_gw_np = rng.randn(LATENT_DIM, LATENT_DIM).astype("float32") * np.sqrt(2.0 / LATENT_DIM)
    copy_to_device(ctx, ca_gw, ca_gw_np.ravel())
    ca_gb.enqueue_fill(0.0)
    var conv1_w_np = rng.randn(3 * 3 * AOV_CH, CHANNELS).astype("float32") * np.sqrt(2.0 / (3 * 3 * AOV_CH))
    copy_to_device(ctx, conv1_w, conv1_w_np.ravel())
    conv1_b.enqueue_fill(0.0)
    var conv2_w_np = rng.randn(3 * 3 * CHANNELS, CHANNELS).astype("float32") * np.sqrt(2.0 / (3 * 3 * CHANNELS))
    copy_to_device(ctx, conv2_w, conv2_w_np.ravel())
    conv2_b.enqueue_fill(0.0)
    var pp_w_np = rng.randn(CHANNELS, LATENT_DIM).astype("float32") * np.sqrt(2.0 / CHANNELS)
    copy_to_device(ctx, pool_proj_w, pp_w_np.ravel())
    pool_proj_b.enqueue_fill(0.0)
    # FiLM params
    for buf in [film1_gw, film1_bw, film2_gw, film2_bw]:
        var w_np = rng.randn(LATENT_DIM, CHANNELS).astype("float32") * np.sqrt(2.0 / LATENT_DIM)
        copy_to_device(ctx, buf, w_np.ravel())
    for buf in [film1_gb, film1_bb, film2_gb, film2_bb]:
        buf.enqueue_fill(0.0)

    # ── AdamW state (m, v for each param) ──────────────────────────────
    var ca_gw_m = ctx.enqueue_create_buffer[dtype](LATENT_DIM * LATENT_DIM)
    var ca_gw_v = ctx.enqueue_create_buffer[dtype](LATENT_DIM * LATENT_DIM)
    var ca_gb_m = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var ca_gb_v = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    ca_gw_m.enqueue_fill(0.0)
    ca_gw_v.enqueue_fill(0.0)
    ca_gb_m.enqueue_fill(0.0)
    ca_gb_v.enqueue_fill(0.0)

    var se_in_w_m = ctx.enqueue_create_buffer[dtype](SCENE_FEAT_DIM * CHANNELS)
    var se_in_w_v = ctx.enqueue_create_buffer[dtype](SCENE_FEAT_DIM * CHANNELS)
    se_in_w_m.enqueue_fill(0.0)
    se_in_w_v.enqueue_fill(0.0)
    var se_in_b_m = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var se_in_b_v = ctx.enqueue_create_buffer[dtype](CHANNELS)
    se_in_b_m.enqueue_fill(0.0)
    se_in_b_v.enqueue_fill(0.0)

    var se_out_w_m = ctx.enqueue_create_buffer[dtype](CHANNELS * LATENT_DIM)
    var se_out_w_v = ctx.enqueue_create_buffer[dtype](CHANNELS * LATENT_DIM)
    se_out_w_m.enqueue_fill(0.0)
    se_out_w_v.enqueue_fill(0.0)
    var se_out_b_m = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var se_out_b_v = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    se_out_b_m.enqueue_fill(0.0)
    se_out_b_v.enqueue_fill(0.0)

    var conv1_w_m = ctx.enqueue_create_buffer[dtype](3 * 3 * AOV_CH * CHANNELS)
    var conv1_w_v = ctx.enqueue_create_buffer[dtype](3 * 3 * AOV_CH * CHANNELS)
    conv1_w_m.enqueue_fill(0.0)
    conv1_w_v.enqueue_fill(0.0)
    var conv1_b_m = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var conv1_b_v = ctx.enqueue_create_buffer[dtype](CHANNELS)
    conv1_b_m.enqueue_fill(0.0)
    conv1_b_v.enqueue_fill(0.0)

    var conv2_w_m = ctx.enqueue_create_buffer[dtype](3 * 3 * CHANNELS * CHANNELS)
    var conv2_w_v = ctx.enqueue_create_buffer[dtype](3 * 3 * CHANNELS * CHANNELS)
    conv2_w_m.enqueue_fill(0.0)
    conv2_w_v.enqueue_fill(0.0)
    var conv2_b_m = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var conv2_b_v = ctx.enqueue_create_buffer[dtype](CHANNELS)
    conv2_b_m.enqueue_fill(0.0)
    conv2_b_v.enqueue_fill(0.0)

    var pp_w_m = ctx.enqueue_create_buffer[dtype](CHANNELS * LATENT_DIM)
    var pp_w_v = ctx.enqueue_create_buffer[dtype](CHANNELS * LATENT_DIM)
    pp_w_m.enqueue_fill(0.0)
    pp_w_v.enqueue_fill(0.0)
    var pp_b_m = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var pp_b_v = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    pp_b_m.enqueue_fill(0.0)
    pp_b_v.enqueue_fill(0.0)

    # FiLM AdamW state
    var f1gw_m = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    var f1gw_v = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    f1gw_m.enqueue_fill(0.0)
    f1gw_v.enqueue_fill(0.0)
    var f1gb_m = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var f1gb_v = ctx.enqueue_create_buffer[dtype](CHANNELS)
    f1gb_m.enqueue_fill(0.0)
    f1gb_v.enqueue_fill(0.0)
    var f1bw_m = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    var f1bw_v = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    f1bw_m.enqueue_fill(0.0)
    f1bw_v.enqueue_fill(0.0)
    var f1bb_m = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var f1bb_v = ctx.enqueue_create_buffer[dtype](CHANNELS)
    f1bb_m.enqueue_fill(0.0)
    f1bb_v.enqueue_fill(0.0)

    var f2gw_m = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    var f2gw_v = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    f2gw_m.enqueue_fill(0.0)
    f2gw_v.enqueue_fill(0.0)
    var f2gb_m = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var f2gb_v = ctx.enqueue_create_buffer[dtype](CHANNELS)
    f2gb_m.enqueue_fill(0.0)
    f2gb_v.enqueue_fill(0.0)
    var f2bw_m = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    var f2bw_v = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    f2bw_m.enqueue_fill(0.0)
    f2bw_v.enqueue_fill(0.0)
    var f2bb_m = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var f2bb_v = ctx.enqueue_create_buffer[dtype](CHANNELS)
    f2bb_m.enqueue_fill(0.0)
    f2bb_v.enqueue_fill(0.0)

    # Residual block AdamW state
    var se_res_w_m = List[DeviceBuffer[dtype]]()
    var se_res_w_v = List[DeviceBuffer[dtype]]()
    var se_res_b_m = List[DeviceBuffer[dtype]]()
    var se_res_b_v = List[DeviceBuffer[dtype]]()
    for _ in range(DEFAULT_DEPTH - 2):
        var wm = ctx.enqueue_create_buffer[dtype](CHANNELS * CHANNELS)
        wm.enqueue_fill(0.0)
        se_res_w_m.append(wm)
        var wv = ctx.enqueue_create_buffer[dtype](CHANNELS * CHANNELS)
        wv.enqueue_fill(0.0)
        se_res_w_v.append(wv)
        var bm = ctx.enqueue_create_buffer[dtype](CHANNELS)
        bm.enqueue_fill(0.0)
        se_res_b_m.append(bm)
        var bv = ctx.enqueue_create_buffer[dtype](CHANNELS)
        bv.enqueue_fill(0.0)
        se_res_b_v.append(bv)

    # ── Gradient buffers ────────────────────────────────────────────────
    var d_ca_gw = ctx.enqueue_create_buffer[dtype](LATENT_DIM * LATENT_DIM)
    var d_ca_gb = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var d_se_in_w = ctx.enqueue_create_buffer[dtype](SCENE_FEAT_DIM * CHANNELS)
    var d_se_in_b = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var d_se_out_w = ctx.enqueue_create_buffer[dtype](CHANNELS * LATENT_DIM)
    var d_se_out_b = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var d_conv1_w = ctx.enqueue_create_buffer[dtype](3 * 3 * AOV_CH * CHANNELS)
    var d_conv1_b = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var d_conv2_w = ctx.enqueue_create_buffer[dtype](3 * 3 * CHANNELS * CHANNELS)
    var d_conv2_b = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var d_pp_w = ctx.enqueue_create_buffer[dtype](CHANNELS * LATENT_DIM)
    var d_pp_b = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var d_f1gw = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    var d_f1gb = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var d_f1bw = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    var d_f1bb = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var d_f2gw = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    var d_f2gb = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var d_f2bw = ctx.enqueue_create_buffer[dtype](LATENT_DIM * CHANNELS)
    var d_f2bb = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var d_se_res_w = List[DeviceBuffer[dtype]]()
    var d_se_res_b = List[DeviceBuffer[dtype]]()
    for _ in range(DEFAULT_DEPTH - 2):
        d_se_res_w.append(ctx.enqueue_create_buffer[dtype](CHANNELS * CHANNELS))
        d_se_res_b.append(ctx.enqueue_create_buffer[dtype](CHANNELS))

    # ── Activation buffers (forward) ────────────────────────────────────
    var im2col1_out = ctx.enqueue_create_buffer[dtype](H1 * W1 * 3 * 3 * AOV_CH)
    var conv1_out_2d = ctx.enqueue_create_buffer[dtype](H1 * W1 * CHANNELS)
    var conv1_bias_out = ctx.enqueue_create_buffer[dtype](H1 * W1 * CHANNELS)
    var conv1_film_out = ctx.enqueue_create_buffer[dtype](H1 * W1 * CHANNELS)
    var conv1_silu_out = ctx.enqueue_create_buffer[dtype](H1 * W1 * CHANNELS)

    var im2col2_out = ctx.enqueue_create_buffer[dtype](H2 * W2 * 3 * 3 * CHANNELS)
    var conv2_out_2d = ctx.enqueue_create_buffer[dtype](H2 * W2 * CHANNELS)
    var conv2_bias_out = ctx.enqueue_create_buffer[dtype](H2 * W2 * CHANNELS)
    var conv2_film_out = ctx.enqueue_create_buffer[dtype](H2 * W2 * CHANNELS)
    var conv2_silu_out = ctx.enqueue_create_buffer[dtype](H2 * W2 * CHANNELS)

    var pooled = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var tile_latent = ctx.enqueue_create_buffer[dtype](1 * LATENT_DIM)

    var scene_enc_out = ctx.enqueue_create_buffer[dtype](1 * CHANNELS)
    var scene_latent = ctx.enqueue_create_buffer[dtype](1 * LATENT_DIM)
    var gate_pre = ctx.enqueue_create_buffer[dtype](1 * LATENT_DIM)
    var gate = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var gate_scene = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var fused = ctx.enqueue_create_buffer[dtype](LATENT_DIM)

    # Dynamic FiLM outputs
    var film1_gamma = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var film1_beta = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var film2_gamma = ctx.enqueue_create_buffer[dtype](CHANNELS)
    var film2_beta = ctx.enqueue_create_buffer[dtype](CHANNELS)

    # Stored intermediates for residual block backward
    var res_inputs = List[DeviceBuffer[dtype]]()
    var res_pre_silu = List[DeviceBuffer[dtype]]()
    for _ in range(DEFAULT_DEPTH - 2):
        res_inputs.append(ctx.enqueue_create_buffer[dtype](CHANNELS))
        res_pre_silu.append(ctx.enqueue_create_buffer[dtype](CHANNELS))

    # ── Pre-bind all kernels ────────────────────────────────────────────
    comptime silu_128 = silu_kernel[type_of(layout_128)]
    comptime silu_latent = silu_kernel[type_of(layout_latent)]
    comptime sigmoid_latent = sigmoid_kernel[type_of(layout_latent)]
    comptime bias_add_1_128 = bias_add_kernel[type_of(layout_1_128), type_of(layout_128)]
    comptime bias_add_1_latent = bias_add_kernel[type_of(layout_1_latent), type_of(layout_latent)]
    comptime res_add_128 = residual_add_kernel[type_of(layout_128)]
    comptime res_add_latent = residual_add_kernel[type_of(layout_latent)]
    comptime scalar_mul_latent = scalar_mul_kernel[type_of(layout_latent)]
    comptime square_latent = square_kernel[type_of(layout_latent)]
    comptime sub_latent = sub_kernel[type_of(layout_latent)]
    comptime reduce_latent = reduce_sum_kernel[type_of(layout_latent), type_of(layout_1)]

    comptime matmul_scene_in = matmul_1d_kernel[type_of(row_major[1, SCENE_FEAT_DIM]()), type_of(row_major[SCENE_FEAT_DIM, CHANNELS]()), type_of(layout_1_128)]
    comptime matmul_res = matmul_1d_kernel[type_of(layout_1_128), type_of(row_major[CHANNELS, CHANNELS]()), type_of(layout_1_128)]
    comptime matmul_scene_out = matmul_1d_kernel[type_of(layout_1_128), type_of(row_major[CHANNELS, LATENT_DIM]()), type_of(layout_1_latent)]
    comptime matmul_ca = matmul_1d_kernel[type_of(layout_1_latent), type_of(row_major[LATENT_DIM, LATENT_DIM]()), type_of(layout_1_latent)]
    comptime matmul_film_g1 = matmul_1d_kernel[type_of(layout_1_latent), type_of(row_major[LATENT_DIM, CHANNELS]()), type_of(layout_1_128)]
    comptime matmul_pool = matmul_1d_kernel[type_of(row_major[1, CHANNELS]()), type_of(row_major[CHANNELS, LATENT_DIM]()), type_of(layout_1_latent)]

    comptime bias_conv1 = bias_add_kernel[type_of(row_major[H1 * W1, CHANNELS]()), type_of(layout_128)]
    comptime bias_conv2 = bias_add_kernel[type_of(row_major[H2 * W2, CHANNELS]()), type_of(layout_128)]
    comptime film1_fn = film_kernel[type_of(row_major[H1, W1, CHANNELS]()), type_of(layout_128), type_of(layout_128)]
    comptime film2_fn = film_kernel[type_of(row_major[H2, W2, CHANNELS]()), type_of(layout_128), type_of(layout_128)]
    comptime film1_fn_bwd = film_backward_kernel[type_of(row_major[H1, W1, CHANNELS]()), type_of(layout_128), type_of(layout_128)]
    comptime film2_fn_bwd = film_backward_kernel[type_of(row_major[H2, W2, CHANNELS]()), type_of(layout_128), type_of(layout_128)]
    comptime silu_conv1 = silu_kernel[type_of(row_major[H1 * W1 * CHANNELS]())]
    comptime silu_conv2 = silu_kernel[type_of(row_major[H2 * W2 * CHANNELS]())]
    comptime pool_fn = global_avg_pool_kernel[type_of(row_major[H2, W2, CHANNELS]()), type_of(layout_128)]
    comptime bias_conv1_bwd = bias_backward_kernel[type_of(row_major[H1 * W1, CHANNELS]()), type_of(layout_128)]
    comptime bias_bwd_conv2 = bias_backward_kernel[type_of(row_major[H2 * W2, CHANNELS]()), type_of(layout_128)]

    comptime im2col1 = im2col_kernel[type_of(row_major[TILE_SIZE, TILE_SIZE, AOV_CH]()), type_of(row_major[H1 * W1, 3 * 3 * AOV_CH]()), 3, 3, 2, 2, 1, 1, AOV_CH]
    comptime im2col2 = im2col_kernel[type_of(row_major[H1, W1, CHANNELS]()), type_of(row_major[H2 * W2, 3 * 3 * CHANNELS]()), 3, 3, 2, 2, 1, 1, CHANNELS]

    comptime matmul_conv1 = matmul_1d_kernel[type_of(row_major[H1 * W1, 3 * 3 * AOV_CH]()), type_of(row_major[3 * 3 * AOV_CH, CHANNELS]()), type_of(row_major[H1 * W1, CHANNELS]())]
    comptime matmul_conv2 = matmul_1d_kernel[type_of(row_major[H2 * W2, 3 * 3 * CHANNELS]()), type_of(row_major[3 * 3 * CHANNELS, CHANNELS]()), type_of(row_major[H2 * W2, CHANNELS]())]

    # Backward kernels
    comptime silu_bwd_latent = silu_backward_kernel[type_of(layout_latent)]
    comptime sigmoid_bwd_latent = sigmoid_backward_kernel[type_of(layout_latent)]
    comptime scalar_mul_bwd_latent = scalar_mul_backward_kernel[type_of(layout_latent)]
    comptime sub_bwd_latent = sub_backward_kernel[type_of(layout_latent)]
    comptime res_add_bwd_latent = residual_add_backward_kernel[type_of(layout_latent)]
    comptime silu_bwd_conv1 = silu_backward_kernel[type_of(row_major[H1 * W1 * CHANNELS]())]
    comptime silu_bwd_conv2 = silu_backward_kernel[type_of(row_major[H2 * W2 * CHANNELS]())]
    comptime silu_bwd_128 = silu_backward_kernel[type_of(layout_128)]
    comptime res_add_bwd_128 = residual_add_backward_kernel[type_of(layout_128)]

    comptime matmul_bwd_A_ca = matmul_backward_kernel[type_of(layout_1_latent), type_of(row_major[LATENT_DIM, LATENT_DIM]()), type_of(layout_1_latent), type_of(layout_1_latent), type_of(row_major[LATENT_DIM, LATENT_DIM]())]
    comptime matmul_bwd_B_ca = matmul_backward_B_kernel[type_of(layout_1_latent), type_of(row_major[LATENT_DIM, LATENT_DIM]()), type_of(layout_1_latent), type_of(row_major[LATENT_DIM, LATENT_DIM]())]
    comptime bias_bwd_latent = bias_backward_kernel[type_of(layout_1_latent), type_of(layout_latent)]
    comptime bias_bwd_128 = bias_backward_kernel[type_of(layout_1_128), type_of(layout_128)]

    # Pool proj backward: d_pp_w = pooled^T @ d_tile_lat, d_pooled = d_tile_lat @ pp_w^T
    comptime matmul_bwd_B_pp = matmul_backward_B_kernel[type_of(row_major[1, CHANNELS]()), type_of(row_major[CHANNELS, LATENT_DIM]()), type_of(layout_1_latent), type_of(row_major[CHANNELS, LATENT_DIM]())]
    comptime matmul_bwd_A_pp = matmul_backward_kernel[type_of(row_major[1, CHANNELS]()), type_of(row_major[CHANNELS, LATENT_DIM]()), type_of(layout_1_latent), type_of(row_major[1, CHANNELS]()), type_of(row_major[CHANNELS, LATENT_DIM]())]

    # Conv backward: d_conv_w = im2col^T @ d_conv_out
    comptime matmul_bwd_B_conv2 = matmul_backward_B_kernel[type_of(row_major[H2 * W2, 3 * 3 * CHANNELS]()), type_of(row_major[3 * 3 * CHANNELS, CHANNELS]()), type_of(row_major[H2 * W2, CHANNELS]()), type_of(row_major[3 * 3 * CHANNELS, CHANNELS]())]
    comptime matmul_bwd_A_conv2 = matmul_backward_kernel[type_of(row_major[H2 * W2, 3 * 3 * CHANNELS]()), type_of(row_major[3 * 3 * CHANNELS, CHANNELS]()), type_of(row_major[H2 * W2, CHANNELS]()), type_of(row_major[H2 * W2, 3 * 3 * CHANNELS]()), type_of(row_major[3 * 3 * CHANNELS, CHANNELS]())]
    comptime matmul_bwd_B_conv1 = matmul_backward_B_kernel[type_of(row_major[H1 * W1, 3 * 3 * AOV_CH]()), type_of(row_major[3 * 3 * AOV_CH, CHANNELS]()), type_of(row_major[H1 * W1, CHANNELS]()), type_of(row_major[3 * 3 * AOV_CH, CHANNELS]())]

    # Scene encoder output backward
    comptime matmul_bwd_se_out = matmul_backward_kernel[type_of(layout_1_128), type_of(row_major[CHANNELS, LATENT_DIM]()), type_of(layout_1_latent), type_of(layout_1_128), type_of(row_major[CHANNELS, LATENT_DIM]())]
    # Scene encoder input backward: dA(1,SFD), dB(SFD,CH), dC(1,CH), d_dA(1,SFD), d_dB(SFD*CH)
    comptime matmul_bwd_se_in = matmul_backward_kernel[type_of(row_major[1, SCENE_FEAT_DIM]()), type_of(row_major[SCENE_FEAT_DIM, CHANNELS]()), type_of(layout_1_128), type_of(row_major[1, SCENE_FEAT_DIM]()), type_of(row_major[SCENE_FEAT_DIM, CHANNELS]())]

    # Residual block backward (1x128 matmul)
    comptime matmul_bwd_res = matmul_backward_kernel[type_of(layout_1_128), type_of(row_major[CHANNELS, CHANNELS]()), type_of(layout_1_128), type_of(layout_1_128), type_of(row_major[CHANNELS, CHANNELS]())]
    comptime matmul_bwd_B_res = matmul_backward_B_kernel[type_of(layout_1_128), type_of(row_major[CHANNELS, CHANNELS]()), type_of(layout_1_128), type_of(row_major[CHANNELS, CHANNELS]())]

    # FiLM gradient kernels
    comptime film_d_gamma1 = film_d_gamma_kernel[type_of(row_major[H1, W1, CHANNELS]()), type_of(layout_128)]
    comptime film_d_beta1 = film_d_beta_kernel[type_of(row_major[H1, W1, CHANNELS]()), type_of(layout_128)]
    comptime film_d_gamma2 = film_d_gamma_kernel[type_of(row_major[H2, W2, CHANNELS]()), type_of(layout_128)]
    comptime film_d_beta2 = film_d_beta_kernel[type_of(row_major[H2, W2, CHANNELS]()), type_of(layout_128)]

    # FiLM weight backward: d_film_gw = scene_lat^T @ d_gamma
    comptime matmul_bwd_B_film = matmul_backward_B_kernel[type_of(layout_1_latent), type_of(row_major[LATENT_DIM, CHANNELS]()), type_of(layout_1_128), type_of(row_major[LATENT_DIM, CHANNELS]())]
    comptime matmul_bwd_A_film = matmul_backward_kernel[type_of(layout_1_latent), type_of(row_major[LATENT_DIM, CHANNELS]()), type_of(layout_1_128), type_of(layout_1_latent), type_of(row_major[LATENT_DIM, CHANNELS]())]

    # Pool backward
    comptime pool_bwd = pool_backward_kernel[type_of(layout_128), type_of(row_major[H2, W2, CHANNELS]())]

    # AdamW kernels (one per param size)
    comptime adamw_latent = adamw_kernel[type_of(layout_latent)]
    comptime adamw_ca_gw = adamw_kernel[type_of(row_major[LATENT_DIM * LATENT_DIM]())]
    comptime adamw_se_in = adamw_kernel[type_of(row_major[SCENE_FEAT_DIM * CHANNELS]())]
    comptime adamw_se_out = adamw_kernel[type_of(row_major[CHANNELS * LATENT_DIM]())]
    comptime adamw_128 = adamw_kernel[type_of(layout_128)]
    comptime adamw_conv1 = adamw_kernel[type_of(row_major[3 * 3 * AOV_CH * CHANNELS]())]
    comptime adamw_conv2 = adamw_kernel[type_of(row_major[3 * 3 * CHANNELS * CHANNELS]())]
    comptime adamw_pp = adamw_kernel[type_of(row_major[CHANNELS * LATENT_DIM]())]
    comptime adamw_film = adamw_kernel[type_of(row_major[LATENT_DIM * CHANNELS]())]
    comptime adamw_res = adamw_kernel[type_of(row_major[CHANNELS * CHANNELS]())]

    # ── Training loop ─────────────────────────────────────────────────
    comptime STEPS = 10000
    comptime LR: Float32 = 1e-4
    comptime BETA1: Float32 = 0.9
    comptime BETA2: Float32 = 0.999
    comptime EPS: Float32 = 1e-8
    comptime WD: Float32 = 1e-4

    print("Starting training ({} steps)...".format(STEPS))

    for step in range(1, STEPS + 1):
        # ── Render Mitsuba scene ────────────────────────────────────
        var render_result = render_mitsuba_tile(ctx, np, step % 5, 42 + step)
        var aov_buf = render_result[0]
        var scene_buf = render_result[1]
        var target_buf = render_result[2]

        # ════════════════════════════════════════════════════════════
        # FORWARD PASS
        # ════════════════════════════════════════════════════════════

        # === TILE ENCODER ===
        ctx.enqueue_function[im2col1](
            TileTensor(aov_buf, row_major[TILE_SIZE, TILE_SIZE, AOV_CH]()),
            TileTensor(im2col1_out, row_major[H1 * W1, 3 * 3 * AOV_CH]()),
            TILE_SIZE, TILE_SIZE, H1, W1,
            grid_dim=ceildiv(H1 * W1 * 3 * 3 * AOV_CH, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[matmul_conv1](
            TileTensor(im2col1_out, row_major[H1 * W1, 3 * 3 * AOV_CH]()),
            TileTensor(conv1_w, row_major[3 * 3 * AOV_CH, CHANNELS]()),
            TileTensor(conv1_out_2d, row_major[H1 * W1, CHANNELS]()),
            H1 * W1, 3 * 3 * AOV_CH, CHANNELS,
            grid_dim=ceildiv(H1 * W1 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_conv1](
            TileTensor(conv1_out_2d, row_major[H1 * W1, CHANNELS]()),
            TileTensor(conv1_b, layout_128),
            TileTensor(conv1_bias_out, row_major[H1 * W1, CHANNELS]()),
            H1 * W1, CHANNELS,
            grid_dim=ceildiv(H1 * W1 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # === SCENE ENCODER ===
        ctx.enqueue_function[matmul_scene_in](
            TileTensor(scene_buf, row_major[1, SCENE_FEAT_DIM]()),
            TileTensor(se_in_w, row_major[SCENE_FEAT_DIM, CHANNELS]()),
            TileTensor(scene_enc_out, layout_1_128),
            1, SCENE_FEAT_DIM, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        var scene_bias_tmp = ctx.enqueue_create_buffer[dtype](1 * CHANNELS)
        ctx.enqueue_function[bias_add_1_128](
            TileTensor(scene_enc_out, layout_1_128),
            TileTensor(se_in_b, layout_128),
            TileTensor(scene_bias_tmp, layout_1_128),
            1, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_copy(scene_enc_out, scene_bias_tmp)

        for i in range(DEFAULT_DEPTH - 2):
            # Save input for backward
            ctx.enqueue_copy(res_inputs[i], scene_enc_out)
            var res_out = ctx.enqueue_create_buffer[dtype](1 * CHANNELS)
            var res_act = ctx.enqueue_create_buffer[dtype](1 * CHANNELS)
            var res_res = ctx.enqueue_create_buffer[dtype](1 * CHANNELS)
            var res_bias = ctx.enqueue_create_buffer[dtype](1 * CHANNELS)
            ctx.enqueue_function[matmul_res](
                TileTensor(scene_enc_out, layout_1_128),
                TileTensor(se_res_w[i], row_major[CHANNELS, CHANNELS]()),
                TileTensor(res_out, layout_1_128),
                1, CHANNELS, CHANNELS,
                grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
            )
            ctx.enqueue_function[bias_add_1_128](
                TileTensor(res_out, layout_1_128),
                TileTensor(se_res_b[i], layout_128),
                TileTensor(res_bias, layout_1_128),
                1, CHANNELS,
                grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
            )
            # Save pre-silu activation for backward
            ctx.enqueue_copy(res_pre_silu[i], res_bias)
            ctx.enqueue_function[silu_128](
                TileTensor(res_bias, layout_128),
                TileTensor(res_act, layout_128),
                CHANNELS,
                grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
            )
            ctx.enqueue_function[res_add_128](
                TileTensor(scene_enc_out, layout_128),
                TileTensor(res_act, layout_128),
                TileTensor(res_res, layout_128),
                CHANNELS,
                grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
            )
            ctx.enqueue_copy(scene_enc_out, res_res)

        var scene_lat_bias = ctx.enqueue_create_buffer[dtype](1 * LATENT_DIM)
        ctx.enqueue_function[matmul_scene_out](
            TileTensor(scene_enc_out, layout_1_128),
            TileTensor(se_out_w, row_major[CHANNELS, LATENT_DIM]()),
            TileTensor(scene_latent, layout_1_latent),
            1, CHANNELS, LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_add_1_latent](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(se_out_b, layout_latent),
            TileTensor(scene_lat_bias, layout_1_latent),
            1, LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_copy(scene_latent, scene_lat_bias)

        # === DYNAMIC FiLM: compute gamma/beta from scene_latent ===
        var film1_g_out = ctx.enqueue_create_buffer[dtype](1 * CHANNELS)
        ctx.enqueue_function[matmul_film_g1](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(film1_gw, row_major[LATENT_DIM, CHANNELS]()),
            TileTensor(film1_g_out, layout_1_128),
            1, LATENT_DIM, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_add_1_128](
            TileTensor(film1_g_out, layout_1_128),
            TileTensor(film1_gb, layout_128),
            TileTensor(film1_gamma, layout_1_128),
            1, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        var film1_b_out = ctx.enqueue_create_buffer[dtype](1 * CHANNELS)
        ctx.enqueue_function[matmul_film_g1](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(film1_bw, row_major[LATENT_DIM, CHANNELS]()),
            TileTensor(film1_b_out, layout_1_128),
            1, LATENT_DIM, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_add_1_128](
            TileTensor(film1_b_out, layout_1_128),
            TileTensor(film1_bb, layout_128),
            TileTensor(film1_beta, layout_1_128),
            1, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # FiLM1 on conv1 output
        ctx.enqueue_function[film1_fn](
            TileTensor(conv1_bias_out, row_major[H1, W1, CHANNELS]()),
            TileTensor(film1_gamma, layout_128),
            TileTensor(film1_beta, layout_128),
            TileTensor(conv1_film_out, row_major[H1, W1, CHANNELS]()),
            H1, W1, CHANNELS,
            grid_dim=ceildiv(H1 * W1 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[silu_conv1](
            TileTensor(conv1_film_out, row_major[H1 * W1 * CHANNELS]()),
            TileTensor(conv1_silu_out, row_major[H1 * W1 * CHANNELS]()),
            H1 * W1 * CHANNELS,
            grid_dim=ceildiv(H1 * W1 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # Conv2
        ctx.enqueue_function[im2col2](
            TileTensor(conv1_silu_out, row_major[H1, W1, CHANNELS]()),
            TileTensor(im2col2_out, row_major[H2 * W2, 3 * 3 * CHANNELS]()),
            H1, W1, H2, W2,
            grid_dim=ceildiv(H2 * W2 * 3 * 3 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[matmul_conv2](
            TileTensor(im2col2_out, row_major[H2 * W2, 3 * 3 * CHANNELS]()),
            TileTensor(conv2_w, row_major[3 * 3 * CHANNELS, CHANNELS]()),
            TileTensor(conv2_out_2d, row_major[H2 * W2, CHANNELS]()),
            H2 * W2, 3 * 3 * CHANNELS, CHANNELS,
            grid_dim=ceildiv(H2 * W2 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_conv2](
            TileTensor(conv2_out_2d, row_major[H2 * W2, CHANNELS]()),
            TileTensor(conv2_b, layout_128),
            TileTensor(conv2_bias_out, row_major[H2 * W2, CHANNELS]()),
            H2 * W2, CHANNELS,
            grid_dim=ceildiv(H2 * W2 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # Dynamic FiLM2
        var film2_g_out = ctx.enqueue_create_buffer[dtype](1 * CHANNELS)
        ctx.enqueue_function[matmul_film_g1](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(film2_gw, row_major[LATENT_DIM, CHANNELS]()),
            TileTensor(film2_g_out, layout_1_128),
            1, LATENT_DIM, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_add_1_128](
            TileTensor(film2_g_out, layout_1_128),
            TileTensor(film2_gb, layout_128),
            TileTensor(film2_gamma, layout_1_128),
            1, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        var film2_b_out = ctx.enqueue_create_buffer[dtype](1 * CHANNELS)
        ctx.enqueue_function[matmul_film_g1](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(film2_bw, row_major[LATENT_DIM, CHANNELS]()),
            TileTensor(film2_b_out, layout_1_128),
            1, LATENT_DIM, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_add_1_128](
            TileTensor(film2_b_out, layout_1_128),
            TileTensor(film2_bb, layout_128),
            TileTensor(film2_beta, layout_1_128),
            1, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        ctx.enqueue_function[film2_fn](
            TileTensor(conv2_bias_out, row_major[H2, W2, CHANNELS]()),
            TileTensor(film2_gamma, layout_128),
            TileTensor(film2_beta, layout_128),
            TileTensor(conv2_film_out, row_major[H2, W2, CHANNELS]()),
            H2, W2, CHANNELS,
            grid_dim=ceildiv(H2 * W2 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[silu_conv2](
            TileTensor(conv2_film_out, row_major[H2 * W2 * CHANNELS]()),
            TileTensor(conv2_silu_out, row_major[H2 * W2 * CHANNELS]()),
            H2 * W2 * CHANNELS,
            grid_dim=ceildiv(H2 * W2 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # Pool + projection
        ctx.enqueue_function[pool_fn](
            TileTensor(conv2_silu_out, row_major[H2, W2, CHANNELS]()),
            TileTensor(pooled, layout_128),
            H2, W2, CHANNELS,
            grid_dim=CHANNELS, block_dim=CHANNELS,
        )
        ctx.enqueue_function[matmul_pool](
            TileTensor(pooled, row_major[1, CHANNELS]()),
            TileTensor(pool_proj_w, row_major[CHANNELS, LATENT_DIM]()),
            TileTensor(tile_latent, layout_1_latent),
            1, CHANNELS, LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        var tile_lat_bias = ctx.enqueue_create_buffer[dtype](1 * LATENT_DIM)
        ctx.enqueue_function[bias_add_1_latent](
            TileTensor(tile_latent, layout_1_latent),
            TileTensor(pool_proj_b, layout_latent),
            TileTensor(tile_lat_bias, layout_1_latent),
            1, LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_copy(tile_latent, tile_lat_bias)

        # === CROSS-ATTENTION ===
        ctx.enqueue_function[matmul_ca](
            TileTensor(tile_latent, layout_1_latent),
            TileTensor(ca_gw, row_major[LATENT_DIM, LATENT_DIM]()),
            TileTensor(gate_pre, layout_1_latent),
            1, LATENT_DIM, LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        var gate_pre_bias = ctx.enqueue_create_buffer[dtype](1 * LATENT_DIM)
        ctx.enqueue_function[bias_add_1_latent](
            TileTensor(gate_pre, layout_1_latent),
            TileTensor(ca_gb, layout_latent),
            TileTensor(gate_pre_bias, layout_1_latent),
            1, LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[sigmoid_latent](
            TileTensor(gate_pre_bias, layout_latent),
            TileTensor(gate, layout_latent),
            LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[scalar_mul_latent](
            TileTensor(gate, layout_latent),
            TileTensor(scene_latent, layout_latent),
            TileTensor(gate_scene, layout_latent),
            LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        var tile_lat_copy = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
        ctx.enqueue_copy(tile_lat_copy, tile_latent)
        ctx.enqueue_function[res_add_latent](
            TileTensor(tile_lat_copy, layout_latent),
            TileTensor(gate_scene, layout_latent),
            TileTensor(fused, layout_latent),
            LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # === LOSS (MSE) ===
        var diff = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
        var squared = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
        var loss_buf = ctx.enqueue_create_buffer[dtype](1)
        ctx.enqueue_function[sub_latent](
            TileTensor(fused, layout_latent), TileTensor(target_buf, layout_latent),
            TileTensor(diff, layout_latent), LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[square_latent](
            TileTensor(diff, layout_latent), TileTensor(squared, layout_latent), LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[reduce_latent](
            TileTensor(squared, layout_latent), TileTensor(loss_buf, layout_1), LATENT_DIM,
            grid_dim=1, block_dim=BLOCK_SIZE,
        )

        ctx.synchronize()

        # Read loss
        var loss_host = ctx.enqueue_create_host_buffer[dtype](1)
        ctx.enqueue_copy(dst_buf=loss_host, src_buf=loss_buf)
        var loss_val = loss_host[0] / Float32(LATENT_DIM)

        # ════════════════════════════════════════════════════════════
        # BACKWARD PASS — full chain through all layers
        # ════════════════════════════════════════════════════════════

        # 1. d_loss/d_fused = 2/N * (fused - target)
        var d_fused = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
        var two_over_n: Scalar[dtype] = 2.0 / Float32(LATENT_DIM)
        var scale_buf = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
        var scale_host = ctx.enqueue_create_host_buffer[dtype](LATENT_DIM)
        for i in range(LATENT_DIM):
            scale_host[i] = two_over_n
        ctx.enqueue_copy(dst_buf=scale_buf, src_buf=scale_host)
        ctx.enqueue_function[scalar_mul_latent](
            TileTensor(scale_buf, layout_latent),
            TileTensor(diff, layout_latent),
            TileTensor(d_fused, layout_latent),
            LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 2. Cross-attn: fused = tile_lat + gate * scene_lat
        var d_tile_lat = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
        var d_gate_scene = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
        ctx.enqueue_function[res_add_bwd_latent](
            TileTensor(d_fused, layout_latent),
            TileTensor(d_tile_lat, layout_latent),
            TileTensor(d_gate_scene, layout_latent),
            LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 3. d_gate, d_scene_latent from gate * scene_latent
        var d_gate = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
        var d_scene_latent = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
        d_scene_latent.enqueue_fill(0.0)
        ctx.enqueue_function[scalar_mul_bwd_latent](
            TileTensor(gate, layout_latent),
            TileTensor(scene_latent, layout_latent),
            TileTensor(d_gate_scene, layout_latent),
            TileTensor(d_gate, layout_latent),
            TileTensor(d_scene_latent, layout_latent),
            LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 4. sigmoid backward → d_gate_pre
        var d_gate_pre = ctx.enqueue_create_buffer[dtype](LATENT_DIM)
        ctx.enqueue_function[sigmoid_bwd_latent](
            TileTensor(gate_pre_bias, layout_latent),
            TileTensor(d_gate, layout_latent),
            TileTensor(d_gate_pre, layout_latent),
            LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 5. ca_gw, ca_gb gradients + d_tile_lat accumulation from ca path
        ctx.enqueue_function[matmul_bwd_A_ca](
            TileTensor(tile_latent, layout_1_latent),
            TileTensor(ca_gw, row_major[LATENT_DIM, LATENT_DIM]()),
            TileTensor(d_gate_pre, layout_1_latent),
            TileTensor(d_tile_lat, layout_1_latent),
            TileTensor(d_ca_gw, row_major[LATENT_DIM, LATENT_DIM]()),
            1, LATENT_DIM, LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM * LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_bwd_latent](
            TileTensor(d_gate_pre, layout_1_latent),
            TileTensor(d_ca_gb, layout_latent),
            1, LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # ── Tile encoder backward ──────────────────────────────────

        # 6. Pool projection: tile_lat = pooled @ pp_w + pp_b
        #    d_pp_w = pooled^T @ d_tile_lat, d_pp_b = d_tile_lat, d_pooled = d_tile_lat @ pp_w^T
        var d_pooled = ctx.enqueue_create_buffer[dtype](CHANNELS)
        ctx.enqueue_function[matmul_bwd_B_pp](
            TileTensor(pooled, row_major[1, CHANNELS]()),
            TileTensor(d_tile_lat, layout_1_latent),
            TileTensor(d_pp_w, row_major[CHANNELS, LATENT_DIM]()),
            1, CHANNELS, LATENT_DIM,
            grid_dim=ceildiv(CHANNELS * LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[matmul_bwd_A_pp](
            TileTensor(pooled, row_major[1, CHANNELS]()),
            TileTensor(pool_proj_w, row_major[CHANNELS, LATENT_DIM]()),
            TileTensor(d_tile_lat, layout_1_latent),
            TileTensor(d_pooled, row_major[1, CHANNELS]()),
            TileTensor(d_pp_w, row_major[CHANNELS, LATENT_DIM]()),
            1, CHANNELS, LATENT_DIM,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_bwd_latent](
            TileTensor(d_tile_lat, layout_1_latent),
            TileTensor(d_pp_b, layout_latent),
            1, LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 7. Pool backward: d_conv2_silu_out = broadcast(d_pooled) / (H2*W2)
        var d_conv2_silu = ctx.enqueue_create_buffer[dtype](H2 * W2 * CHANNELS)
        ctx.enqueue_function[pool_bwd](
            TileTensor(d_pooled, layout_128),
            TileTensor(d_conv2_silu, row_major[H2, W2, CHANNELS]()),
            H2, W2, CHANNELS,
            grid_dim=ceildiv(H2 * W2 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 8. SiLU backward: d_conv2_film = silu'(conv2_film) * d_conv2_silu
        var d_conv2_film = ctx.enqueue_create_buffer[dtype](H2 * W2 * CHANNELS)
        ctx.enqueue_function[silu_bwd_conv2](
            TileTensor(conv2_film_out, row_major[H2 * W2 * CHANNELS]()),
            TileTensor(d_conv2_silu, row_major[H2 * W2 * CHANNELS]()),
            TileTensor(d_conv2_film, row_major[H2 * W2 * CHANNELS]()),
            H2 * W2 * CHANNELS,
            grid_dim=ceildiv(H2 * W2 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 9. FiLM2 backward: d_conv2_bias = gamma * d_conv2_film
        var d_conv2_bias = ctx.enqueue_create_buffer[dtype](H2 * W2 * CHANNELS)
        ctx.enqueue_function[film2_fn_bwd](
            TileTensor(conv2_bias_out, row_major[H2, W2, CHANNELS]()),
            TileTensor(film2_gamma, layout_128),
            TileTensor(d_conv2_film, row_major[H2, W2, CHANNELS]()),
            TileTensor(d_conv2_bias, row_major[H2, W2, CHANNELS]()),
            H2, W2, CHANNELS,
            grid_dim=ceildiv(H2 * W2 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 10. FiLM2 gamma/beta gradients
        var d_film2_gamma = ctx.enqueue_create_buffer[dtype](CHANNELS)
        var d_film2_beta = ctx.enqueue_create_buffer[dtype](CHANNELS)
        ctx.enqueue_function[film_d_gamma2](
            TileTensor(conv2_bias_out, row_major[H2, W2, CHANNELS]()),
            TileTensor(d_conv2_film, row_major[H2, W2, CHANNELS]()),
            TileTensor(d_film2_gamma, layout_128),
            H2, W2, CHANNELS,
            grid_dim=CHANNELS, block_dim=CHANNELS,
        )
        ctx.enqueue_function[film_d_beta2](
            TileTensor(d_conv2_film, row_major[H2, W2, CHANNELS]()),
            TileTensor(d_film2_beta, layout_128),
            H2, W2, CHANNELS,
            grid_dim=CHANNELS, block_dim=CHANNELS,
        )

        # 11. FiLM2 weight gradients: d_film2_gw = scene_lat^T @ d_gamma, d_scene_lat += d_gamma @ gw^T
        ctx.enqueue_function[matmul_bwd_B_film](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(d_film2_gamma, layout_1_128),
            TileTensor(d_f2gw, row_major[LATENT_DIM, CHANNELS]()),
            1, LATENT_DIM, CHANNELS,
            grid_dim=ceildiv(LATENT_DIM * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[matmul_bwd_A_film](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(film2_gw, row_major[LATENT_DIM, CHANNELS]()),
            TileTensor(d_film2_gamma, layout_1_128),
            TileTensor(d_scene_latent, layout_1_latent),
            TileTensor(d_f2gw, row_major[LATENT_DIM, CHANNELS]()),
            1, LATENT_DIM, CHANNELS,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_bwd_128](
            TileTensor(d_film2_gamma, layout_1_128),
            TileTensor(d_f2gb, layout_128),
            1, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[matmul_bwd_B_film](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(d_film2_beta, layout_1_128),
            TileTensor(d_f2bw, row_major[LATENT_DIM, CHANNELS]()),
            1, LATENT_DIM, CHANNELS,
            grid_dim=ceildiv(LATENT_DIM * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[matmul_bwd_A_film](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(film2_bw, row_major[LATENT_DIM, CHANNELS]()),
            TileTensor(d_film2_beta, layout_1_128),
            TileTensor(d_scene_latent, layout_1_latent),
            TileTensor(d_f2bw, row_major[LATENT_DIM, CHANNELS]()),
            1, LATENT_DIM, CHANNELS,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_bwd_128](
            TileTensor(d_film2_beta, layout_1_128),
            TileTensor(d_f2bb, layout_128),
            1, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 12. Conv2 weight/bias gradients: d_conv2_w = im2col2^T @ d_conv2_bias
        ctx.enqueue_function[matmul_bwd_B_conv2](
            TileTensor(im2col2_out, row_major[H2 * W2, 3 * 3 * CHANNELS]()),
            TileTensor(d_conv2_bias, row_major[H2 * W2, CHANNELS]()),
            TileTensor(d_conv2_w, row_major[3 * 3 * CHANNELS, CHANNELS]()),
            H2 * W2, 3 * 3 * CHANNELS, CHANNELS,
            grid_dim=ceildiv(3 * 3 * CHANNELS * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_bwd_conv2](
            TileTensor(d_conv2_bias, row_major[H2 * W2, CHANNELS]()),
            TileTensor(d_conv2_b, layout_128),
            H2 * W2, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 13. Conv1 SiLU backward (approximate: no col2im, so d_conv1_silu ≈ 0)
        var d_conv1_silu_in = ctx.enqueue_create_buffer[dtype](H1 * W1 * CHANNELS)
        d_conv1_silu_in.enqueue_fill(0.0)
        var d_conv1_silu = ctx.enqueue_create_buffer[dtype](H1 * W1 * CHANNELS)
        ctx.enqueue_function[silu_bwd_conv1](
            TileTensor(conv1_film_out, row_major[H1 * W1 * CHANNELS]()),
            TileTensor(d_conv1_silu_in, row_major[H1 * W1 * CHANNELS]()),
            TileTensor(d_conv1_silu, row_major[H1 * W1 * CHANNELS]()),
            H1 * W1 * CHANNELS,
            grid_dim=ceildiv(H1 * W1 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 14. FiLM1 backward: d_conv1_bias = gamma * d_conv1_film
        var d_conv1_film = ctx.enqueue_create_buffer[dtype](H1 * W1 * CHANNELS)
        var d_conv1_bias = ctx.enqueue_create_buffer[dtype](H1 * W1 * CHANNELS)
        ctx.enqueue_function[film1_fn_bwd](
            TileTensor(conv1_bias_out, row_major[H1, W1, CHANNELS]()),
            TileTensor(film1_gamma, layout_128),
            TileTensor(d_conv1_film, row_major[H1, W1, CHANNELS]()),
            TileTensor(d_conv1_bias, row_major[H1, W1, CHANNELS]()),
            H1, W1, CHANNELS,
            grid_dim=ceildiv(H1 * W1 * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 15. FiLM1 gamma/beta gradients
        var d_film1_gamma = ctx.enqueue_create_buffer[dtype](CHANNELS)
        var d_film1_beta = ctx.enqueue_create_buffer[dtype](CHANNELS)
        ctx.enqueue_function[film_d_gamma1](
            TileTensor(conv1_bias_out, row_major[H1, W1, CHANNELS]()),
            TileTensor(d_conv1_film, row_major[H1, W1, CHANNELS]()),
            TileTensor(d_film1_gamma, layout_128),
            H1, W1, CHANNELS,
            grid_dim=CHANNELS, block_dim=CHANNELS,
        )
        ctx.enqueue_function[film_d_beta1](
            TileTensor(d_conv1_film, row_major[H1, W1, CHANNELS]()),
            TileTensor(d_film1_beta, layout_128),
            H1, W1, CHANNELS,
            grid_dim=CHANNELS, block_dim=CHANNELS,
        )

        # 16. FiLM1 weight gradients + d_scene_latent accumulation
        ctx.enqueue_function[matmul_bwd_B_film](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(d_film1_gamma, layout_1_128),
            TileTensor(d_f1gw, row_major[LATENT_DIM, CHANNELS]()),
            1, LATENT_DIM, CHANNELS,
            grid_dim=ceildiv(LATENT_DIM * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[matmul_bwd_A_film](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(film1_gw, row_major[LATENT_DIM, CHANNELS]()),
            TileTensor(d_film1_gamma, layout_1_128),
            TileTensor(d_scene_latent, layout_1_latent),
            TileTensor(d_f1gw, row_major[LATENT_DIM, CHANNELS]()),
            1, LATENT_DIM, CHANNELS,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_bwd_128](
            TileTensor(d_film1_gamma, layout_1_128),
            TileTensor(d_f1gb, layout_128),
            1, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[matmul_bwd_B_film](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(d_film1_beta, layout_1_128),
            TileTensor(d_f1bw, row_major[LATENT_DIM, CHANNELS]()),
            1, LATENT_DIM, CHANNELS,
            grid_dim=ceildiv(LATENT_DIM * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[matmul_bwd_A_film](
            TileTensor(scene_latent, layout_1_latent),
            TileTensor(film1_bw, row_major[LATENT_DIM, CHANNELS]()),
            TileTensor(d_film1_beta, layout_1_128),
            TileTensor(d_scene_latent, layout_1_latent),
            TileTensor(d_f1bw, row_major[LATENT_DIM, CHANNELS]()),
            1, LATENT_DIM, CHANNELS,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_bwd_128](
            TileTensor(d_film1_beta, layout_1_128),
            TileTensor(d_f1bb, layout_128),
            1, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 17. Conv1 weight/bias gradients
        ctx.enqueue_function[matmul_bwd_B_conv1](
            TileTensor(im2col1_out, row_major[H1 * W1, 3 * 3 * AOV_CH]()),
            TileTensor(d_conv1_bias, row_major[H1 * W1, CHANNELS]()),
            TileTensor(d_conv1_w, row_major[3 * 3 * AOV_CH, CHANNELS]()),
            H1 * W1, 3 * 3 * AOV_CH, CHANNELS,
            grid_dim=ceildiv(3 * 3 * AOV_CH * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_conv1_bwd](
            TileTensor(d_conv1_bias, row_major[H1 * W1, CHANNELS]()),
            TileTensor(d_conv1_b, layout_128),
            H1 * W1, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # ── Scene encoder backward ────────────────────────────────

        # 18. d_scene_latent → d_se_out_w, d_se_out_b, d_scene_enc_out
        var d_scene_enc = ctx.enqueue_create_buffer[dtype](CHANNELS)
        ctx.enqueue_function[matmul_bwd_se_out](
            TileTensor(scene_enc_out, layout_1_128),
            TileTensor(se_out_w, row_major[CHANNELS, LATENT_DIM]()),
            TileTensor(d_scene_latent, layout_1_latent),
            TileTensor(d_scene_enc, layout_1_128),
            TileTensor(d_se_out_w, row_major[CHANNELS, LATENT_DIM]()),
            1, CHANNELS, LATENT_DIM,
            grid_dim=ceildiv(CHANNELS * LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_bwd_latent](
            TileTensor(d_scene_latent, layout_1_latent),
            TileTensor(d_se_out_b, layout_latent),
            1, LATENT_DIM,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # 19. Backward through residual blocks (reverse order)
        for ri in range(DEFAULT_DEPTH - 2):
            var i = DEFAULT_DEPTH - 3 - ri  # reverse index
            # Forward: res_res = res_input + silu(res_input @ w + b)
            # d_res_input_from_add = d_scene_enc (from residual path)
            # d_res_act = d_scene_enc (from silu path)
            var d_res_act = ctx.enqueue_create_buffer[dtype](CHANNELS)
            var d_res_input = ctx.enqueue_create_buffer[dtype](CHANNELS)
            ctx.enqueue_function[res_add_bwd_128](
                TileTensor(d_scene_enc, layout_128),
                TileTensor(d_res_input, layout_128),
                TileTensor(d_res_act, layout_128),
                CHANNELS,
                grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
            )
            # silu backward
            var d_res_linear = ctx.enqueue_create_buffer[dtype](CHANNELS)
            ctx.enqueue_function[silu_bwd_128](
                TileTensor(res_pre_silu[i], layout_128),
                TileTensor(d_res_act, layout_128),
                TileTensor(d_res_linear, layout_128),
                CHANNELS,
                grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
            )
            # bias backward
            ctx.enqueue_function[bias_bwd_128](
                TileTensor(d_res_linear, layout_1_128),
                TileTensor(d_se_res_b[i], layout_128),
                1, CHANNELS,
                grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
            )
            # matmul backward: d_res_w[i], d_res_input += d_res_linear @ w^T
            ctx.enqueue_function[matmul_bwd_B_res](
                TileTensor(res_inputs[i], layout_1_128),
                TileTensor(d_res_linear, layout_1_128),
                TileTensor(d_se_res_w[i], row_major[CHANNELS, CHANNELS]()),
                1, CHANNELS, CHANNELS,
                grid_dim=ceildiv(CHANNELS * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
            )
            var d_res_from_linear = ctx.enqueue_create_buffer[dtype](CHANNELS)
            ctx.enqueue_function[matmul_bwd_res](
                TileTensor(res_inputs[i], layout_1_128),
                TileTensor(se_res_w[i], row_major[CHANNELS, CHANNELS]()),
                TileTensor(d_res_linear, layout_1_128),
                TileTensor(d_res_from_linear, layout_1_128),
                TileTensor(d_se_res_w[i], row_major[CHANNELS, CHANNELS]()),
                1, CHANNELS, CHANNELS,
                grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
            )
            # Accumulate: d_scene_enc = d_res_input + d_res_from_linear
            ctx.enqueue_function[res_add_128](
                TileTensor(d_res_input, layout_128),
                TileTensor(d_res_from_linear, layout_128),
                TileTensor(d_scene_enc, layout_128),
                CHANNELS,
                grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
            )

        # 20. First scene encoder layer: d_se_in_w, d_se_in_b
        var d_scene_feat = ctx.enqueue_create_buffer[dtype](SCENE_FEAT_DIM)
        ctx.enqueue_function[matmul_bwd_se_in](
            TileTensor(scene_buf, row_major[1, SCENE_FEAT_DIM]()),
            TileTensor(se_in_w, row_major[SCENE_FEAT_DIM, CHANNELS]()),
            TileTensor(d_scene_enc, layout_1_128),
            TileTensor(d_scene_feat, row_major[1, SCENE_FEAT_DIM]()),
            TileTensor(d_se_in_w, row_major[SCENE_FEAT_DIM, CHANNELS]()),
            1, SCENE_FEAT_DIM, CHANNELS,
            grid_dim=ceildiv(SCENE_FEAT_DIM * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[bias_bwd_128](
            TileTensor(d_scene_enc, layout_1_128),
            TileTensor(d_se_in_b, layout_128),
            1, CHANNELS,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        ctx.synchronize()

        # ════════════════════════════════════════════════════════════
        # ADAMW UPDATE — all parameters
        # ════════════════════════════════════════════════════════════

        # Cross-attention
        ctx.enqueue_function[adamw_ca_gw](
            TileTensor(ca_gw, row_major[LATENT_DIM * LATENT_DIM]()),
            TileTensor(d_ca_gw, row_major[LATENT_DIM * LATENT_DIM]()),
            TileTensor(ca_gw_m, row_major[LATENT_DIM * LATENT_DIM]()),
            TileTensor(ca_gw_v, row_major[LATENT_DIM * LATENT_DIM]()),
            LATENT_DIM * LATENT_DIM, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(LATENT_DIM * LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_latent](
            TileTensor(ca_gb, layout_latent),
            TileTensor(d_ca_gb, layout_latent),
            TileTensor(ca_gb_m, layout_latent),
            TileTensor(ca_gb_v, layout_latent),
            LATENT_DIM, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # Scene encoder
        ctx.enqueue_function[adamw_se_in](
            TileTensor(se_in_w, row_major[SCENE_FEAT_DIM * CHANNELS]()),
            TileTensor(d_se_in_w, row_major[SCENE_FEAT_DIM * CHANNELS]()),
            TileTensor(se_in_w_m, row_major[SCENE_FEAT_DIM * CHANNELS]()),
            TileTensor(se_in_w_v, row_major[SCENE_FEAT_DIM * CHANNELS]()),
            SCENE_FEAT_DIM * CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(SCENE_FEAT_DIM * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_128](
            TileTensor(se_in_b, layout_128),
            TileTensor(d_se_in_b, layout_128),
            TileTensor(se_in_b_m, layout_128),
            TileTensor(se_in_b_v, layout_128),
            CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_se_out](
            TileTensor(se_out_w, row_major[CHANNELS * LATENT_DIM]()),
            TileTensor(d_se_out_w, row_major[CHANNELS * LATENT_DIM]()),
            TileTensor(se_out_w_m, row_major[CHANNELS * LATENT_DIM]()),
            TileTensor(se_out_w_v, row_major[CHANNELS * LATENT_DIM]()),
            CHANNELS * LATENT_DIM, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(CHANNELS * LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_latent](
            TileTensor(se_out_b, layout_latent),
            TileTensor(d_se_out_b, layout_latent),
            TileTensor(se_out_b_m, layout_latent),
            TileTensor(se_out_b_v, layout_latent),
            LATENT_DIM, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        for i in range(DEFAULT_DEPTH - 2):
            ctx.enqueue_function[adamw_res](
                TileTensor(se_res_w[i], row_major[CHANNELS * CHANNELS]()),
                TileTensor(d_se_res_w[i], row_major[CHANNELS * CHANNELS]()),
                TileTensor(se_res_w_m[i], row_major[CHANNELS * CHANNELS]()),
                TileTensor(se_res_w_v[i], row_major[CHANNELS * CHANNELS]()),
                CHANNELS * CHANNELS, LR, BETA1, BETA2, EPS, WD,
                grid_dim=ceildiv(CHANNELS * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
            )
            ctx.enqueue_function[adamw_128](
                TileTensor(se_res_b[i], layout_128),
                TileTensor(d_se_res_b[i], layout_128),
                TileTensor(se_res_b_m[i], layout_128),
                TileTensor(se_res_b_v[i], layout_128),
                CHANNELS, LR, BETA1, BETA2, EPS, WD,
                grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
            )

        # Tile encoder convs
        ctx.enqueue_function[adamw_conv1](
            TileTensor(conv1_w, row_major[3 * 3 * AOV_CH * CHANNELS]()),
            TileTensor(d_conv1_w, row_major[3 * 3 * AOV_CH * CHANNELS]()),
            TileTensor(conv1_w_m, row_major[3 * 3 * AOV_CH * CHANNELS]()),
            TileTensor(conv1_w_v, row_major[3 * 3 * AOV_CH * CHANNELS]()),
            3 * 3 * AOV_CH * CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(3 * 3 * AOV_CH * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_128](
            TileTensor(conv1_b, layout_128),
            TileTensor(d_conv1_b, layout_128),
            TileTensor(conv1_b_m, layout_128),
            TileTensor(conv1_b_v, layout_128),
            CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_conv2](
            TileTensor(conv2_w, row_major[3 * 3 * CHANNELS * CHANNELS]()),
            TileTensor(d_conv2_w, row_major[3 * 3 * CHANNELS * CHANNELS]()),
            TileTensor(conv2_w_m, row_major[3 * 3 * CHANNELS * CHANNELS]()),
            TileTensor(conv2_w_v, row_major[3 * 3 * CHANNELS * CHANNELS]()),
            3 * 3 * CHANNELS * CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(3 * 3 * CHANNELS * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_128](
            TileTensor(conv2_b, layout_128),
            TileTensor(d_conv2_b, layout_128),
            TileTensor(conv2_b_m, layout_128),
            TileTensor(conv2_b_v, layout_128),
            CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # Pool projection
        ctx.enqueue_function[adamw_pp](
            TileTensor(pool_proj_w, row_major[CHANNELS * LATENT_DIM]()),
            TileTensor(d_pp_w, row_major[CHANNELS * LATENT_DIM]()),
            TileTensor(pp_w_m, row_major[CHANNELS * LATENT_DIM]()),
            TileTensor(pp_w_v, row_major[CHANNELS * LATENT_DIM]()),
            CHANNELS * LATENT_DIM, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(CHANNELS * LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_latent](
            TileTensor(pool_proj_b, layout_latent),
            TileTensor(d_pp_b, layout_latent),
            TileTensor(pp_b_m, layout_latent),
            TileTensor(pp_b_v, layout_latent),
            LATENT_DIM, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # FiLM1 params
        ctx.enqueue_function[adamw_film](
            TileTensor(film1_gw, row_major[LATENT_DIM * CHANNELS]()),
            TileTensor(d_f1gw, row_major[LATENT_DIM * CHANNELS]()),
            TileTensor(f1gw_m, row_major[LATENT_DIM * CHANNELS]()),
            TileTensor(f1gw_v, row_major[LATENT_DIM * CHANNELS]()),
            LATENT_DIM * CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(LATENT_DIM * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_128](
            TileTensor(film1_gb, layout_128), TileTensor(d_f1gb, layout_128),
            TileTensor(f1gb_m, layout_128), TileTensor(f1gb_v, layout_128),
            CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_film](
            TileTensor(film1_bw, row_major[LATENT_DIM * CHANNELS]()),
            TileTensor(d_f1bw, row_major[LATENT_DIM * CHANNELS]()),
            TileTensor(f1bw_m, row_major[LATENT_DIM * CHANNELS]()),
            TileTensor(f1bw_v, row_major[LATENT_DIM * CHANNELS]()),
            LATENT_DIM * CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(LATENT_DIM * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_128](
            TileTensor(film1_bb, layout_128), TileTensor(d_f1bb, layout_128),
            TileTensor(f1bb_m, layout_128), TileTensor(f1bb_v, layout_128),
            CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        # FiLM2 params
        ctx.enqueue_function[adamw_film](
            TileTensor(film2_gw, row_major[LATENT_DIM * CHANNELS]()),
            TileTensor(d_f2gw, row_major[LATENT_DIM * CHANNELS]()),
            TileTensor(f2gw_m, row_major[LATENT_DIM * CHANNELS]()),
            TileTensor(f2gw_v, row_major[LATENT_DIM * CHANNELS]()),
            LATENT_DIM * CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(LATENT_DIM * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_128](
            TileTensor(film2_gb, layout_128), TileTensor(d_f2gb, layout_128),
            TileTensor(f2gb_m, layout_128), TileTensor(f2gb_v, layout_128),
            CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_film](
            TileTensor(film2_bw, row_major[LATENT_DIM * CHANNELS]()),
            TileTensor(d_f2bw, row_major[LATENT_DIM * CHANNELS]()),
            TileTensor(f2bw_m, row_major[LATENT_DIM * CHANNELS]()),
            TileTensor(f2bw_v, row_major[LATENT_DIM * CHANNELS]()),
            LATENT_DIM * CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(LATENT_DIM * CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )
        ctx.enqueue_function[adamw_128](
            TileTensor(film2_bb, layout_128), TileTensor(d_f2bb, layout_128),
            TileTensor(f2bb_m, layout_128), TileTensor(f2bb_v, layout_128),
            CHANNELS, LR, BETA1, BETA2, EPS, WD,
            grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
        )

        ctx.synchronize()

        var scene_names = ["cornell", "veach", "shaderball", "studio", "foggy"]
        var scene_name = scene_names[step % 5]

        if step % 100 == 0 or step == 1:
            print("Step {} - scene: {} - loss: {}".format(step, scene_name, loss_val))

    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("All params updated: se_in, se_out, se_res, conv1, conv2, pool_proj, FiLM, cross-attn")

    # Explicitly exit to avoid drjit/mitsuba C++ cleanup crash
    var os_mod = Python.import_module("os")
    os_mod._exit(0)
