"""Mojo GPU AOV Denoiser - Python Extension Module.

Simple training interface that manages internal state.
"""

from std.math import ceildiv, exp
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.sync import barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.python import Python, PythonObject
from std.atomic import Atomic
from layout import TileTensor, TensorLayout, row_major

# ── Comptime constants ────────────────────────────────────────────
comptime dtype = DType.float32
comptime SCENE_FEAT_DIM = 18
comptime LATENT_DIM = 128
comptime CHANNELS = 128
comptime BLOCK_SIZE = 256

# ── Layouts ─────────────────────────────────────────────────────────
comptime layout_128 = row_major[CHANNELS]()
comptime layout_latent = row_major[LATENT_DIM]()
comptime layout_1_128 = row_major[1, CHANNELS]()
comptime layout_1_latent = row_major[1, LATENT_DIM]()
comptime layout_1 = row_major[1]()

# ── Kernels ─────────────────────────────────────────────────────────

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
    comptime assert input.flat_rank == 1
    var tid = global_idx.x
    if tid == 0:
        output[0] = rebind[output.ElementType](0.0)
    barrier()
    if tid < size:
        var val = rebind[Scalar[dtype]](input[tid])
        _ = Atomic.fetch_add(output.data, val)
    barrier()


def matmul_1d_kernel[LT1: TensorLayout, LT2: TensorLayout, LT3: TensorLayout](
    A: TileTensor[dtype, LT1, MutAnyOrigin],
    B: TileTensor[dtype, LT2, MutAnyOrigin],
    C: TileTensor[dtype, LT3, MutAnyOrigin],
    M: Int, K: Int, N: Int,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2
    var tid = global_idx.x
    if tid < M * N:
        var row = tid // N
        var col = tid % N
        var acc: C.ElementType = 0.0
        for k in range(K):
            var a_val = rebind[Scalar[dtype]](A[row, k])
            var b_val = rebind[Scalar[dtype]](B[k, col])
            acc += a_val * b_val
        C[row, col] = acc


# ── State holder (passed as PythonObject) ────────────────────────────

struct DenoiserState:
    var ctx: DeviceContext
    var se_in_w: DeviceBuffer[dtype]
    var se_in_b: DeviceBuffer[dtype]
    var se_out_w: DeviceBuffer[dtype]
    var se_out_b: DeviceBuffer[dtype]
    var ca_gw: DeviceBuffer[dtype]
    var ca_gb: DeviceBuffer[dtype]
    var depth: Int

    def __init__(out self, depth: Int, np: PythonObject) raises:
        comptime assert has_accelerator(), "Requires GPU"
        self.ctx = DeviceContext()
        self.depth = depth

        # Allocate buffers
        self.se_in_w = self.ctx.enqueue_create_buffer[dtype](SCENE_FEAT_DIM * CHANNELS)
        self.se_in_b = self.ctx.enqueue_create_buffer[dtype](CHANNELS)
        self.se_out_w = self.ctx.enqueue_create_buffer[dtype](CHANNELS * LATENT_DIM)
        self.se_out_b = self.ctx.enqueue_create_buffer[dtype](LATENT_DIM)
        self.ca_gw = self.ctx.enqueue_create_buffer[dtype](LATENT_DIM * LATENT_DIM)
        self.ca_gb = self.ctx.enqueue_create_buffer[dtype](LATENT_DIM)

        # Xavier init
        var rng = np.random.RandomState(42)

        var se_in_w_np = rng.randn(SCENE_FEAT_DIM, CHANNELS).astype("float32") * np.sqrt(2.0 / SCENE_FEAT_DIM)
        self._copy_to_device(self.se_in_w, se_in_w_np.ravel())
        self.se_in_b.enqueue_fill(0.0)

        var se_out_w_np = rng.randn(CHANNELS, LATENT_DIM).astype("float32") * np.sqrt(2.0 / CHANNELS)
        self._copy_to_device(self.se_out_w, se_out_w_np.ravel())
        self.se_out_b.enqueue_fill(0.0)

        var ca_gw_np = rng.randn(LATENT_DIM, LATENT_DIM).astype("float32") * np.sqrt(2.0 / LATENT_DIM)
        self._copy_to_device(self.ca_gw, ca_gw_np.ravel())
        self.ca_gb.enqueue_fill(0.0)

    fn _copy_to_device(mut self, buf: DeviceBuffer[dtype], np_array: PythonObject) raises:
        var arr_len = len(np_array)
        var host_buf = self.ctx.enqueue_create_host_buffer[dtype](arr_len)
        for i in range(arr_len):
            var val = Float32(py=np_array[i])
            host_buf[i] = val
        self.ctx.enqueue_copy(dst_buf=buf, src_buf=host_buf)

    fn _copy_from_device(mut self, buf: DeviceBuffer[dtype], size: Int) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var host_buf = self.ctx.enqueue_create_host_buffer[dtype](size)
        self.ctx.enqueue_copy(dst_buf=host_buf, src_buf=buf)
        var result = np.zeros(size, dtype="float32")
        for i in range(size):
            result[i] = PythonObject(host_buf[i])
        return result


# ── Python exports ──────────────────────────────────────────────────

@export
def create(depth: Int = 8) raises -> PythonObject:
    """Create and return a DenoiserState."""
    var np = Python.import_module("numpy")
    var state = DenoiserState(depth, np)
    return PythonObject(alloc=state^)


@export
def train_step(
    state_py: PythonObject,
    scene_feat: PythonObject,
    target_latent: PythonObject,
    tile_latent: PythonObject,
) raises -> PythonObject:
    """Run one training step."""
    var state_ptr = state_py.downcast_value_ptr[DenoiserState]()

    # Copy inputs
    var scene_buf = state_ptr.ctx.enqueue_create_buffer[dtype](SCENE_FEAT_DIM)
    var target_buf = state_ptr.ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var tile_buf = state_ptr.ctx.enqueue_create_buffer[dtype](LATENT_DIM)

    state_ptr._copy_to_device(scene_buf, scene_feat)
    state_ptr._copy_to_device(target_buf, target_latent)
    state_ptr._copy_to_device(tile_buf, tile_latent)

    # Simplified: scene latent = linear(scene_feat) @ linear_out + bias
    var scene_hid = state_ptr.ctx.enqueue_create_buffer[dtype](1 * CHANNELS)
    var scene_hid_bias = state_ptr.ctx.enqueue_create_buffer[dtype](1 * CHANNELS)
    var scene_latent = state_ptr.ctx.enqueue_create_buffer[dtype](1 * LATENT_DIM)
    var scene_latent_bias = state_ptr.ctx.enqueue_create_buffer[dtype](1 * LATENT_DIM)

    comptime matmul_in = matmul_1d_kernel[
        type_of(row_major[1, SCENE_FEAT_DIM]()),
        type_of(row_major[SCENE_FEAT_DIM, CHANNELS]()),
        type_of(layout_1_128)
    ]
    comptime matmul_out = matmul_1d_kernel[
        type_of(layout_1_128),
        type_of(row_major[CHANNELS, LATENT_DIM]()),
        type_of(layout_1_latent)
    ]
    comptime bias_128 = bias_add_kernel[type_of(layout_1_128), type_of(layout_128)]
    comptime bias_latent = bias_add_kernel[type_of(layout_1_latent), type_of(layout_latent)]
    comptime silu = silu_kernel[type_of(layout_128)]
    comptime sigmoid = sigmoid_kernel[type_of(layout_latent)]
    comptime scalar_mul = scalar_mul_kernel[type_of(layout_latent)]
    comptime res_add = residual_add_kernel[type_of(layout_latent)]
    comptime square = square_kernel[type_of(layout_latent)]
    comptime sub = sub_kernel[type_of(layout_latent)]
    comptime reduce = reduce_sum_kernel[type_of(layout_latent), type_of(layout_1)]
    comptime matmul_ca = matmul_1d_kernel[
        type_of(layout_1_latent),
        type_of(row_major[LATENT_DIM, LATENT_DIM]()),
        type_of(layout_1_latent)
    ]

    # Scene encoder (simplified, no residual blocks for now)
    state_ptr.ctx.enqueue_function[matmul_in](
        TileTensor(scene_buf, row_major[1, SCENE_FEAT_DIM]()),
        TileTensor(state_ptr.se_in_w, row_major[SCENE_FEAT_DIM, CHANNELS]()),
        TileTensor(scene_hid, layout_1_128),
        1, SCENE_FEAT_DIM, CHANNELS,
        grid_dim=1, block_dim=(1, 16),
    )
    state_ptr.ctx.enqueue_function[bias_128](
        TileTensor(scene_hid, layout_1_128),
        TileTensor(state_ptr.se_in_b, layout_128),
        TileTensor(scene_hid_bias, layout_1_128),
        1, CHANNELS,
        grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
    )
    state_ptr.ctx.enqueue_function[silu](
        TileTensor(scene_hid_bias, layout_128),
        TileTensor(scene_hid, layout_128),
        CHANNELS,
        grid_dim=ceildiv(CHANNELS, BLOCK_SIZE), block_dim=BLOCK_SIZE,
    )
    state_ptr.ctx.enqueue_function[matmul_out](
        TileTensor(scene_hid, layout_1_128),
        TileTensor(state_ptr.se_out_w, row_major[CHANNELS, LATENT_DIM]()),
        TileTensor(scene_latent, layout_1_latent),
        1, CHANNELS, LATENT_DIM,
        grid_dim=1, block_dim=(1, 16),
    )
    state_ptr.ctx.enqueue_function[bias_latent](
        TileTensor(scene_latent, layout_1_latent),
        TileTensor(state_ptr.se_out_b, layout_latent),
        TileTensor(scene_latent_bias, layout_1_latent),
        1, LATENT_DIM,
        grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
    )

    # Gate
    var gate_pre = state_ptr.ctx.enqueue_create_buffer[dtype](1 * LATENT_DIM)
    var gate_pre_bias = state_ptr.ctx.enqueue_create_buffer[dtype](1 * LATENT_DIM)
    var gate = state_ptr.ctx.enqueue_create_buffer[dtype](LATENT_DIM)

    state_ptr.ctx.enqueue_function[matmul_ca](
        TileTensor(tile_buf, layout_1_latent),
        TileTensor(state_ptr.ca_gw, row_major[LATENT_DIM, LATENT_DIM]()),
        TileTensor(gate_pre, layout_1_latent),
        1, LATENT_DIM, LATENT_DIM,
        grid_dim=1, block_dim=(1, 16),
    )
    state_ptr.ctx.enqueue_function[bias_latent](
        TileTensor(gate_pre, layout_1_latent),
        TileTensor(state_ptr.ca_gb, layout_latent),
        TileTensor(gate_pre_bias, layout_1_latent),
        1, LATENT_DIM,
        grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
    )
    state_ptr.ctx.enqueue_function[sigmoid](
        TileTensor(gate_pre_bias, layout_latent),
        TileTensor(gate, layout_latent),
        LATENT_DIM,
        grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
    )

    # Fusion
    var gate_scene = state_ptr.ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var tile_copy = state_ptr.ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    state_ptr.ctx.enqueue_copy(tile_copy, tile_buf)
    var fused = state_ptr.ctx.enqueue_create_buffer[dtype](LATENT_DIM)

    state_ptr.ctx.enqueue_function[scalar_mul](
        TileTensor(gate, layout_latent),
        TileTensor(scene_latent_bias, layout_latent),
        TileTensor(gate_scene, layout_latent),
        LATENT_DIM,
        grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
    )
    state_ptr.ctx.enqueue_function[res_add](
        TileTensor(tile_copy, layout_latent),
        TileTensor(gate_scene, layout_latent),
        TileTensor(fused, layout_latent),
        LATENT_DIM,
        grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
    )

    # Loss
    var diff = state_ptr.ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var squared = state_ptr.ctx.enqueue_create_buffer[dtype](LATENT_DIM)
    var loss_buf = state_ptr.ctx.enqueue_create_buffer[dtype](1)

    state_ptr.ctx.enqueue_function[sub](
        TileTensor(fused, layout_latent),
        TileTensor(target_buf, layout_latent),
        TileTensor(diff, layout_latent),
        LATENT_DIM,
        grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
    )
    state_ptr.ctx.enqueue_function[square](
        TileTensor(diff, layout_latent),
        TileTensor(squared, layout_latent),
        LATENT_DIM,
        grid_dim=ceildiv(LATENT_DIM, BLOCK_SIZE), block_dim=BLOCK_SIZE,
    )
    state_ptr.ctx.enqueue_function[reduce](
        TileTensor(squared, layout_latent),
        TileTensor(loss_buf, layout_1),
        LATENT_DIM,
        grid_dim=1, block_dim=BLOCK_SIZE,
    )

    state_ptr.ctx.synchronize()

    var loss_host_buf = state_ptr.ctx.enqueue_create_host_buffer[dtype](1)
    state_ptr.ctx.enqueue_copy(dst_buf=loss_host_buf, src_buf=loss_buf)
    var loss_val = loss_host_buf[0] / Float32(LATENT_DIM)

    var scene_latent_np = state_ptr._copy_from_device(scene_latent_bias, LATENT_DIM)
    var fused_np = state_ptr._copy_from_device(fused, LATENT_DIM)

    return PythonObject({
        "loss": PythonObject(loss_val),
        "scene_latent": scene_latent_np,
        "fused": fused_np,
    })
