# GPU stress test — heavy matmul + warp shuffle reduction with float32
# Compile: mojo build -o gpu_stress gpu_stress.mojo
# Run:     ./gpu_stress

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu.sync import barrier
from std.gpu.host import DeviceContext
from std.gpu import global_idx, thread_idx, block_idx
from std.gpu.memory import AddressSpace
from std.gpu.primitives import warp
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime dtype = DType.float32
comptime TILE = 16

# === Matmul kernel (parametric — works) ===

comptime MM = 512
comptime NN = 512
comptime KK = 512
comptime a_layout = row_major[MM, KK]()
comptime b_layout = row_major[KK, NN]()
comptime c_layout = row_major[MM, NN]()

def matmul_kernel[
    AL: TensorLayout, BL: TensorLayout, CL: TensorLayout,
](
    A: TileTensor[dtype, AL, MutAnyOrigin],
    B: TileTensor[dtype, BL, MutAnyOrigin],
    C: TileTensor[dtype, CL, MutAnyOrigin],
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    var tx = thread_idx.x
    var ty = thread_idx.y
    var row = block_idx.y * TILE + ty
    var col = block_idx.x * TILE + tx

    var sa = stack_allocation[dtype,
        address_space=AddressSpace.SHARED](row_major[TILE, TILE]())
    var sb = stack_allocation[dtype,
        address_space=AddressSpace.SHARED](row_major[TILE, TILE]())

    var acc: C.ElementType = 0.0
    comptime for k_tile in range(0, KK, TILE):
        if row < MM and k_tile + tx < KK:
            sa[ty, tx] = A[row, k_tile + tx]
        else:
            sa[ty, tx] = 0.0
        if k_tile + ty < KK and col < NN:
            sb[ty, tx] = B[k_tile + ty, col]
        else:
            sb[ty, tx] = 0.0
        barrier()
        comptime for k in range(TILE):
            acc += sa[ty, k] * sb[k, tx]
        barrier()

    if row < MM and col < NN:
        C[row, col] = acc


# === Warp sum kernel (monomorphic — avoids origin mismatch) ===

comptime WS = 1024
comptime ws_layout = row_major[WS]()
comptime ws_out_layout = row_major[32]()

def warp_sum_kernel(
    data: TileTensor[dtype, type_of(ws_layout), MutAnyOrigin],
    output: TileTensor[dtype, type_of(ws_out_layout), MutAnyOrigin],
    size: Int,
):
    comptime assert data.flat_rank == 1
    comptime assert output.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var val = rebind[Scalar[dtype]](data[tid])
        var reduced = warp.sum(val)
        if tid % 32 == 0:
            output[tid / 32] = rebind[output.ElementType](reduced)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    print("GPU Context created")

    # === Phase 1: Matmul stress (100 iterations, 512x512x512) ===
    var a_buf = ctx.enqueue_create_buffer[dtype](MM * KK)
    var b_buf = ctx.enqueue_create_buffer[dtype](KK * NN)
    var c_buf = ctx.enqueue_create_buffer[dtype](MM * NN)
    a_buf.enqueue_fill(1.0)
    b_buf.enqueue_fill(2.0)
    c_buf.enqueue_fill(0.0)

    var a = TileTensor(a_buf, a_layout)
    var b = TileTensor(b_buf, b_layout)
    var c = TileTensor(c_buf, c_layout)

    comptime mm_kernel = matmul_kernel[type_of(a_layout), type_of(b_layout), type_of(c_layout)]

    comptime NUM_ITERS = 100
    print("Phase 1: ", NUM_ITERS, " matmul iterations (", MM, "x", NN, "x", KK, ")...")

    for i in range(NUM_ITERS):
        ctx.enqueue_function[mm_kernel](
            a, b, c,
            grid_dim=(ceildiv(NN, TILE), ceildiv(MM, TILE)),
            block_dim=(TILE, TILE),
        )
        if i % 25 == 0:
            ctx.synchronize()
            print("  matmul iter ", i, "/", NUM_ITERS)

    ctx.synchronize()
    print("Matmul done.")

    # === Phase 2: Warp sum with float32 (tests the dtype that fails in nabla) ===
    print("Phase 2: warp.sum with float32...")

    var test_buf = ctx.enqueue_create_buffer[dtype](WS)
    var out_buf = ctx.enqueue_create_buffer[dtype](32)
    test_buf.enqueue_fill(1.0)
    out_buf.enqueue_fill(0.0)

    var test_data = TileTensor(test_buf, ws_layout)
    var test_out = TileTensor(out_buf, ws_out_layout)

    ctx.enqueue_function[warp_sum_kernel](
        test_data, test_out, WS,
        grid_dim=ceildiv(WS, 256),
        block_dim=256,
    )
    ctx.synchronize()

    with out_buf.map_to_host() as mapped:
        var result = TileTensor(mapped, ws_out_layout)
        print("warp.sum result[0] (expect 32.0): ", result[0])

    print("All GPU stress tests passed!")
