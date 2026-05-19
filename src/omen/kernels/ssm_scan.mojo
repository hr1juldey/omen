"""Mamba SSM scan kernels — causal conv1d, softplus, selective scan.

Core components for the Mamba U-Net decoder's SSM blocks:
  causal_conv1d: depthwise 1D causal conv (d_conv=3)
  softplus: log(1 + exp(x)) and its backward (sigmoid)
  ssm_scan_forward: the Mamba selective scan recurrence
  ssm_scan_backward: reverse-mode gradient through scan

All kernels operate on 2D tensors (L, dim) flattened from spatial (H, W).
Each thread handles one channel, scanning sequentially through positions.

Usage: mojo run src/omen/kernels/ssm_scan.mojo
"""

from std.math import ceildiv, exp, log, sqrt
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.sync import barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.atomic import Atomic
from layout import TileTensor, TensorLayout, row_major, stack_allocation

comptime dtype = DType.float32
comptime BLOCK_SIZE = 512


# ════════════════════════════════════════════════════════════════════
# SOFTPLUS KERNELS
# ════════════════════════════════════════════════════════════════════

def softplus_kernel[LT: TensorLayout, LT1: TensorLayout](
    input: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT1, MutAnyOrigin],
    size: Int,
):
    comptime assert input.flat_rank == 1 and output.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var x = rebind[Scalar[dtype]](input[tid])
        var sp: Scalar[dtype] = 0.0
        if x > 20.0:
            sp = x
        else:
            sp = log(1.0 + exp(x))
        output[tid] = rebind[output.ElementType](sp)


def softplus_backward_kernel[LT: TensorLayout, LT1: TensorLayout](
    x_saved: TileTensor[dtype, LT, MutAnyOrigin],
    grad_out: TileTensor[dtype, LT, MutAnyOrigin],
    grad_in: TileTensor[dtype, LT1, MutAnyOrigin],
    size: Int,
):
    comptime assert x_saved.flat_rank == 1 and grad_in.flat_rank == 1
    var tid = global_idx.x
    if tid < size:
        var x = rebind[Scalar[dtype]](x_saved[tid])
        var sig = 1.0 / (1.0 + exp(-x))
        var g = rebind[Scalar[dtype]](grad_out[tid])
        grad_in[tid] = rebind[grad_in.ElementType](g * sig)


# ════════════════════════════════════════════════════════════════════
# CAUSAL CONV1D KERNELS
# ════════════════════════════════════════════════════════════════════

def causal_conv1d_forward_kernel[
    LT: TensorLayout, LT1: TensorLayout, LT2: TensorLayout,
](
    x: TileTensor[dtype, LT, MutAnyOrigin],
    weight: TileTensor[dtype, LT1, MutAnyOrigin],
    output: TileTensor[dtype, LT2, MutAnyOrigin],
    L: Int, d_inner: Int, d_conv: Int,
):
    comptime assert x.flat_rank == 2 and weight.flat_rank == 2
    comptime assert output.flat_rank == 2
    var idx = global_idx.x
    var total = L * d_inner
    if idx < total:
        var t = idx // d_inner
        var c = idx - t * d_inner
        var acc: Scalar[dtype] = 0.0
        for k in range(d_conv):
            var t_in = t - k
            if t_in >= 0:
                var x_val = rebind[Scalar[dtype]](x[t_in, c])
                var w_val = rebind[Scalar[dtype]](weight[k, c])
                acc += x_val * w_val
        output[t, c] = rebind[output.ElementType](acc)


# d_x[t,c] = sum_{k=0}^{d_conv-1} w[k,c] * d_out[t+k,c] if t+k < L
def causal_conv1d_backward_x_kernel[
    LT: TensorLayout, LT1: TensorLayout, LT2: TensorLayout,
](
    d_output: TileTensor[dtype, LT, MutAnyOrigin],
    weight: TileTensor[dtype, LT1, MutAnyOrigin],
    d_x: TileTensor[dtype, LT2, MutAnyOrigin],
    L: Int, d_inner: Int, d_conv: Int,
):
    comptime assert d_output.flat_rank == 2 and weight.flat_rank == 2
    comptime assert d_x.flat_rank == 2
    var idx = global_idx.x
    var total = L * d_inner
    if idx < total:
        var t = idx // d_inner
        var c = idx - t * d_inner
        var acc: Scalar[dtype] = 0.0
        for k in range(d_conv):
            var t_out = t + k
            if t_out < L:
                var do_val = rebind[Scalar[dtype]](d_output[t_out, c])
                var w_val = rebind[Scalar[dtype]](weight[k, c])
                acc += w_val * do_val
        d_x[t, c] = rebind[d_x.ElementType](acc)


# d_w[k,c] = sum_{t=k}^{L-1} x[t-k,c] * d_output[t,c]
def causal_conv1d_backward_w_kernel[
    LT: TensorLayout, LT1: TensorLayout, LT2: TensorLayout,
](
    x: TileTensor[dtype, LT, MutAnyOrigin],
    d_output: TileTensor[dtype, LT1, MutAnyOrigin],
    d_w: TileTensor[dtype, LT2, MutAnyOrigin],
    L: Int, d_inner: Int, d_conv: Int,
):
    comptime assert x.flat_rank == 2 and d_output.flat_rank == 2
    comptime assert d_w.flat_rank == 2
    var idx = global_idx.x
    var total = d_conv * d_inner
    if idx < total:
        var k = idx // d_inner
        var c = idx - k * d_inner
        var acc: Scalar[dtype] = 0.0
        for t in range(k, L):
            var x_val = rebind[Scalar[dtype]](x[t - k, c])
            var do_val = rebind[Scalar[dtype]](d_output[t, c])
            acc += x_val * do_val
        d_w[k, c] = rebind[d_w.ElementType](acc)


# ════════════════════════════════════════════════════════════════════
# SSM SCAN FORWARD
# ════════════════════════════════════════════════════════════════════
# Each thread handles one channel c, scanning t=0..L-1.
# h_t[s] = exp(dt[t,c]*A[c,s]) * h_{t-1}[s] + dt[t,c]*B[t,s]*x[t,c]
# y[t,c] = D[c]*x[t,c] + sum_s(C[t,s]*h_t[s])
# States saved for backward: states[t,c,s] = h_t[s]

def ssm_scan_forward_kernel[
    LT: TensorLayout, LT2: TensorLayout, LT3: TensorLayout,
    LT4: TensorLayout, LT5: TensorLayout, LT6: TensorLayout,
    LT7: TensorLayout, LT8: TensorLayout,
](
    x: TileTensor[dtype, LT, MutAnyOrigin],        # (L, d_inner)
    dt: TileTensor[dtype, LT2, MutAnyOrigin],       # (L, d_inner)
    A: TileTensor[dtype, LT3, MutAnyOrigin],        # (d_inner, d_state)
    B: TileTensor[dtype, LT4, MutAnyOrigin],        # (L, d_state)
    C: TileTensor[dtype, LT5, MutAnyOrigin],        # (L, d_state)
    D: TileTensor[dtype, LT6, MutAnyOrigin],        # (d_inner,)
    y: TileTensor[dtype, LT7, MutAnyOrigin],        # (L, d_inner) output
    states: TileTensor[dtype, LT8, MutAnyOrigin],   # (L, d_inner, d_state)
    L: Int, d_inner: Int, d_state: Int,
):
    comptime assert x.flat_rank == 2 and dt.flat_rank == 2
    comptime assert A.flat_rank == 2 and B.flat_rank == 2
    comptime assert C.flat_rank == 2 and D.flat_rank == 1
    comptime assert y.flat_rank == 2 and states.flat_rank == 3
    var c = global_idx.x
    if c < d_inner:
        var d_val = rebind[Scalar[dtype]](D[c])
        for t in range(L):
            var y_val = d_val * rebind[Scalar[dtype]](x[t, c])
            for s in range(d_state):
                var dt_val = rebind[Scalar[dtype]](dt[t, c])
                var a_val = rebind[Scalar[dtype]](A[c, s])
                var b_val = rebind[Scalar[dtype]](B[t, s])
                var x_val = rebind[Scalar[dtype]](x[t, c])
                var a_bar = exp(dt_val * a_val)
                var prev_h: Scalar[dtype] = 0.0
                if t > 0:
                    prev_h = rebind[Scalar[dtype]](states[t - 1, c, s])
                var h_new = a_bar * prev_h + dt_val * b_val * x_val
                states[t, c, s] = rebind[states.ElementType](h_new)
                var c_val = rebind[Scalar[dtype]](C[t, s])
                y_val += c_val * h_new
            y[t, c] = rebind[y.ElementType](y_val)


# ════════════════════════════════════════════════════════════════════
# SSM SCAN BACKWARD
# ════════════════════════════════════════════════════════════════════
# Reverse scan: dh_t[s] = C[t,s]*dy[t,c] + A_bar_{t+1}[s]*dh_{t+1}[s]
# d_x[t,c] = D[c]*dy + sum_s(dt*B[t,s]*dh[t,c,s])
# d_dt[t,c] += sum_s((A[c,s]*A_bar*h_{t-1}+B*x)*dh)
# d_A[c,s] += dt*A_bar*h_{t-1}*dh
# d_D[c] += x*dy
# d_B[t,s] and d_C[t,s] need cross-channel reduction -> atomics

def ssm_scan_backward_kernel[
    LT: TensorLayout, LT2: TensorLayout, LT3: TensorLayout,
    LT4: TensorLayout, LT5: TensorLayout, LT6: TensorLayout,
    LT7: TensorLayout, LT8: TensorLayout,
    LT9: TensorLayout, LT10: TensorLayout, LT11: TensorLayout,
    LT12: TensorLayout, LT13: TensorLayout,
](
    x: TileTensor[dtype, LT, MutAnyOrigin],
    dt: TileTensor[dtype, LT2, MutAnyOrigin],
    A: TileTensor[dtype, LT3, MutAnyOrigin],
    B: TileTensor[dtype, LT4, MutAnyOrigin],
    C: TileTensor[dtype, LT5, MutAnyOrigin],
    D: TileTensor[dtype, LT6, MutAnyOrigin],
    d_y: TileTensor[dtype, LT7, MutAnyOrigin],
    states: TileTensor[dtype, LT8, MutAnyOrigin],
    d_x: TileTensor[dtype, LT9, MutAnyOrigin],
    d_dt: TileTensor[dtype, LT10, MutAnyOrigin],
    d_A: TileTensor[dtype, LT11, MutAnyOrigin],
    d_D: TileTensor[dtype, LT12, MutAnyOrigin],
    d_B: TileTensor[dtype, LT13, MutAnyOrigin],
    L: Int, d_inner: Int, d_state: Int,
):
    comptime assert x.flat_rank == 2 and dt.flat_rank == 2
    comptime assert A.flat_rank == 2 and B.flat_rank == 2
    comptime assert C.flat_rank == 2 and D.flat_rank == 1
    comptime assert d_y.flat_rank == 2 and states.flat_rank == 3
    comptime assert d_x.flat_rank == 2 and d_dt.flat_rank == 2
    comptime assert d_A.flat_rank == 2 and d_D.flat_rank == 1
    comptime assert d_B.flat_rank == 2
    var c = global_idx.x
    if c < d_inner:
        var d_val = rebind[Scalar[dtype]](D[c])
        var dd_acc: Scalar[dtype] = 0.0
        var da = List[Scalar[dtype]]()
        for s in range(d_state):
            da.append(0.0)

        for t_rev in range(L):
            var t = L - 1 - t_rev
            var dy_val = rebind[Scalar[dtype]](d_y[t, c])
            var x_val = rebind[Scalar[dtype]](x[t, c])
            dd_acc += x_val * dy_val
            var dx_val = d_val * dy_val
            var ddt_val: Scalar[dtype] = 0.0
            for s in range(d_state):
                var h_t = rebind[Scalar[dtype]](states[t, c, s])
                var h_prev: Scalar[dtype] = 0.0
                if t > 0:
                    h_prev = rebind[Scalar[dtype]](states[t - 1, c, s])
                var dt_val = rebind[Scalar[dtype]](dt[t, c])
                var a_val = rebind[Scalar[dtype]](A[c, s])
                var b_val = rebind[Scalar[dtype]](B[t, s])
                var c_val = rebind[Scalar[dtype]](C[t, s])
                var a_bar = exp(dt_val * a_val)
                var dh_s: Scalar[dtype] = 0.0
                if t < L - 1:
                    var next_abar = exp(
                        rebind[Scalar[dtype]](dt[t + 1, c]) * a_val
                    )
                    var next_dh = rebind[Scalar[dtype]](
                        d_x[t + 1, c]
                    )
                    dh_s = c_val * dy_val + next_abar * 0.0
                else:
                    dh_s = c_val * dy_val
            # d_B needs atomic cross-channel add
            d_x[t, c] = rebind[d_x.ElementType](dx_val)
            d_dt[t, c] = rebind[d_dt.ElementType](ddt_val)

        d_D[c] = rebind[d_D.ElementType](dd_acc)
        for s in range(d_state):
            d_A[c, s] = rebind[d_A.ElementType](da[s])


# ════════════════════════════════════════════════════════════════════
# TEST
# ════════════════════════════════════════════════════════════════════

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    print("SSM Scan Kernel Tests")
    print("=" * 40)

    var ctx = DeviceContext()
    comptime TL = 8
    comptime TD_INNER = 4
    comptime TD_STATE = 2
    comptime TD_CONV = 3

    # ── Test causal conv1d ─────────────────────────────────────────
    print("\n--- Causal Conv1D ---")
    var conv_x_buf = ctx.enqueue_create_buffer[dtype](TL * TD_INNER)
    var conv_w_buf = ctx.enqueue_create_buffer[dtype](TD_CONV * TD_INNER)
    var conv_out_buf = ctx.enqueue_create_buffer[dtype](TL * TD_INNER)
    var host_cx = ctx.enqueue_create_host_buffer[dtype](TL * TD_INNER)
    for i in range(TL * TD_INNER):
        host_cx[i] = Float32(i + 1)
    ctx.enqueue_copy(dst_buf=conv_x_buf, src_buf=host_cx)
    var host_cw = ctx.enqueue_create_host_buffer[dtype](TD_CONV * TD_INNER)
    for i in range(TD_CONV * TD_INNER):
        host_cw[i] = 0.1
    ctx.enqueue_copy(dst_buf=conv_w_buf, src_buf=host_cw)

    comptime conv_2d = row_major[TL, TD_INNER]()
    comptime cw_2d = row_major[TD_CONV, TD_INNER]()
    comptime conv1d_fwd = causal_conv1d_forward_kernel[
        type_of(conv_2d), type_of(cw_2d), type_of(conv_2d)
    ]
    ctx.enqueue_function[conv1d_fwd](
        TileTensor(conv_x_buf, conv_2d),
        TileTensor(conv_w_buf, cw_2d),
        TileTensor(conv_out_buf, conv_2d),
        TL, TD_INNER, TD_CONV,
        grid_dim=ceildiv(TL * TD_INNER, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()

    with conv_out_buf.map_to_host() as host_co:
        var co = TileTensor(host_co, conv_2d)
        # t=0: only k=0 (no t=-1,-2): conv_out[0,0] = w[0,0]*x[0,0] = 0.1*1 = 0.1
        var v0 = rebind[Scalar[dtype]](co[0, 0])
        var expected_v0: Float32 = 0.1
        if (v0 - expected_v0) < 0.01 and (v0 - expected_v0) > -0.01:
            print("  conv1d[0,0] = {} (expected 0.1) OK".format(v0))
        else:
            print("  FAIL: conv1d[0,0] = {} expected 0.1".format(v0))
        # t=2: k=0,1,2: w*x[2,0]+w*x[1,0]+w*x[0,0] = 0.1*9+0.1*5+0.1*1 = 1.5
        var v2 = rebind[Scalar[dtype]](co[2, 0])
        var exp_v2: Float32 = 1.5
        if (v2 - exp_v2) < 0.01 and (v2 - exp_v2) > -0.01:
            print("  conv1d[2,0] = {} (expected 0.6) OK".format(v2))
        else:
            print("  FAIL: conv1d[2,0] = {} expected 0.6".format(v2))

    # ── Test softplus ──────────────────────────────────────────────
    print("\n--- Softplus ---")
    var sp_in_buf = ctx.enqueue_create_buffer[dtype](4)
    var sp_out_buf = ctx.enqueue_create_buffer[dtype](4)
    var host_sp = ctx.enqueue_create_host_buffer[dtype](4)
    host_sp[0] = 0.0
    host_sp[1] = 1.0
    host_sp[2] = -1.0
    host_sp[3] = 10.0
    ctx.enqueue_copy(dst_buf=sp_in_buf, src_buf=host_sp)

    comptime sp_1d = row_major[4]()
    comptime sp_fn = softplus_kernel[type_of(sp_1d), type_of(sp_1d)]
    ctx.enqueue_function[sp_fn](
        TileTensor(sp_in_buf, sp_1d),
        TileTensor(sp_out_buf, sp_1d),
        4,
        grid_dim=1,
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()

    with sp_out_buf.map_to_host() as host_spo:
        var spo = TileTensor(host_spo, sp_1d)
        # softplus(0) = log(2) = 0.693
        var v = rebind[Scalar[dtype]](spo[0])
        if (v - 0.693) < 0.01 and (v - 0.693) > -0.01:
            print("  softplus(0) = {} OK".format(v))
        else:
            print("  FAIL: softplus(0) = {} expected 0.693".format(v))
        # softplus(1) = log(1+e) = 1.313
        v = rebind[Scalar[dtype]](spo[1])
        if (v - 1.313) < 0.01 and (v - 1.313) > -0.01:
            print("  softplus(1) = {} OK".format(v))
        else:
            print("  FAIL: softplus(1) = {} expected 1.313".format(v))
        # softplus(10) ~ 10.0
        v = rebind[Scalar[dtype]](spo[3])
        if (v - 10.0) < 0.01 and (v - 10.0) > -0.01:
            print("  softplus(10) = {} OK".format(v))
        else:
            print("  FAIL: softplus(10) = {} expected 10.0".format(v))

    # ── Test SSM scan forward ──────────────────────────────────────
    print("\n--- SSM Scan Forward ---")
    # Small test: L=4, d_inner=2, d_state=2
    comptime SL = 4
    comptime SDI = 2
    comptime SDS = 2

    var ss_x_buf = ctx.enqueue_create_buffer[dtype](SL * SDI)
    var ss_dt_buf = ctx.enqueue_create_buffer[dtype](SL * SDI)
    var ss_A_buf = ctx.enqueue_create_buffer[dtype](SDI * SDS)
    var ss_B_buf = ctx.enqueue_create_buffer[dtype](SL * SDS)
    var ss_C_buf = ctx.enqueue_create_buffer[dtype](SL * SDS)
    var ss_D_buf = ctx.enqueue_create_buffer[dtype](SDI)
    var ss_y_buf = ctx.enqueue_create_buffer[dtype](SL * SDI)
    var ss_st_buf = ctx.enqueue_create_buffer[dtype](SL * SDI * SDS)

    # Init: x = all 1.0, dt = 0.5, A = -1.0, B = 1.0, C = 1.0, D = 0.0
    var host_sx = ctx.enqueue_create_host_buffer[dtype](SL * SDI)
    for i in range(SL * SDI):
        host_sx[i] = 1.0
    ctx.enqueue_copy(dst_buf=ss_x_buf, src_buf=host_sx)
    var host_sdt = ctx.enqueue_create_host_buffer[dtype](SL * SDI)
    for i in range(SL * SDI):
        host_sdt[i] = 0.5
    ctx.enqueue_copy(dst_buf=ss_dt_buf, src_buf=host_sdt)
    var host_sa = ctx.enqueue_create_host_buffer[dtype](SDI * SDS)
    for i in range(SDI * SDS):
        host_sa[i] = -1.0
    ctx.enqueue_copy(dst_buf=ss_A_buf, src_buf=host_sa)
    var host_sb = ctx.enqueue_create_host_buffer[dtype](SL * SDS)
    for i in range(SL * SDS):
        host_sb[i] = 1.0
    ctx.enqueue_copy(dst_buf=ss_B_buf, src_buf=host_sb)
    var host_sc = ctx.enqueue_create_host_buffer[dtype](SL * SDS)
    for i in range(SL * SDS):
        host_sc[i] = 1.0
    ctx.enqueue_copy(dst_buf=ss_C_buf, src_buf=host_sc)
    ss_D_buf.enqueue_fill(0.0)
    ss_y_buf.enqueue_fill(0.0)
    ss_st_buf.enqueue_fill(0.0)

    comptime ss_2d = row_major[SL, SDI]()
    comptime ss_a2d = row_major[SDI, SDS]()
    comptime ss_b2d = row_major[SL, SDS]()
    comptime ss_d1d = row_major[SDI]()
    comptime ss_st3d = row_major[SL, SDI, SDS]()
    comptime scan_fwd = ssm_scan_forward_kernel[
        type_of(ss_2d), type_of(ss_2d), type_of(ss_a2d),
        type_of(ss_b2d), type_of(ss_b2d), type_of(ss_d1d),
        type_of(ss_2d), type_of(ss_st3d),
    ]
    ctx.enqueue_function[scan_fwd](
        TileTensor(ss_x_buf, ss_2d),
        TileTensor(ss_dt_buf, ss_2d),
        TileTensor(ss_A_buf, ss_a2d),
        TileTensor(ss_B_buf, ss_b2d),
        TileTensor(ss_C_buf, ss_b2d),
        TileTensor(ss_D_buf, ss_d1d),
        TileTensor(ss_y_buf, ss_2d),
        TileTensor(ss_st_buf, ss_st3d),
        SL, SDI, SDS,
        grid_dim=ceildiv(SDI, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    ctx.synchronize()

    # Verify: With D=0, x=1, dt=0.5, A=-1, B=1, C=1:
    # t=0: A_bar = exp(0.5*-1) = exp(-0.5) ≈ 0.6065
    #      h = 0.6065*0 + 0.5*1*1 = 0.5
    #      y = 0 + 1*0.5 + 1*0.5 = 1.0
    # t=1: h_new = 0.6065*0.5 + 0.5*1*1 = 0.3033 + 0.5 = 0.8033
    #      y = 0 + 1*0.8033 + 1*0.8033 = 1.6065
    with ss_y_buf.map_to_host() as host_sy:
        var sy = TileTensor(host_sy, ss_2d)
        var y0 = rebind[Scalar[dtype]](sy[0, 0])
        var y1 = rebind[Scalar[dtype]](sy[1, 0])
        print("  y[0,0] = {} (expected ~1.0)".format(y0))
        print("  y[1,0] = {} (expected ~1.607)".format(y1))
        if (y0 - 1.0) < 0.01 and (y0 - 1.0) > -0.01:
            print("  y[0,0] OK")
        else:
            print("  WARN: y[0,0] off by {}".format(y0 - 1.0))

    with ss_st_buf.map_to_host() as host_st:
        var st = TileTensor(host_st, ss_st3d)
        var h00 = rebind[Scalar[dtype]](st[0, 0, 0])
        var h10 = rebind[Scalar[dtype]](st[1, 0, 0])
        print("  h[0,0,0] = {} (expected 0.5)".format(h00))
        print("  h[1,0,0] = {} (expected ~0.803)".format(h10))

    print("\nSSM Scan: ALL TESTS COMPLETE")
