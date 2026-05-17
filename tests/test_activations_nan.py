#!/usr/bin/env python3
"""Micro-test: activation backward safety on CPU + GPU.

Tests both nabla builtins (nb.sigmoid/nb.silu) and our custom
sigmoid_gpu/silu_gpu on CPU and GPU, inside and outside @nb.compile.

Usage:
    uv run python tests/test_activations_nan.py
"""

import sys

sys.path.insert(0, "src")

import numpy as np
import nabla as nb
from max.driver import Accelerator, accelerator_count

from omen.kernels.activations import sigmoid_gpu, silu_gpu, square

PASS = 0
FAIL = 0


def _report(name, ok, detail=""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def _has_nan(t):
    return bool(np.any(np.isnan(t.to_numpy())))


vals = np.array([[1e8, -1e8, 1e4, -1e4, 1.0, 0.0]], dtype=np.float32)

# ═══════════════════════════════════════════════════════════════
print("\n=== CPU backward tests ===")
# ═══════════════════════════════════════════════════════════════
x_cpu = nb.Tensor.from_dlpack(vals.copy())

# Built-in (known good)
loss, grad = nb.value_and_grad(lambda x: nb.sigmoid(x).sum(), argnums=0)(x_cpu)
_report(
    "nb.sigmoid backward (CPU)", not _has_nan(grad), "baseline — should be NaN-free"
)

loss, grad = nb.value_and_grad(lambda x: nb.silu(x).sum(), argnums=0)(x_cpu)
_report("nb.silu backward (CPU)", not _has_nan(grad), "baseline — should be NaN-free")

# Our custom
loss, grad = nb.value_and_grad(lambda x: sigmoid_gpu(x).sum(), argnums=0)(x_cpu)
_report("sigmoid_gpu backward (CPU)", not _has_nan(grad))

loss, grad = nb.value_and_grad(lambda x: silu_gpu(x).sum(), argnums=0)(x_cpu)
_report("silu_gpu backward (CPU)", not _has_nan(grad))

loss, grad = nb.value_and_grad(lambda x: square(x).sum(), argnums=0)(x_cpu)
_report("square backward (CPU)", not _has_nan(grad))


gpu_available = accelerator_count() > 0
print(f"\n=== GPU backward tests (available={gpu_available}) ===")
# ═══════════════════════════════════════════════════════════════

if gpu_available:
    dev = Accelerator()

    def _to_gpu(arr):
        t = nb.Tensor.from_dlpack(arr.copy())
        return nb.ops.transfer_to(t, dev)

    def _to_cpu(t):
        return nb.ops.transfer_to(t, nb.CPU())

    x_gpu = _to_gpu(vals)

    # ── Built-in on GPU ──
    for name, fn in [("nb.sigmoid", nb.sigmoid), ("nb.silu", nb.silu)]:
        try:
            loss, grad = nb.value_and_grad(lambda x: fn(x).sum(), argnums=0)(x_gpu)
            grad_np = _to_cpu(grad).to_numpy()
            _report(f"{name} backward (GPU)", not np.any(np.isnan(grad_np)))
        except Exception as e:
            err = str(e).split("\n")[0][:100]
            _report(f"{name} backward (GPU)", False, err)

    # ── Custom on GPU ──
    for name, fn in [("sigmoid_gpu", sigmoid_gpu), ("silu_gpu", silu_gpu)]:
        try:
            loss, grad = nb.value_and_grad(lambda x: fn(x).sum(), argnums=0)(x_gpu)
            grad_np = _to_cpu(grad).to_numpy()
            _report(f"{name} backward (GPU)", not np.any(np.isnan(grad_np)))
        except Exception as e:
            err = str(e).split("\n")[0][:100]
            _report(f"{name} backward (GPU)", False, err)

    # square on GPU
    x_sq_gpu = _to_gpu(np.array([[3.0, -3.0, 0.0, 1e20]], dtype=np.float32))
    try:
        loss, grad = nb.value_and_grad(lambda x: square(x).sum(), argnums=0)(x_sq_gpu)
        _report("square backward (GPU)", not _has_nan(_to_cpu(grad)))
    except Exception as e:
        _report("square backward (GPU)", False, str(e)[:100])

    # ═══════════════════════════════════════════════════════════
    print("\n=== @nb.compile + value_and_grad on GPU ===")
    # ═══════════════════════════════════════════════════════════

    # Test A: nb.sigmoid inside @nb.compile
    @nb.compile
    def compiled_builtin(params, x, target):
        def loss_fn(p, x, t):
            h = nb.silu(x @ p["w"] + p["b"])
            diff = h - t
            return nb.mean(nb.mul(diff, diff))

        loss, grads = nb.value_and_grad(loss_fn, argnums=0)(params, x, target)
        return loss, grads

    # Test B: custom sigmoid_gpu inside @nb.compile
    @nb.compile
    def compiled_custom(params, x, target):
        def loss_fn(p, x, t):
            h = silu_gpu(x @ p["w"] + p["b"])
            diff = h - t
            return nb.mean(nb.mul(diff, diff))

        loss, grads = nb.value_and_grad(loss_fn, argnums=0)(params, x, target)
        return loss, grads

    micro_params = {
        "w": _to_gpu(np.random.randn(6, 6).astype(np.float32) * 0.02),
        "b": _to_gpu(np.zeros((1, 6), dtype=np.float32)),
    }
    x2 = _to_gpu(np.random.randn(1, 6).astype(np.float32))
    t2 = _to_gpu(np.random.randn(1, 6).astype(np.float32))

    for label, fn in [("nb.silu", compiled_builtin), ("silu_gpu", compiled_custom)]:
        try:
            loss, grads = fn(micro_params, x2, t2)
            loss_np = float(_to_cpu(loss).to_numpy())
            has_nan = any(_has_nan(_to_cpu(g)) for g in grads.values())
            _report(f"@nb.compile + {label} (GPU)", not has_nan, f"loss={loss_np:.4e}")
        except Exception as e:
            err = str(e).split("\n")[0][:100]
            _report(f"@nb.compile + {label} (GPU)", False, err)

    # ═══════════════════════════════════════════════════════════
    print("\n=== Multi-step compiled train on GPU ===")
    # ═══════════════════════════════════════════════════════════
    from nabla.nn.optim import adamw_init, adamw_update

    for label, compile_fn in [
        ("nb.silu", compiled_builtin),
        ("silu_gpu", compiled_custom),
    ]:
        p = {
            "w": _to_gpu(np.random.randn(6, 6).astype(np.float32) * 0.02),
            "b": _to_gpu(np.zeros((1, 6), dtype=np.float32)),
        }
        opt = adamw_init(p)
        ok = True
        for step in range(1, 6):
            try:
                loss, grads = compile_fn(p, x2, t2)
                loss_val = float(_to_cpu(loss).to_numpy())
                has_nan_g = any(_has_nan(_to_cpu(g)) for g in grads.values())
                if np.isnan(loss_val) or has_nan_g:
                    ok = False
                    _report(
                        f"{label} step {step}",
                        False,
                        f"loss={loss_val:.4e} nan={has_nan_g}",
                    )
                    break
                p, opt = adamw_update(p, grads, opt, lr=5e-5, weight_decay=0.01)
            except Exception as e:
                ok = False
                _report(f"{label} step {step}", False, str(e)[:80])
                break
        if ok:
            final_loss = float(_to_cpu(compile_fn(p, x2, t2)[0]).to_numpy())
            _report(f"{label} 5-step train", True, f"final_loss={final_loss:.4e}")

else:
    print("  (skipped — no GPU detected)")

# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL > 0:
    print("See FAIL details above.")
else:
    print("All passed.")
print(f"{'=' * 60}")
sys.exit(1 if FAIL > 0 else 0)
