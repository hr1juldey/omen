## Context

MAX's `nb.conv2d` uses cuDNN's `conv_transpose` for the backward pass (VJP). When 2+ `conv_transpose` ops exist in the same compiled GPU kernel, the MAX runtime fails to load `cudnnCreate` symbol → SIGABRT. A single conv2d backward works fine. This blocks all multi-layer CNN training on GPU.

Nabla exposes `Operation.vjp_rule(primals, cotangents, outputs, kwargs)` as a public overridable API. The existing `Conv2DOp.vjp_rule` in nabla source calls `conv2d_transpose` — the broken op. We can write a custom `Operation` subclass that computes the same gradients using only `matmul`, `reshape`, `pad`, and `concatenate` — all of which have working GPU VJPs.

Two Mojo GPU kernels already exist: `conv2d_im2col.mojo` (forward patch extraction) and `conv2im.mojo` (backward scatter). These use `call_custom_kernel` which may hit `std::bad_cast` — so we design a fallback to pure-nabla im2col.

## Goals / Non-Goals

**Goals:**
- Prove multi-layer conv2d backward works on GPU without cuDNN conv_transpose
- Custom `Operation` subclass with `vjp_rule` using only nabla matmul ops
- Progressive scale-up test (16x16 → 64x64) with RAM guards
- Numerical gradient verification (finite differences) at each phase
- Fallback path if Mojo `call_custom_kernel` fails

**Non-Goals:**
- Production-ready conv2d (no optimized Winograd, no shared memory tiling)
- Modifying any `src/omen/` code
- Replacing native `nb.conv2d` globally
- Supporting all conv2d configurations (dilation, groups, stride > 2)

## Decisions

### 1. `Operation` base class (not `UnaryOperation`)

`Conv2dMojoOp` has two inputs (x, filter) and needs `vjp_rule` for autograd. `UnaryOperation` only supports `_derivative` for elementwise ops — wrong abstraction. `Operation.vjp_rule(primals, cotangents, outputs, kwargs) -> list[Tensor | None]` is the correct API.

### 2. Im2col + matmul for forward (not direct convolution)

Direct convolution in Mojo requires shared memory tiling, which is complex. Im2col reduces conv2d to a single `matmul` — which nabla already handles perfectly on GPU with working VJPs. The Mojo GPU kernel only needs to do the memory rearrangement (im2col), which is simpler and already written.

### 3. VJP via matmul transpose (not conv_transpose)

The backward of `Y = patches @ filter_flat` is:
- `grad_filter = patches.T @ cotangent_flat` (matmul)
- `grad_patches = cotangent_flat @ filter_flat.T` (matmul)
- `grad_input = col2im(grad_patches)` (scatter or pure-nabla scatter via pad+add)

No `conv_transpose` needed. All ops (`matmul`, `reshape`, `pad`, `concatenate`) have working GPU VJPs in nabla.

### 4. Dual-path: Mojo GPU im2col OR pure-nabla im2col

Try Mojo `call_custom_kernel` first. If it fails (`std::bad_cast` or any exception), fall back to pure-nabla im2col (pad/slice/concat pattern from `conv2d_safe`). The VJP rule is identical regardless of forward path — it only uses the stored `patches` tensor.

### 5. Progressive scale-up with cleanup between phases

Each test phase runs at a larger resolution. Between phases: `gc.collect()`, `nb.GRAPH.clear_all()`, delete all tensors, check RSS. This prevents RAM explosions from accumulated JIT compilation artifacts.

### 6. Numerical gradient verification

At each phase, compare analytical gradients (from `vjp_rule`) against numerical gradients (finite differences with eps=1e-3). Must match within `max_diff < 0.1` (loose because float32 im2col matmul has higher error than cuDNN's fused kernel).

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| `call_custom_kernel` `std::bad_cast` | Automatic fallback to pure-nabla im2col |
| JIT compilation RAM spike (>16GB) | Progressive scale-up; 20GB RSS limit with gc.collect() between phases |
| Numerical gradient mismatch | Loose tolerance (0.1); verify with finite differences at each phase |
| Pure-nabla im2col slow (94s/conv2d JIT) | Only first compile is slow; subsequent reuses cache. Test file focuses on correctness, not speed |
| im2col memory overhead (patches matrix large) | Start at 16x16 with tiny channels; scale up gradually |
| VJP rule bug (wrong gradient formula) | Numerical gradient verification catches this immediately |

## Migration Plan

Not applicable — standalone test file. No production code changes.

## Open Questions

- Will `call_custom_kernel` work for `conv2d_im2col` on GPU? (Other kernels like `moe_dispatch` crash with `std::bad_cast`, but `conv2d_im2col` uses the same `foreach` pattern as the tutorial's `add_one` which works)
- What's the maximum resolution we can reach before im2col patches matrix OOMs? (64x64 with 32 channels should be ~16MB — well within limits)
