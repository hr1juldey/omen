## Why

MAX's native `nb.conv2d` backward pass crashes with `cudnnCreate symbol not found` (SIGABRT) when 2+ conv2d layers compute gradients on GPU. This is a **blocker** for any multi-layer CNN training (ResNet, U-Net, VAE, etc.). Nabla's `Operation` base class exposes a public `vjp_rule` API that allows custom backward implementations — we can bypass the broken `conv_transpose` by writing our own VJP using pure nabla matmul ops.

## What Changes

- New `Conv2dMojoOp(Operation)` subclass with custom `vjp_rule` that computes gradients via im2col + matmul instead of cuDNN `conv_transpose`
- Forward pass: Mojo GPU im2col kernel (already exists in `src/omen/kernels/conv2d_im2col.mojo`) extracts patches, then nabla `matmul` computes output
- Backward pass: `vjp_rule` computes `grad_filter` and `grad_input` using only nabla matmul/reshape/pad — zero cuDNN dependency
- Fallback to pure-nabla im2col if `call_custom_kernel` fails (`std::bad_cast`)
- Standalone test file proving multi-layer conv2d backward works on GPU with progressive scale-up and RAM guards

## Capabilities

### New Capabilities
- `mojo-conv2d-op`: Custom nabla `Operation` subclass with Mojo GPU forward + pure-nabla backward, bypassing broken MAX conv_transpose
- `conv2d-backward-test`: Progressive GPU test proving multi-layer conv2d backward works (16x16→64x64), with RAM guards and numerical gradient verification

### Modified Capabilities

_(none — no existing specs are modified; this is a standalone prototype)_

## Impact

- **New file**: `tests/test_gpu_mojo_conv2d_backward.py` (~400 LOC standalone test)
- **Read-only dependency**: `src/omen/kernels/conv2d_im2col.mojo`, `conv2im.mojo` (existing Mojo GPU kernels)
- **No modifications**: `src/omen/` code is untouched
- **Dependencies**: nabla `Operation`, `call_custom_kernel`, `nb.matmul`, `nb.reshape`, `nb.pad`, `nb.concatenate` — all existing APIs
- **GPU**: Requires NVIDIA GPU with CUDA (same as existing GPU tests)
- **RAM**: 20GB process RSS limit with gc.collect() between phases to prevent RAM explosions
