# [BUG]: conv_transpose crashes with `cudnnCreate symbol not found` when 2+ conv2d layers use backward pass on GPU

## Summary

When computing gradients through **2 or more native `nb.conv2d` layers** on GPU (via `nb.value_and_grad`), the backward pass crashes with `ABORT: symbol not found: cudnnCreate` (SIGABRT, exit code 132). A single `conv2d` with backward works correctly. This makes multi-layer conv2d architectures impossible to train on GPU using nabla/MAX.

## Description

### What happened

We are building a JEPA (Joint-Embedding Predictive Architecture) denoiser using nabla for GPU training. The architecture requires multiple conv2d layers in both the encoder and decoder. When we call `nb.value_and_grad` on a loss function containing 2+ `nb.conv2d` operations, the backward pass (which uses `conv_transpose` internally for the VJP) crashes with a fatal error:

```
ABORT: oss/modular/mojo/stdlib/std/ffi/__init__.mojo:629:18: symbol not found: cudnnCreate
```

This is a hard crash (SIGABRT), not a Python exception. The process terminates immediately.

### What was expected

Multiple conv2d layers should support backward pass (gradient computation) on GPU, just as a single conv2d layer does. This is essential for any non-trivial CNN architecture (U-Net, ResNet, VAE, etc.).

### Scope of the problem

- **1 conv2d + backward**: WORKS on GPU at all tested sizes (16x16, 32x32, 64x64) with channels up to 32. JIT compiles in ~54s, executes correctly, gradients are numerically correct.
- **2+ conv2d + backward**: CRASHES even at the smallest possible size (16x16, 4→8→16 channels). The crash occurs during backward graph compilation/execution, not at any specific spatial size or channel count.
- **Forward-only with 2+ conv2d**: Works fine (0.1s, no crash). Only the backward pass triggers the bug.

### Impact

This is a **blocker** for any multi-layer CNN training on GPU with nabla/MAX. All standard architectures (ResNet, U-Net, VAE, Diffusion models, etc.) require multiple conv2d layers with gradient computation.

## Steps to reproduce

### Minimal reproducer 1: Two conv2d layers (CRASHES)

```python
import numpy as np
import nabla as nb
from max.driver import CPU, Accelerator, accelerator_count

assert accelerator_count() > 0, "No GPU available"
dev = Accelerator()

# Create small input and params
x = nb.ops.transfer_to(
    nb.Tensor.from_dlpack(np.random.randn(1, 16, 16, 4).astype(np.float32)), dev
)
params = {
    "w0": nb.ops.transfer_to(
        nb.Tensor.from_dlpack(np.random.randn(3, 3, 4, 8).astype(np.float32) * 0.01),
        dev,
    ),
    "b0": nb.ops.transfer_to(nb.Tensor.from_dlpack(np.zeros(8, dtype=np.float32)), dev),
    "w1": nb.ops.transfer_to(
        nb.Tensor.from_dlpack(np.random.randn(3, 3, 8, 16).astype(np.float32) * 0.01),
        dev,
    ),
    "b1": nb.ops.transfer_to(
        nb.Tensor.from_dlpack(np.zeros(16, dtype=np.float32)), dev
    ),
}


def loss_fn(p, x):
    h = nb.conv2d(x, p["w0"], padding=(1, 1, 1, 1), bias=p["b0"])
    h = nb.conv2d(h, p["w1"], padding=(1, 1, 1, 1), bias=p["b1"])
    return nb.mean(h * h)


# Forward pass works fine
loss_val = loss_fn(params, x)
loss_f = float(nb.ops.transfer_to(loss_val, CPU()).to_numpy())
print(f"Forward OK: loss={loss_f:.6f}")

# Backward pass CRASHES
loss_val, grads = nb.value_and_grad(loss_fn, argnums=0)(params, x)
# The next line triggers the crash:
for k, v in grads.items():
    g = nb.ops.transfer_to(v, CPU()).to_numpy()
    print(f"grad {k}: shape={g.shape}")
```

**Expected**: Gradients are computed successfully.
**Actual**: Process crashes with `ABORT: symbol not found: cudnnCreate`.

### Minimal reproducer 2: Single conv2d layer (WORKS)

Same as above but with only 1 conv2d:

```python
def loss_fn(p, x):
    h = nb.conv2d(x, p["w0"], padding=(1, 1, 1, 1), bias=p["b0"])
    return nb.mean(h * h)
```

This works at all spatial sizes and channel counts tested.

### Systematic testing results

| # conv2d | Spatial | Channels | Result |
|----------|---------|----------|--------|
| 1 | 16x16 | 4→8 | SUCCESS (0.1s) |
| 1 | 32x32 | 4→16 | SUCCESS (55.9s) |
| 1 | 64x64 | 4→16 | SUCCESS (54.1s) |
| 1 | 64x64 | 4→32 | SUCCESS (53.9s) |
| 2 | 16x16 | 4→8→16 | CRASH (cudnnCreate) |
| 2 | 64x64 | 4→8→16 | CRASH (cudnnCreate) |
| 2 | 64x64 | 4→16→32 | CRASH (cudnnCreate) |
| 3 | 64x64 | 4→16→32→64 | CRASH (cudnnCreate) |

All tests use stride=1, padding=(1,1,1,1), 3x3 kernel. The crash is **independent** of spatial size and channel count — it only depends on having 2+ conv2d layers with backward.

## Error output

```
ABORT: oss/modular/mojo/stdlib/std/ffi/__init__.mojo:629:18: symbol not found: cudnnCreate
```

When the crash occurs with a more complex model, the error also includes cuDNN allocation failures:

```
ValueError: An error occurred in kernel entry point named "region_154":
An error occurred in kernel named "stub_123":
cuDNN call failed with status CUDNN_STATUS_ALLOC_FAILED

Fusion info:
  mo.conv_transpose : (!mo.tensor<[1, 64, 64, 3], f32, gpu:0, {layout = #mo.layout<NHWC>}>,
                       !mo.tensor<[3, 3, 16, 3], f32, gpu:0, {layout = #mo.layout<RSCF>}>,
                       ...) -> !mo.tensor<[1, 66, 66, 16], f32, gpu:0>
```

Note: The `CUDNN_STATUS_ALLOC_FAILED` error occurs when the first conv_transpose succeeds but subsequent ones fail. With truly minimal examples (16x16), it crashes with `cudnnCreate symbol not found` before even attempting allocation.

## Environment details

### Software

| Component | Version |
|-----------|---------|
| MAX | 26.4.0.dev2026051506 (nightly) |
| Python | 3.13.5 |
| OS | Pop!_OS 24.04 LTS (Ubuntu derivative), kernel 6.18.7-76061807-generic |
| nabla | Latest from GitHub main (no version attribute) |
| NVIDIA driver | 580.159.03 |
| CUDA | 13.0 |

### Hardware

| Component | Details |
|-----------|---------|
| GPU | NVIDIA GeForce RTX 3060 |
| VRAM | 12,288 MiB |
| System RAM | 32 GB |

## Severity/frequency

**Severity: Blocker**

This completely prevents training any multi-layer CNN architecture on GPU with nabla. All modern vision architectures (ResNet, U-Net, VAE, Diffusion models, etc.) require gradient computation through multiple conv2d layers.

**Frequency: Always** — 100% reproducible with any 2+ conv2d backward pass on GPU.

## Analysis and hypotheses

### Why single conv2d works but multiple don't

1. A single `nb.conv2d` backward uses one `conv_transpose` operation. cuDNN loads successfully and allocates workspace for this single operation.

2. With 2+ `nb.conv2d` layers, the backward graph contains 2+ `conv_transpose` operations. When nabla compiles the backward graph into an MLIR module, it creates a fused kernel containing multiple `conv_transpose` ops. The Mojo FFI layer then tries to dynamically load cuDNN symbols for the second `conv_transpose`, and fails with `symbol not found: cudnnCreate`.

3. This suggests a bug in the MAX runtime's dynamic loading of cuDNN when multiple `conv_transpose` ops are present in the same compiled region. The cuDNN library handle may not be properly shared or re-initialized across multiple `conv_transpose` invocations within the same kernel.

### Related issues

- #5543 — Conv2D compilation failure on non-unit dilations (conv2d limitations)
- #6248 — Conv2d produces incorrect results when C_in >= 8 (numerical correctness)
- #6461 — Conv2d corrupts boundary output rows of padded input (padding bug)
- #5614 — Cannot use grouped convolutions in graph mode (graph mode limitations)

These issues collectively suggest that the conv2d implementation in MAX has several edge cases that need attention.

### Workaround

We have confirmed that using exactly **1 conv2d layer** with backward works at 64x64 resolution. For our training pipeline, we use:

- 1 native `nb.conv2d` in the encoder (with backward)
- Linear (matmul) layers for all other spatial processing
- Numpy-based optimizer (CPU round-trip for graph breaking)

This is a severely limiting workaround but proves the basic GPU training pipeline is functional.

## Additional context

This bug was discovered during development of the Omen JEPA denoiser project, which uses nabla for GPU-accelerated neural network training integrated with Mitsuba 3 path tracing. The project is open-source and the test demonstrating this bug is at `tests/test_gpu_native_conv_train.py` in our repository.
