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

### Related issues — conv2d has a pattern of bugs

The `conv_transpose` multi-layer crash is not an isolated incident. The conv2d implementation in MAX has at least **4 open bugs** that collectively make it unreliable for production use:

#### #5543 — Conv2D compilation failure on non-unit dilations

- **Status**: Open
- **Symptom**: `ValueError: Non-unit dilation is not supported yet.`
- **Impact**: Any architecture using dilated convolutions (DeepLab, WaveNet, atrous spatial pyramid pooling) cannot compile. Notably, `conv2d_transpose` supports non-unit dilation but `conv2d` does not — an asymmetric API contract.
- **Discovered by**: TilliFe while developing conv2d kernels for the Nabla library (Nov 2025)
- **Environment**: Apple M3, macOS 26.0.1, MAX 25.6.1

#### #6248 — Conv2d produces incorrect results when C_in >= 8

- **Status**: Open
- **Symptom**: Numerically incorrect output that diverges from PyTorch and numpy ground truth. Error grows with channel count.
- **Impact**: Blocks any model with C_in >= 8 from producing correct results. Since most real architectures have C_in >= 16 from the second layer onward, this makes conv2d unreliable for anything beyond toy examples.
- **Root cause hypothesis**: Filter packing issue tied to SIMD/micro-kernel tile width (AVX2: 8 floats). The C_in=8 threshold strongly suggests boundary misalignment in `pack_filter`, similar to the `groups > 1` bug tracked internally as KERN-2567.
- **Error magnitude**: max_diff ranges from 0.017 (C_in=8) to 0.166 (C_in=192, K=7) — not subtle numerical noise but significantly wrong values.
- **Discovered by**: itsdevcoffee (Mar 2026), blocks NSF-HiFiGAN vocoder in mojo-audio pipeline
- **Environment**: NVIDIA RTX 4060 Ti, CUDA 13.0, MAX 26.3.0

#### #6461 — Conv2d silently corrupts boundary output rows of padded input (CPU)

- **Status**: Open, PR #6508 pending
- **Symptom**: `ops.conv2d` on CPU silently corrupts boundary output rows whenever its input was zero-padded (via `ops.pad` or `conv2d`'s own `padding=`). The same input constructed via `ops.concat([zeros, x, zeros])` produces correct output.
- **Impact**: Silent correctness bug — no error, no warning, just wrong numbers. This is the most dangerous type of bug because it can go undetected during development and corrupt model training silently.
- **Reproduced across**: MAX 26.2.0, 26.3.0.dev2026042005, 26.3.0.dev2026042605 — corruption pattern shifts between versions but never resolves.
- **Discovered by**: richardkiss (Apr 2026) while porting VoxCPM audio VAE to MAX. The reconstructed audio came out as "saturated garbage" — bisected to a single conv2d op.
- **Notable**: Issue includes a PEP 723 self-contained reproducer (`uv run 01_mre.py` with no setup).

#### #5614 — Cannot use grouped convolutions in graph mode

- **Status**: Open, Needs Triage
- **Symptom**: Grouped convolutions (groups > 1) fail when used inside MAX graph mode. Only groups=1 (standard convolution) works.
- **Impact**: Blocks depthwise convolutions (MobileNet, EfficientNet), grouped convolutions (ResNeXt), and any architecture using `groups` parameter. These are fundamental building blocks of efficient modern architectures.
- **Discovered by**: gabrieldemarmiesse (Nov 2025)
- **Environment**: MAX graph API, CPU

#### Pattern analysis

These 5 bugs (including this one) reveal a systemic pattern:

| Bug ID | Layer | Component | Type |
|--------|-------|-----------|------|
| This issue | Runtime/cuDNN | `conv_transpose` multi-instance | Crash (SIGABRT) |
| #5543 | Compiler | Dilation support | Feature gap |
| #6248 | Kernel | Filter packing (SIMD) | Silent numerical error |
| #6461 | Kernel | Padding boundary | Silent data corruption |
| #5614 | Graph compiler | Grouped conv lowering | Compilation error |

The conv2d implementation appears to have correctness gaps at every level: compilation (dilation, groups), kernel execution (filter packing, padding), and runtime (cuDNN multi-instance). For users building production neural network training pipelines, this means conv2d cannot be relied upon beyond the simplest single-layer, groups=1, dilation=1, C_in<8 case.

### Workaround

We have confirmed that using exactly **1 conv2d layer** with backward works at 64x64 resolution. For our training pipeline, we use:

- 1 native `nb.conv2d` in the encoder (with backward)
- Linear (matmul) layers for all other spatial processing
- Numpy-based optimizer (CPU round-trip for graph breaking)

This is a severely limiting workaround but proves the basic GPU training pipeline is functional.

#### Attempted workaround: Custom Mojo GPU kernels

We investigated writing conv2d as a custom Mojo GPU kernel to bypass the broken MAX runtime conv2d/conv_transpose path. Two Mojo GPU kernels already exist in our codebase (`conv2d_im2col.mojo` for forward, `conv2im.mojo` for backward col2im), registered as nabla custom operations via `call_custom_kernel`. This approach failed because:

1. **No VJP registration for custom ops**: Nabla's autodiff cannot backpropagate through `call_custom_kernel` ops. There is no public API to register custom VJP (backward) rules for custom operations. Without this, the Mojo GPU kernels are forward-only.

2. **`std::bad_cast` runtime bug**: The `call_custom_kernel` API has type marshalling issues that cause `std::bad_cast` crashes at runtime for several of our other custom kernels (`moe_dispatch`, `mla_compress`, `ssim_kernel`). This makes the custom kernel path unreliable.

3. **Im2col approach still multi-op**: Even with working Mojo GPU kernels for im2col/col2im, the conv2d computation still requires `nb.matmul` — so each conv2d becomes 2 custom ops + 1 matmul, which may still trigger slow JIT compilation.

## Additional context

This bug was discovered during development of the Omen JEPA denoiser project, which uses nabla for GPU-accelerated neural network training integrated with Mitsuba 3 path tracing. The project is open-source and the test demonstrating this bug is at `tests/test_gpu_native_conv_train.py` in our repository.

We are committed to helping resolve this issue and can provide:

- Full reproduction scripts (attached above)
- Access to our training logs showing the crash at various model sizes
- GPU environment for testing fixes (NVIDIA RTX 3060, CUDA 13.0, driver 580.159.03)
- The `tests/test_gpu_native_conv_train.py` file which systematically tests 1-N conv2d layers with backward
