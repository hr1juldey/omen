# Nabla ML Framework — Study for Omen JEPA Engine

## Executive Summary

Nabla is a **complete deep learning framework** with Python API and Mojo/MAX backend.
It has everything Omen needs: `nn.Module`, `AdamW`, autograd, conv2d, attention,
custom Mojo kernels, LoRA, checkpointing, distributed training, and `@nb.compile`.

**The point of Mojo is efficiency** — one stack for everything. Nabla IS the training
framework. No PyTorch, no fallbacks. We solve memory constraints WITHIN nabla.

**Memory budget**: 3-4GB RAM for training. RTX 3060 12GB (6GB for rendering in prod,
4GB for denoiser). All solutions must stay within these limits.

---

## Memory Management Solutions (WITHIN Nabla)

### The Problem

Nabla compiles Python functions into MAX MLIR graphs. Each compiled graph for our model
consumes **~6-10GB RAM** in the `_GRAPH_CACHE`. The cache has no built-in eviction on
the global `_GRAPH_CACHE`, but `@nb.compile` has per-function `max_cache_size`.

### Solution 1: `@nb.compile(max_cache_size=1)` — Single compiled graph

```python
@nb.compile(max_cache_size=1)
def train_step(model, opt_state, noisy, gt, scene_graph):
    loss, grads = nb.value_and_grad(loss_fn, argnums=0)(model, noisy, gt, scene_graph)
    model, opt_state = nb.nn.optim.adamw_update(model, grads, opt_state, lr=lr)
    return model, opt_state, loss
```

Effect: Keeps only 1 compiled entry. New shape → evict old → recompile. Peak = 1 graph.

### Solution 2: `nb.GRAPH.clear_all()` — Manual flush

```python
nb.GRAPH.clear_all()  # Free all compiled graphs
```

Call after each frame's sub-steps. Next step recompiles (~10s first time) then caches.
With `@nb.compile`, cache hit rate is ~98% for same shapes.

### Solution 3: Pipeline Parallelism — Split model across stages

From Example 07: Each pipeline stage compiles only its portion of the model.
Memory per stage = `total_graph_size / num_stages`.

```python
mesh = DeviceMesh("pp", (4,), ("stage",))
# 4 stages: each compiles ~2.5GB instead of ~10GB
```

This is the **nabla-native way** to handle large models on constrained devices.
Even on a single GPU, pipeline stages reduce peak memory.

### Solution 4: QLoRA (4-bit quantization)

From Example 11: Freeze base weights at NF4 (4 bits), train only low-rank adapters.
Saves ~75% memory on weights. Combined with pipeline, could fit in 3-4GB.

```python
from nabla.nn.finetune import init_lora_adapter, lora_linear

adapter = init_lora_adapter(base_weight, rank=8, init_std=0.01)
output = lora_linear(x, frozen_weight, adapter, alpha=16.0)
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT` | MAX device memory budget |
| `MODULAR_MAX_SHM_WATERMARK=0.9` | Shared memory watermark |
| `NABLA_DEBUG_CACHE=1` | Debug cache evaluation |

### Recommended Strategy for 3-4GB

1. Use `@nb.compile(max_cache_size=2)` on train step (forward + backward graphs)
2. Call `nb.GRAPH.clear_all()` between frames
3. Use 256x256 tiles (last resort) — smaller tensors in graph
4. If still over budget: pipeline parallelism (split model into 2-4 stages)
5. Scene-specific: QLoRA adapters instead of full model training

---

## Examples Reference (nablaml.com)

| # | Example | Key APIs | Relevance to Omen |
|---|---------|----------|-------------------|
| 01 | Tensors & Ops | `nb.Tensor`, arithmetic, reshape | Basic tensor bridge |
| 02 | Autodiff | `nb.grad`, `nb.value_and_grad`, `nb.jacrev` | Training gradient computation |
| 03 | Graph Tracing | Tracing, lazy evaluation, `realize()` | Understanding memory model |
| 04a | MLP Training (PyTorch-style) | `nn.Module`, `loss.backward()`, `optimizer.step()` | Imperative training pattern |
| 04b | MLP Training (JAX-style) | `value_and_grad`, `adamw_init/update`, pytrees | Functional training pattern |
| 05 | Transforms & `@nb.compile` | `vmap`, `grad`, `compile`, cache stats | Compilation and caching |
| 06a | Transformer (PyTorch-style) | `TransformerEncoderLayer`, `Embedding`, `MultiHeadAttention` | Attention-based model |
| 06b | Transformer (JAX-style) | Functional transformer, compiled step | Compiled attention training |
| 07 | Pipeline Parallelism (GPipe) | `DeviceMesh`, `ppermute`, micro-batches | **Memory-efficient staged compute** |
| 08 | 2D Parallel (PP + DP) | `DeviceMesh("2d")`, sharding, `DimSpec` | Multi-GPU scaling |
| 09 | Pipeline Parallel Inference | Staged inference with pipeline | Production inference pattern |
| 10 | Compiled vs Eager vs JAX | Benchmarks, `realize_all()` | Performance comparison |
| 11 | LoRA & QLoRA | `init_lora_adapter`, `lora_linear`, NF4 quantization | **Scene-specific adaptation** |
| 12 | Custom Mojo Kernels | `@compiler.register`, `foreach`, `UnaryOperation` | Custom GPU kernels |
| 13 | CNN Training | `conv2d`, `avg_pool2d`, `max_pool2d`, compiled CNN | Image processing architecture |

---

## Key APIs for Omen

### 1. Model Definition (PyTorch-style)

```python
import nabla as nb

class OmenJEPA(nb.nn.Module):
    def __init__(self, ...):
        self.encoder = nb.nn.Linear(in_dim, 192)
        self.ar_predictor = ...
        self.confidence_head = ...

    def forward(self, x):
        ...
```

### 2. Layers Available

| Layer | API | Notes |
|-------|-----|-------|
| Linear | `nb.nn.Linear(in, out)` | Full autograd |
| LayerNorm | `nb.nn.LayerNorm(dim)` | Pre-norm variant |
| Embedding | `nb.nn.Embedding(vocab, dim)` | Token embeddings |
| TransformerEncoderLayer | `nb.nn.TransformerEncoderLayer(d_model, num_heads, dim_ff)` | Pre-norm |
| MultiHeadAttention | `nb.nn.MultiHeadAttention(d_model, num_heads)` | `forward(q, k, v, mask)` |
| Dropout | `nb.nn.Dropout(p)` | Training/eval aware |
| Conv2d | `nb.conv2d(x, filter, stride, padding)` | NHWC, HWIO filters |
| Conv2d Transpose | `nb.conv2d_transpose(x, filter, ...)` | Deconv/upsample |
| Pool2d | `nb.avg_pool2d()`, `nb.max_pool2d()` | Strided pooling |

### 3. Functional API (JAX-style)

```python
import nabla.nn.functional as F

F.linear(x, weight, bias)
F.layer_norm(x, weight, bias, eps=1e-5)
F.scaled_dot_product_attention(q, k, v, mask)
F.mse_loss(pred, target)
F.cross_entropy_loss(logits, targets)
```

### 4. Optimizer

**Functional (JAX-style) — recommended for `@nb.compile`:**
```python
opt_state = nb.nn.optim.adamw_init(model)

@nb.compile(max_cache_size=2)
def train_step(model, opt_state, noisy, gt, sg):
    loss, grads = nb.value_and_grad(loss_fn, argnums=0)(model, noisy, gt, sg)
    model, opt_state = nb.nn.optim.adamw_update(model, grads, opt_state, lr=1e-4)
    return model, opt_state, loss
```

**Imperative (PyTorch-style) — for debugging:**
```python
optimizer = nb.nn.optim.AdamW(model, lr=5e-5, weight_decay=1e-3)
model.train()
model.zero_grad()
loss = model(inputs)
loss.backward()
model = optimizer.step()  # MUST reassign
```

### 5. Autograd Transforms

| Transform | API | Purpose |
|-----------|-----|---------|
| Reverse-mode | `nb.grad(fn)(x)` | d(loss)/d(weights) |
| Value + grad | `nb.value_and_grad(fn)(x)` | Forward + backward in one |
| vmap | `nb.vmap(fn, in_axes=0)` | Auto-batching |
| compile | `@nb.compile` | JIT to MAX graph (23-31x speedup) |

### 6. Custom Mojo Kernels

```python
from nabla.ops import UnaryOperation, call_custom_kernel
from max.graph import TensorType

class MyOp(UnaryOperation):
    @property
    def name(self) -> str:          # MUST be @property
        return "kernel_name"

    def kernel(self, args, kwargs): # NOT kernel(self, x, **kwargs)
        x = args[0]                 # args is list[TensorValue]
        out_type = TensorType(dtype=x.dtype, shape=output_shape, device=x.device)
        result = call_custom_kernel("kernel_name", kernel_dir, x, out_type)
        return [result]             # MUST return list

# Multi-input:
call_custom_kernel("name", dir, [t1, t2], out_type)  # list, not separate args

# Call site:
result = op([tensor], {})[0]       # NOT op(tensor, {})
```

Mojo side:
```mojo
@compiler.register("kernel_name")
struct MyKernel:
    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank],
        ctx: DeviceContextPtr,
    ):
        @parameter
        fn kernel_fn[width: Int](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return x.load[width](idx) + 1

        foreach[kernel_fn, target=target](output, ctx)
```

### 7. LoRA & QLoRA (Scene-Specific Adaptation)

```python
from nabla.nn.finetune import (
    init_lora_adapter,
    lora_linear,
    merge_lora_weight,
    save_finetune_checkpoint,
    load_finetune_checkpoint,
)

# LoRA: rank-r adapter on top of frozen weights
adapter = init_lora_adapter(base_weight, rank=8, init_std=0.01)
output = lora_linear(x, frozen_weight, adapter, alpha=16.0)

# QLoRA: frozen weights quantized to NF4 (4-bit), saves ~75% memory
save_finetune_checkpoint(path, lora_params=adapter, optimizer_state=opt_state)
```

### 8. Compilation Stats

```python
@nb.compile
def compiled_step(model, ...):
    ...

# After running:
print(compiled_step.stats)
# CompilationStats(hits=60, misses=1, fallbacks=0, hit_rate=98.4%)
```

### 9. Pipeline Parallelism

```python
from nabla.core.sharding import DeviceMesh, DimSpec
from nabla.ops import communication

mesh = DeviceMesh("pp", (4,), ("stage",))

# Split model into 4 stages, each with its own weights
# Stage-to-stage communication via ppermute
# Memory per stage = total / 4
```

### 10. Distributed Training

```python
mesh = DeviceMesh("2d", (2, 4), ("dp", "pp"))  # 2 DP x 4 PP = 8 GPUs
# Weights sharded on PP, data sharded on DP
```

---

## Tensor Interop

```python
# From numpy (Mitsuba renders to numpy)
tensor = nb.Tensor.from_dlpack(numpy_array)

# To numpy
array = tensor.to_numpy()

# Device transfer
tensor = tensor.cuda()  # GPU (when MAX detects GPU)
tensor = tensor.cpu()   # CPU
```

---

## Production Inference Path (Blender Addon)

```python
# Mojo .so loads in Blender's Python via ctypes
import ctypes
lib = ctypes.CDLL("lib/omen_denoise.so")

# Or compiled nabla inference
@nb.compile
def omen_denoise(model, noisy, scene_graph):
    return model(noisy, scene_graph)
```

Memory budget in production (RTX 3060 12GB):
- Renderer (Cycles/Eevee): ~6-8GB VRAM
- Denoiser (Mojo/nabla): ~2-4GB VRAM (forward pass only, no backward graph)
- Compiled inference: `@nb.compile` with `max_cache_size=1`

---

## What's NOT in Nabla (Needs Custom Implementation)

1. **SIGReg loss**: Write as custom op with Mojo kernel
2. **AdaLN-zero modulation**: Build from `nb.nn.Linear` + SiLU manually
3. **ConditionalBlock**: Compose from Linear, SiLU, LayerNorm
4. **Scene graph encoder**: Custom architecture using Linear + attention
5. **ConfidenceHead**: MLP (Linear → SiLU → Linear → SiLU → Linear → Sigmoid)
6. **Gradient clipping**: Manual with `tree_map` + norm
7. **CircularBuffer**: For temporal history management
8. **Gradient checkpointing**: Parameter exists but NOT implemented in current nabla

---

## Bottom Line

Nabla is the **one framework** for Omen. Training, inference, custom kernels, compilation,
LoRA, pipeline parallelism — all in one stack with Mojo efficiency.

Memory strategy for 3-4GB:
1. `@nb.compile(max_cache_size=2)` on train step
2. `nb.GRAPH.clear_all()` between frames
3. Pipeline parallelism if model graph is too large
4. QLoRA for scene-specific adaptation (4-bit weights + low-rank adapters)
5. 256x256 tiles as last resort

Compiled speedup: 23-31x over eager. Cache hit rate: ~98% after first compilation.
