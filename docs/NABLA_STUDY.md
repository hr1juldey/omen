# Nabla ML Framework — Study for Omen JEPA Engine

## Executive Summary

Nabla is a **complete deep learning framework** with Python API and Mojo/MAX backend.
It has everything Omen needs: `nn.Module`, `AdamW`, autograd, conv2d, attention,
custom Mojo kernels, LoRA, checkpointing, distributed training, and `@nb.compile`.

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

- `nb.nn.Module` with `forward()`, `parameters()`, `state_dict()`, `load_state_dict()`
- `model.train()` / `model.eval()` for mode switching

### 2. Layers Available

| Layer | API | Notes |
|-------|-----|-------|
| Linear | `nb.nn.Linear(in, out)` | Full autograd |
| LayerNorm | `nb.nn.LayerNorm(dim)` | Pre-norm variant |
| Embedding | `nb.nn.Embedding(vocab, dim)` | Token embeddings |
| TransformerEncoderLayer | `nb.nn.TransformerEncoderLayer(d_model, num_heads, dim_ff)` | Pre-norm |
| MultiHeadAttention | `nb.nn.MultiHeadAttention(d_model, num_heads)` | `forward(q, k, v, mask)` |
| Dropout | `nb.nn.Dropout(p)` | Training/eval aware |
| Conv2d | `nb.conv2d(x, filter, stride, padding)` | NHWC layout, HWIO filters |
| Conv2d Transpose | `nb.conv2d_transpose(x, filter, ...)` | Deconv/upsample |
| Pool2d | `nb.avg_pool2d()`, `nb.max_pool2d()` | Strided pooling |

### 3. Functional API (JAX-style alternative)

```python
import nabla.nn.functional as F

F.linear(x, weight, bias)
F.layer_norm(x, weight, bias, eps=1e-5)
F.scaled_dot_product_attention(q, k, v, mask)
F.embedding(ids, weight)
F.dropout(x, p=0.1)
F.gelu(x)
F.relu(x)
F.silu(x)
F.sigmoid(x)
```

### 4. Optimizer

**Stateful (PyTorch-style):**
```python
optimizer = nb.nn.optim.AdamW(model, lr=5e-5, weight_decay=1e-3)
# In loop:
model.train()
model.zero_grad()
loss = model(inputs)
loss.backward()
model = optimizer.step()  # MUST reassign (lazy execution)
```

**Functional (JAX-style):**
```python
opt_state = nb.nn.optim.adamw_init(model)
# In loop:
loss_val, grads = nb.value_and_grad(loss_fn)(model, data)
model, opt_state = nb.nn.optim.adamw_update(model, grads, opt_state, lr=5e-5)
```

### 5. Autograd Transforms

| Transform | API | Purpose |
|-----------|-----|---------|
| Reverse-mode | `nb.grad(fn)(x)` | d(loss)/d(weights) |
| Value + grad | `nb.value_and_grad(fn)(x)` | Forward + backward in one |
| Forward-mode | `nb.jacfwd(fn)(x)` | Jacobian forward |
| Reverse Jacobian | `nb.jacrev(fn)(x)` | Jacobian reverse |
| VJP | `nb.vjp(fn, *primals)` | Vector-Jacobian product |
| JVP | `nb.jvp(fn, primals, tangents)` | Jacobian-vector product |
| vmap | `nb.vmap(fn, in_axes=0)` | Auto-batching |
| compile | `@nb.compile` or `nb.compile(fn)` | JIT to MAX graph |

### 6. Custom Mojo Kernels

CRITICAL for Omen — merge kernels, edge-aware upsampling, SIGReg in Mojo:

```python
from nabla.ops import UnaryOperation, call_custom_kernel

class MergeOp(UnaryOperation):
    @property
    def name(self): return "omen_merge"

    def kernel(self, args, kwargs):
        result = call_custom_kernel("omen_merge", kernel_dir, *args)
        return [result]

    def _derivative(self, primals, output):
        return ...  # gradient rule

    # OR for non-elementwise:
    def vjp_rule(self, primals, output, ct):
        ...
    def jvp_rule(self, primals, tangents):
        ...
```

Mojo kernel side:
```mojo
@compiler.register("omen_merge")
struct MergeKernel:
    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor,
        low_res: InputTensor[dtype=output.dtype, rank=output.rank],
        high_res: InputTensor[dtype=output.dtype, rank=output.rank],
        ctx: DeviceContextPtr,
    ):
        foreach[merge_fn, target=target](output, ctx)
```

### 7. LoRA Fine-Tuning (Scene-Specific Adaptation)

Built-in LoRA/QLoRA for scene-specific model adaptation:

```python
from nabla.nn.finetune import (
    init_lora_adapter,
    lora_linear,
    merge_lora_weight,
    save_finetune_checkpoint,
    load_finetune_checkpoint,
)

adapter = init_lora_adapter(base_weight, rank=8, init_std=0.01)
output = lora_linear(x, frozen_weight, adapter, alpha=16.0)
save_finetune_checkpoint(path, lora_params=adapter, optimizer_state=opt_state)
loaded_adapter, loaded_opt, meta = load_finetune_checkpoint(path, lora_template=template)
```

### 8. Checkpointing

```python
state = model.state_dict()           # OrderedDict[str, Tensor]
model.load_state_dict(state)         # Load weights back

# LoRA-specific:
nb.nn.finetune.save_finetune_checkpoint(path, lora_params=..., optimizer_state=...)
loaded = nb.nn.finetune.load_finetune_checkpoint(path, lora_template=...)
```

### 9. Compilation for Inference

```python
@nb.compile
def inference_step(model, noisy_frame, scene_graph):
    model.training = False
    latent = model.encode(noisy_frame, scene_graph)
    clean = model.decode(latent)
    return clean

# OR with dynamic batch dim:
compiled = nb.compile(inference_step, dynamic_dims={0: {0: "batch"}})
```

### 10. Distributed Training

```python
from nabla.core.sharding import DeviceMesh, DimSpec, PartitionSpec as P
from nabla.transforms import shard_map

mesh = DeviceMesh("2d", (2, 4), ("dp", "pp"))  # 2 DP x 4 PP = 8 GPUs
@shard_map(mesh, in_specs={0: spec}, out_specs={0: out_spec})
def train_step(params, batch):
    ...
```

Pipeline primitives: `ppermute` for stage communication, `all_reduce`, `all_gather`.

## Architecture Implications for Omen

### What Nabla Gives Us (No Need to Build)

1. **Autograd**: `nb.grad` / `loss.backward()` — no custom backward needed
2. **AdamW optimizer**: `nb.nn.optim.AdamW` with lr, weight_decay, betas — matches lewm.yaml
3. **Attention**: `nb.nn.MultiHeadAttention` and `F.scaled_dot_product_attention`
4. **Transformer layers**: `nb.nn.TransformerEncoderLayer(d_model, num_heads, dim_ff)`
5. **Conv2d / Conv2d_transpose**: For encoder-decoder architectures
6. **LayerNorm, GELU, SiLU, Sigmoid**: All activations for ConditionalBlock
7. **LoRA**: Built-in for scene-specific fine-tuning
8. **Checkpointing**: `state_dict()` / `load_state_dict()` + LoRA-specific
9. **Compilation**: `@nb.compile` for fast inference
10. **Custom Mojo kernels**: For SIGReg, merge, geometry-aware ops

### Training Paradigm Decision

**Use PyTorch-style (imperative)** for Omen:
- Easier debugging during development
- Matches spec's `omen_train_step` flow
- Natural for variable-length scene graphs

```python
model.train()
optimizer = nb.nn.optim.AdamW(model, lr=5e-5, weight_decay=1e-3)

for iteration in range(500):
    model.zero_grad()
    predicted = model(noisy_render, scene_graph)
    pred_loss = nb.mean(nb.square(predicted - ground_truth))
    sigreg_loss = SIGReg(model.embeddings)
    total_loss = pred_loss + 0.09 * sigreg_loss
    total_loss.backward()
    model = optimizer.step()
```

### Inference Paradigm

**Use `@nb.compile`** for production:
```python
@nb.compile
def omen_denoise(model, noisy, scene_graph):
    return model(noisy, scene_graph)
```

### Custom Ops Needed for Omen

1. **SIGReg loss**: Epps-Pulley statistic (17 knots, 1024 projections) -> custom Mojo kernel
2. **Edge-aware merge**: Geometry-guided upsampling -> custom Mojo kernel
3. **Scene graph encoding**: Struct -> tensor encoding -> standard Nabla ops
4. **Gradient clipping**: Use `nb.tree_map` + norm computation before optimizer step

### Critical: Two Paradigms for Training

| Paradigm | Gradient | Optimizer | When to Use |
|----------|----------|-----------|-------------|
| PyTorch | `loss.backward()` + `.grad` | `AdamW(model)` -> `opt.step()` | Development |
| JAX | `nb.value_and_grad(fn)(...)` | `adamw_init` + `adamw_update` | Production |

Start PyTorch-style, migrate to JAX-style + `@nb.compile` for production.

## Tensor Interop (Bridge from Mitsuba)

```python
# From numpy (Mitsuba renders to numpy)
tensor = nb.Tensor.from_dlpack(numpy_array)

# To numpy
array = tensor.to_numpy()

# Device transfer
tensor = tensor.cuda()  # GPU
tensor = tensor.cpu()   # CPU
```

Bridge: Mitsuba renders -> numpy -> `from_dlpack()` -> Nabla tensor -> JEPA inference.

## What's NOT in Nabla (Needs Custom Implementation)

1. **SIGReg loss**: Write as custom op with Mojo kernel
2. **AdaLN-zero modulation**: Build from `nb.nn.Linear` + SiLU manually
3. **ConditionalBlock**: Compose from Linear, SiLU, LayerNorm
4. **Scene graph encoder**: Custom architecture using Linear + attention
5. **ConfidenceHead**: MLP (Linear -> SiLU -> Linear -> SiLU -> Linear -> Sigmoid)
6. **Gradient clipping**: Manual with `tree_map` + norm
7. **CircularBuffer**: For temporal history management

## Bottom Line

Nabla is **production-ready** for Omen. It has full autograd, AdamW, Transformer layers,
attention, conv2d, all activations, custom Mojo GPU kernels, LoRA, checkpointing,
JIT compilation, and distributed training.

Key insight: Since both Mitsuba (Dr.Jit) and Nabla are Python-callable, the bridge
can use **DLPack zero-copy** (`nb.Tensor.from_dlpack(dr_tensor)`) instead of raw
C pointers. This simplifies the architecture significantly.
