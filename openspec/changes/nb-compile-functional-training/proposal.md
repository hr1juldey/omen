## Why

Eager `value_and_grad` creates a NEW compiled graph entry (6-8GB) every tile. With `clear_all()` destroying the cache after each tile, every tile recompiles from scratch — 262s and 20GB peak RAM per compilation. At 512x512 (4 tiles), the second tile's compilation pushes total RAM past 31GB, triggering the Linux OOM killer. The MAX compiler's ~17-20GB peak RAM per compilation is sustainable ONCE, but fatal when repeated. Nabla's `@nb.compile` compiles the entire forward+backward+optimizer into ONE graph entry, reused on cache hits (~5ms/step). The previous `@nb.compile` attempt failed because `model.load_state_dict(params)` is a Python side effect only executed during the first trace — model weights froze on cache hits. This proposal restructures training to use nabla's official JAX-style pattern: pure functional forward pass with direct params dict usage, no `load_state_dict` inside the compiled function.

## What Changes

- **BREAKING**: Replace `compute_training_loss` (which calls `model.load_state_dict`) with `pure_loss_fn` that uses params dict directly — no model object mutation inside compiled code
- **BREAKING**: Wrap entire train step (forward + backward + per-component optimizer) inside `@nb.compile` — removes `clear_all()` and eliminates recompilation
- **BREAKING**: Replace cosine decay LR schedule with constant per-component LRs (Python floats in cache key trigger recompilation — Phase 1 uses constants, scheduling is Phase 2)
- Remove `_realize_params`, `_realize_optimizer_state`, `_realize_grads` — lazy tensor chain management is unnecessary inside `@nb.compile` (the compiled graph handles all tensor lifecycle)
- Remove `_run_tile` eager fallback path — single compiled path only
- Use `sigmoid_gpu` / `silu_gpu` from `activations.py` in functional forward pass for GPU-safe backward VJPs
- Per-component `adamw_update` calls inside `@nb.compile` — Python loop over 9 components unrolled during tracing, each with its own constant LR

## Capabilities

### New Capabilities
- `functional-forward-pass`: Pure functional forward pass using flat params dict directly — no nn.Module state mutation. Covers scene encoder, render encoder, cross attention, decoder, SIGReg physics loss.
- `compiled-train-step`: Single `@nb.compile`-wrapped function containing forward + backward + per-component AdamW optimizer updates. Returns new params and optimizer state (no side effects).

### Modified Capabilities

## Impact

- **Core trainer** (`src/omen/training/trainer/core.py`): Major rewrite — replace eager `_run_tile` with compiled step, remove lazy tensor management
- **Loss function** (`src/omen/training/trainer/loss.py`): Replace `compute_training_loss` with `pure_loss_fn` — no model.load_state_dict
- **Model sub-modules** (`src/omen/model/`): Each module (SceneGraphEncoder, RenderFeatureEncoder, CrossAttentionFusion, Decoder, etc.) needs a functional version that accepts params dict
- **Optimizers** (`src/omen/training/trainer/optimizers.py`): Restructure optimizer state to match flat params dict pytree structure expected by `adamw_update` inside `@nb.compile`
- **Training loop** (`scripts/start_training.py`): Update to call compiled step, remove per-tile RAM guard (RAM stays flat after warmup)
- **RAM profile**: 20-25GB one-time peak during first compilation, then ~10GB steady state for all subsequent steps
- **Compilation**: ~300-350s one-time warmup, then ~5ms/step for 10,000+ steps
