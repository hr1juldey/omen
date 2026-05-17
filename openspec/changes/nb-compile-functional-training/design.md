## Context

Omen's training pipeline uses nabla's `value_and_grad` in eager mode with `clear_all()` after each tile. Each eager call creates a NEW compiled graph entry (~6-8GB). Destroying the cache forces recompilation: 262s and 20GB peak RAM per tile. At 4+ tiles, total RAM exceeds 31GB → OOM kill.

The previous `@nb.compile` attempt (May 16) froze model weights because `model.load_state_dict(params)` is a Python side effect — traced once, skipped on cache hits. This design follows nabla's official JAX-style pattern (example 5, example 13) where the entire train step is a pure function of params, grads, and optimizer state.

Current architecture: 8M params across 139 weight tensors, 9 optimizer components with different LRs, tiled training at 256x256 tiles. Model sub-modules: SceneGraphEncoder, RenderFeatureEncoder, CrossAttentionFusion, Decoder (U-Net), ConfidenceHead, ARPredictor, SIGRegLoss, EpisodicCorrection.

## Goals / Non-Goals

**Goals:**
- Eliminate RAM bomb: single `@nb.compile` entry, reused for all tiles and all steps
- Enable overnight training: 10,000+ steps at ~5ms/step after one-time 300s warmup
- Preserve per-component LRs: 9 components each with their own constant learning rate
- Preserve physics-based SIGReg loss inside compiled function
- Safe GPU backward: use sigmoid_gpu/silu_gpu for scalar-free VJP chains

**Non-Goals:**
- LR scheduling (cosine decay, warmup) — Phase 2, requires passing LR as 0-D tensor or computing inside compiled function
- GPU execution — Phase 2, first validate on CPU with @nb.compile, then add `device=Accelerator()`
- Mojo custom kernel integration — existing kernels (conv2d_im2col, ssim) remain as fallbacks
- Model architecture changes — same 8M param model, same sub-modules, same loss structure
- Refactoring core.py under 100-line CLAUDE_POLICY limit — deferred until training works

## Decisions

### D1: Functional forward pass pattern

**Decision**: Create `pure_loss_fn(params, noisy, gt, scene_latent)` that extracts weights from the flat params dict and applies operations directly — no `model.load_state_dict`, no `self.xxx`.

**Rationale**: This is the ONLY pattern that works with `@nb.compile` + `value_and_grad`. Nabla's examples (ex05, ex13) all use this pattern. The `model.load_state_dict` approach is PyTorch-style imperative, incompatible with compiled functional training.

**Implementation**: Each sub-module gets a companion `_forward_fn(params_prefix, *inputs)` static method or standalone function that mirrors the Module's `__call__` but takes params dict directly. The top-level `pure_loss_fn` chains these.

**Alternative considered**: Modify `nn.Module.__call__` to accept params override. Rejected — too invasive to nabla internals, and nabla Module is designed for imperative use.

### D2: Per-component optimizer inside @nb.compile

**Decision**: Multiple `adamw_update` calls inside the compiled function, one per component. Python loop is unrolled during tracing. Each call has its own constant `lr` keyword argument.

**Rationale**: The `adamw_update` function is a pure nabla function that works inside `@nb.compile` (proven in nabla examples). The component loop produces separate graph ops per component. Constant LRs are baked into the graph — cache hits work because tensor shapes/dtypes don't change.

**Why not single adamw_update**: Different components need different LRs (encoder=5e-5, episodic=2e-2). Single call would force uniform LR. Per-component calls maintain the learning rate diversity.

**Alternative considered**: Custom optimizer that takes LR tensor. Rejected for Phase 1 — adds complexity, untested with `@nb.compile`.

### D3: Constant LR (no scheduling)

**Decision**: Use fixed per-component LRs. No cosine decay, no warmup.

**Rationale**: Nabla's cache key includes float values (verified in `compile.py:_build_cache_key`). Changing `lr` float between calls creates a new cache key → full recompilation. 10,000 different LR values = 10,000 compilations. Constant LR = 1 compilation.

**Phase 2 path**: Pass LR as 0-D tensor (`shape=()`, `dtype=f32`) — always same cache key regardless of value. Requires `adamw_update` to accept tensor `lr` (currently takes `float`), or manual AdamW math inside compiled function.

### D4: sigmoid_gpu / silu_gpu for safe backward

**Decision**: Use `sigmoid_gpu(x) = 1/(1+exp(-x))` and `silu_gpu(x) = x * sigmoid_gpu(x)` in the functional forward pass.

**Rationale**: Nabla's Python VJPs for sigmoid/silu create `sub(1.0, tensor)` → `ensure_tensor(1.0)` → CPU scalar → mixed device error on GPU. Even inside `@nb.compile`, the Python VJPs run during tracing. The decomposed versions use only exp/neg/add/div/mul — all have scalar-free VJPs.

**Alternative considered**: Trust `@nb.compile` to handle constants properly. Rejected — untested, and safe activations add zero overhead (same mathematical result, same number of ops).

### D5: Optimizer state as pytree input to compiled function

**Decision**: Pass per-component optimizer states as a dict of pytrees `{name: {"m": pytree, "v": pytree, "step": 0}}`. The compiled function receives and returns these.

**Rationale**: `adamw_init` returns `{"m": pytree, "v": pytree, "step": 0}`. Inside `@nb.compile`, the step becomes a 0-D tensor. The state pytree structure must match between calls for cache hits — which it does since param names/shapes don't change.

### D6: File organization (CLAUDE_POLICY compliant — 100 LOC/file max)

**Decision**: Split functional code into focused files, each under 100 LOC:

```
src/omen/model/functional/           # Functional forward passes (one per sub-module)
├── __init__.py                      # Exports pure_loss_fn
├── scene_encoder.py                 # scene_encoder_fn(params, scene_graph)
├── render_encoder.py                # render_encoder_fn(params, rgba)
├── cross_attn.py                    # cross_attn_fn(params, render_lat, scene_lat)
├── decoder.py                       # decoder_fn(params, latent, noisy_image)
├── sigreg.py                        # sigreg_fn(predicted_latent, config) — 0 params
├── confidence.py                    # confidence_fn(params, latent, h, w)
├── ar_predictor.py                  # ar_predictor_fn(params, history, current, delta)
└── episodic.py                      # episodic_fn(params, latent)

src/omen/training/trainer/           # Compiled training
├── compiled_step.py                 # @nb.compile decorated function (~80 LOC)
├── compiled_trainer.py              # CompiledOmenTrainer class (~90 LOC)
├── pure_loss.py                     # pure_loss_fn chaining sub-modules (~60 LOC)
├── loss.py                          # existing — kept for eager fallback
└── core.py                          # existing — kept for eager fallback
```

**DRY**: LR constants imported from `omen.training.trainer.optimizers.COMPONENT_LRS` — single source of truth. No magic numbers.

**Rationale**: Clean separation — each file has one responsibility, stays under 100 LOC. Existing code preserved for eager mode/testing. Compiled path is opt-in. Can remove old code after compiled path is proven.

## Risks / Trade-offs

**[Compilation OOM]** → First compilation peaks at 20-25GB RAM. On 32GB system with ~8GB for OS/other, this leaves ~7GB headroom. If compilation exceeds 25GB, reduce tile size to 128x128 (smaller activation tensors, smaller graph).

**[Compilation time]** → 300-350s one-time warmup. Acceptable for 10,000+ step training sessions. Not suitable for interactive development — use eager mode for debugging.

**[Wrong functional forward output]** → If functional forward doesn't exactly match the Module's forward, gradients will be wrong. Mitigation: test that `pure_loss_fn(params, ...)` produces identical output to `compute_training_loss(params, model, ...)` for the same inputs.

**[Per-component LR ordering]** → If component loop order changes between steps, the compiled graph would be invalidated. Mitigation: use fixed `sorted(components.keys())` iteration.

**[Future LR scheduling breaks cache]** → Cosine decay requires either recompilation every N steps or passing LR as tensor. Phase 2 decision needed. Current constant LR is sufficient for initial convergence from random init.

## Open Questions

1. Should we run Phase 1 on CPU or GPU? CPU avoids device issues but is slower. GPU is faster but backward VJP bugs may still exist inside `@nb.compile`.
2. Should we keep the eager fallback path in `start_training.py`, or remove it entirely?
3. What tile size to use for initial validation? 256x256 (current) or smaller?
