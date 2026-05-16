## Context

Omen trains a JEPA denoiser that predicts noise residuals (gt - noisy) from 4spp renders. The training and inference pipeline is ONE system: user presses Render Animation, Omen renders a few purge frames at 256spp (training data), trains on the scene, then denoises all subsequent 4spp frames.

Current state: `OmenTrainer` feeds full images to `nb.value_and_grad` on CPU. Nabla defaults to CPU (`NABLA_DEFAULT_DEVICE` defaults to `"cpu"`). The `_GRAPH_CACHE` compiles a ~6-8GB MLIR module on first call and caches it — this is fine for fixed-resolution training (one entry, constant RAM) but untested at 4K. `gpu_budget.py` and `async_pipeline.py` exist for inference but are not imported by training code.

Hardware: 32GB RAM, 12GB VRAM (NVIDIA). A 4K (3840x2160) forward+backward pass through the U-Net decoder with skip connections would exceed 12GB VRAM.

## Goals / Non-Goals

**Goals:**
- Train on GPU using 512x512 tiles from 4K renders
- Encode scene graph once per full render, share latent across all tiles
- Fall back to CPU per-tile on GPU OOM
- Run training in tmux for IDE-independent long sessions
- Verify nabla cache stays at 1 entry for fixed tile size

**Non-Goals:**
- Patching nabla's `_GRAPH_CACHE` with LRU eviction (unnecessary for fixed tile size)
- Overlapping tiles with context margins (add later if boundary artifacts appear)
- Gradient accumulation across tiles (update per-tile is simpler and safer)
- Tile-based MoE routing (discarded — routing from scene graph knowledge, not pixels)
- Blender addon integration (future work — first make it work with pure Python+Mitsuba)
- Custom Mojo GPU kernels for training (blocked by MAX framework `_core.so` bug — all 6 kernels verified working on CPU)

## Decisions

### D1: Tiling at trainer level, NOT model level

The model (`OmenJEPA`) does not know about tiles. It receives (1, H, W, C) tensors where H=W=512. All tiling logic lives in `OmenTrainer.train_step_tiled()`.

**Why:** Model architecture stays clean. Existing inference path (full images) continues to work unchanged. Tile extraction is a training optimization, not a model concern.

**Alternative rejected:** Tiling in `TrainingDataGenerator` — would couple render logic to tile logic and prevent sharing scene latents.

### D2: Scene graph encoded once, shared across tiles

`SceneGraphEncoder` produces a (1, 1024) latent from the full scene's geometry/materials/lights. This is computed once per `train_step_tiled()` call and passed to all tiles via `_encode_with_scene_latent()`.

**Why:** Scene graph is per-scene, not per-tile. A 512x512 crop of a Cornell Box is still a Cornell Box — the 3D scene doesn't change per tile. Encoding once saves compute and is architecturally correct.

**Cost:** ~4KB latent tensor, broadcast to each tile — negligible VRAM.

### D3: Optimizer update per-tile (not gradient accumulation)

Each tile produces its own gradients and triggers an optimizer step immediately. Loss is averaged across tiles for reporting.

**Why:** Lower peak memory (don't accumulate 40 sets of gradients). Simpler error handling (OOM on tile 5 doesn't lose tiles 1-4 progress).

**Trade-off:** 40 optimizer steps per "full image" step instead of 1. But AdamW is cheap relative to the forward+backward pass.

### D4: GPU via NABLA_DEFAULT_DEVICE + .cuda(), not device context manager

Set `NABLA_DEFAULT_DEVICE=gpu` before importing nabla. Move input tensors to GPU with `.cuda()`. On OOM, catch RuntimeError and recreate tensors on CPU.

**Why:** Simplest approach. No device context manager complexity. nabla's own `Tensor.cuda()` handles the transfer.

**Alternative rejected:** Custom device context manager wrapping `nb.set_device` — more complex, not needed for single-GPU.

### D5: No overlap, no padding for edge tiles

Tiles at image boundaries may be smaller than 512x512 (e.g., 3840 % 512 = 256). These smaller tiles are processed as-is. nabla's cache will have a second entry for the smaller shape — acceptable since there are at most 2 unique tile shapes per dimension.

**Why:** Padding adds complexity. The model should generalize to slightly different sizes. If it doesn't, we add padding later.

## Risks / Trade-offs

**[RAM: nabla compilation on first tile]** → First `value_and_grad` call compiles ~6-8GB MLIR module. Subsequent tiles with same 512x512 shape are cache hits. Total RAM: ~8-10GB peak, constant. Acceptable on 32GB system.

**[VRAM: 512x512 exceeds 12GB]** → Unlikely (model is small), but if it happens, per-tile OOM fallback catches it. Mitigation: `gpu_budget.can_fit_tiles()` pre-checks before GPU launch.

**[RAM: second cache entry for edge tiles]** → Edge tiles (256px, 112px) trigger additional compilations. At most ~4 unique shapes → ~4 cache entries → ~32GB worst case on 32GB system. Mitigation: if this becomes a problem, pad edge tiles to 512x512 (revisit D5).

**[Training quality: per-tile optimizer steps]** → 40 gradient updates per full image instead of 1. May cause oscillation. Mitigation: lower learning rate for tiled training, or batch multiple tiles before update (future).

**[Training quality: no tile overlap]** → Boundary artifacts possible. Mitigation: inference uses full-image path, not tiles. Tiles only for training.

**[nabla GPU stability]** → nabla nightly may have GPU bugs. Mitigation: CPU fallback is always available. GPU path is opt-in via env var.
