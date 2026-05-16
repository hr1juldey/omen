## Context

Omen's training pipeline (`scripts/start_training.py` + `online_gen.py` + `trainer/core.py`) currently handles single-scene training. Animation support exists but is hardcoded to Cornell (`from omen.scenes import cornell_animations`). All 5 scenes (cornell, veach, shaderball, studio, foggy) have both `build_*()` and `*_animations()` generators in `scenes.py`, but only Cornell's animation path is wired.

Current training loop: render GT@256spp + noisy@4spp → encode scene graph once → tile-based training → 1 optimizer step per render.

Tile size is 512x512, so 2K (2048x1080) produces 8x3=24 tiles with identical shape — nabla cache stays at 1 entry (~6-8GB). The LRU-3 eviction patch in nabla's engine.py caps RAM at ~11GB.

## Goals / Non-Goals

**Goals:**
- Train on all 5 scenes in a single training run
- Support multiple optimizer steps per rendered frame (reuse expensive render)
- Raise GT SPP to 512 for cleaner supervision signal
- Add LR warmup + cosine decay schedule
- Generic animation dispatch — any scene → its animation generator

**Non-Goals:**
- GPU training (nabla `.cuda()` has transfer_to bug — separate fix)
- Multi-GPU or distributed training
- Replay buffer or dataset caching (online generation is sufficient)
- Changing tile size or model architecture

## Decisions

### 1. Scene → animation dispatch via naming convention

**Decision**: Use `f"{scene_name}_animations"` to resolve the generator function from `scenes.py`.

**Rationale**: All 5 generators follow the same naming pattern. A dispatch dict is simpler than reflection and makes missing generators obvious at import time.

**Alternative considered**: Import all `*_animations` functions explicitly — rejected because adding a new scene requires touching the training script.

### 2. Multi-scene loop: round-robin with per-scene scene graph

**Decision**: Cycle through scenes in sorted order. Each scene gets its own `build_*()` → scene_graph → encode → tiles flow.

**Rationale**: Scene graphs differ per scene (different geometry, materials, lights). Round-robin ensures balanced exposure. The scene graph is encoded once per scene per round — tiles within a scene share the same latent.

### 3. Steps-per-frame: repeat optimizer, not repeat render

**Decision**: Render once (expensive), run `train_step_tiled()` N times on the same rendered pair with different dropout/noise augmentation.

**Rationale**: Rendering at SPP=512 takes seconds; optimizer step is milliseconds. Reusing the render amortizes cost. Each step uses the same data but gets fresh gradients from the model's changing state.

**Alternative considered**: Re-render with different seeds — rejected because SPP=512 render is the bottleneck.

### 4. LR schedule: warmup + cosine decay in trainer

**Decision**: Add `_compute_scheduled_lr()` method to `OmenTrainer` that replaces the current `_compute_lr()`. Linear warmup for first `warmup_steps`, then cosine decay to `min_lr`.

**Rationale**: Standard practice for vision transformers. Keeps existing surprise-modulation as a multiplier on top of the schedule.

### 5. GT SPP=512 in TrainingDataGenerator

**Decision**: Change default `gt_spp` parameter from 256 to 512 in `TrainingDataGenerator.__init__()`.

**Rationale**: Cleaner ground truth → cleaner residual signal → faster convergence. Render time increases ~2x but SPP=512 is still fast on GPU via `cuda_ad_rgb`.

## Risks / Trade-offs

- **[Risk] GT SPP=512 doubles render time** → Acceptable: GPU rendering via cuda_ad_rgb mitigates. Training is offline.
- **[Risk] Steps-per-frame may overfit on single frame** → Mitigation: Dropout in model + surprise-modulated LR naturally regularizes.
- **[Risk] Round-robin scene order biases toward alphabetically first scenes** → Low risk: with enough steps, all scenes get equal exposure.
- **[Risk] Scene with no animation generator (future scenes)** → Mitigation: Dispatch falls back to static multi-camera training if no generator found.
