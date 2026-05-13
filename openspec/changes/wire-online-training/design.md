## Context

Omen has a complete JEPA model architecture (~8M params) with Nabla-based training:
- `OmenTrainer` (trainer.py) — full forward/backward/optimizer step loop
- `data_gen` — generates noisy/clean pairs from Mitsuba renders (4spp vs 256spp)
- `checkpoint.py` — save/load model weights + optimizer state
- `OmenJEPA.compute_loss()` — JEPA latent prediction loss + SIGReg

But the render pipeline (`denoiser.py` → `jepa_bridge.py`) only does inference. It loads a
pretrained model and runs encode→decode. No training ever happens.

The user's vision: **no external dataset**. The system learns from the scenes it renders.
Progressive rendering naturally produces training signal — early batches are noisy, the
accumulated result is the target.

Current data flow:
```
sync → build_scene → render_denoiser(scene, bridge, spp)
  → _render_with_aov(scene, spp=4)  [noisy]
  → bridge.denoise(noisy_rgb, aux, scene_graph)  [inference only]
  → return clean RGBA
```

Target data flow:
```
sync → build_scene → render_denoiser(scene, bridge, spp)
  → if no checkpoint: bridge.initialize_model()  [bootstrap]
  → _render_with_aov(scene, spp=4)  [noisy input]
  → _render_with_aov(scene, spp=256)  [pseudo-GT target]
  → bridge.train_step(noisy, gt, scene_graph)  [online learning]
  → bridge.denoise(noisy_rgb, aux, scene_graph)  [now smarter]
  → return clean RGBA
  → checkpoint.save() after N steps
```

## Goals / Non-Goals

**Goals:**
- Wire `OmenTrainer.train_step()` into the denoiser render loop
- Initialize JEPA model from scratch when no checkpoint exists
- Use progressive render batches as self-supervised training signal
- Trigger LoRA fine-tuning after 3+ renders of same scene
- Persist learned weights via checkpoint system
- Graceful degradation: inference-only when Nabla unavailable

**Non-Goals:**
- Pre-training on external datasets (that's a separate workflow)
- Changing the JEPA model architecture
- Changing Mojo GPU kernel implementations
- Distributed/multi-GPU training
- UI for training progress in Blender viewport

## Decisions

### Decision 1: Training signal from progressive renders

**Choice**: Use `data_gen.generate_denoiser_pair(scene, spp_noisy=4, spp_gt=256)`
within `denoiser.py` before the denoising step.

**Rationale**: Same Mitsuba scene, same camera. The 4spp render is the noisy input,
the 256spp render is the pseudo-ground-truth. No external data needed. JEPA learns
to predict the clean latent from the noisy latent.

**Alternative**: Train on final accumulated result only. Rejected — requires waiting
for full render before training, no immediate benefit.

### Decision 2: Training phase placement in render pipeline

**Choice**: Add training in `denoiser.py` BEFORE denoising. Render training pair,
train, then denoise the actual output.

**Rationale**: Model improves before it denoises the real frame. Even 1-2 training
steps per render make the model better for the current scene.

**Alternative**: Background training thread. Rejected — adds complexity, race
conditions with model weights. Keep it synchronous for now.

### Decision 3: Model bootstrap when no checkpoint exists

**Choice**: `JEPABridge.__init__()` checks for checkpoint. If none found, creates
fresh `OmenJEPA()` with random weights and runs initial training on first render.

**Rationale**: "If model is not there, make one from the scene given." The model
starts random but learns immediately from the first scene rendered.

### Decision 4: Per-scene LoRA fine-tuning

**Choice**: After 3+ renders of the same scene (detected via scene topology hash
from `scene_cache.py`), initialize LoRA adapters (rank=8) on encoder weights and
fine-tune 50 iterations.

**Rationale**: Matches the design doc plan. LoRA is lightweight (rank=8 means ~0.5%
extra parameters). Frozen base model, only adapter weights update per-scene.

### Decision 5: Checkpoint cadence

**Choice**: Save checkpoint every 10 training steps and on session close.

**Rationale**: Frequent enough to not lose much work on crash. Not so frequent it
bottlenecks rendering.

## Risks / Trade-offs

- **[Training slows rendering]** → Mitigation: training runs at reduced resolution
  (e.g., 480x270) regardless of final render resolution. Training cost is fixed ~0.5s.
- **[Random model produces worse results than raw Mitsuba]** → Mitigation: compare
  SSIM between denoised and noisy. If denoised is worse, return raw noisy render.
- **[Nabla not installed]** → Mitigation: graceful fallback. Skip training, skip
  denoising, return raw Mitsuba render. Log warning once.
- **[Overfitting to single scene]** → Mitigation: LoRA adapters are per-scene.
  Base model weights frozen after initial training. LoRA rank=8 limits capacity.
- **[Checkpoint disk growth]** → Mitigation: keep only latest checkpoint + last 3
  scene-specific LoRA adapters. Auto-cleanup old checkpoints.
