## ADDED Requirements

### Requirement: Denoiser pipeline SHALL train on every render
The `render_denoiser()` function in `modes/denoiser.py` SHALL invoke `OmenTrainer.train_step()` before denoising. It SHALL generate a training pair (4spp noisy, 256spp pseudo-GT) from the same Mitsuba scene, call `train_step(noisy, gt, scene_graph)`, and then proceed with denoising the actual render output.

#### Scenario: First render trains from scratch
- **WHEN** `render_denoiser()` is called with a scene and no model checkpoint exists
- **THEN** the pipeline SHALL render a 4spp noisy image and a 256spp pseudo-GT from the scene
- **AND** call `trainer.train_step(noisy, gt, scene_graph)`
- **AND** proceed to denoise the actual render using the now-trained model

#### Scenario: Subsequent renders continue training
- **WHEN** `render_denoiser()` is called and a model checkpoint exists
- **THEN** the pipeline SHALL load the checkpoint before training
- **AND** run one training step with a fresh noisy/GT pair (different seed)
- **AND** denoise the actual render using the updated model

#### Scenario: Nabla unavailable
- **WHEN** Nabla is not installed or import fails
- **THEN** the pipeline SHALL skip training and denoising entirely
- **AND** return the raw Mitsuba render without AI enhancement
- **AND** log a single warning "Nabla unavailable, skipping AI pipeline"

### Requirement: Training SHALL run at reduced resolution
Training pairs SHALL be rendered at a fixed reduced resolution (max 480x270) regardless of the final render resolution. This caps training cost at ~0.5s per render.

#### Scenario: 4K render triggers 480p training
- **WHEN** the user renders at 3840x2160
- **THEN** training pairs SHALL be generated at 480x270
- **AND** the final denoised output SHALL be at the full 3840x2160 resolution

### Requirement: LoRA fine-tuning after repeated scene renders
After the same scene (detected via topology hash from `scene_cache.py`) has been rendered 3+ times, the system SHALL initialize LoRA adapters (rank=8) on encoder weights and run 50 fine-tuning iterations using cached training pairs.

#### Scenario: Scene rendered 3 times triggers LoRA
- **WHEN** the same scene topology hash has been rendered 3 times
- **THEN** initialize LoRA adapters on the model's encoder weights with rank=8
- **AND** collect cached training pairs from previous renders
- **AND** run 50 fine-tuning iterations with AdamW (lr=5e-5)
- **AND** save LoRA adapter weights to scene-specific checkpoint
- **AND** log "Fine-tuned model for scene (hash: {hash}, 50 iters)"

#### Scenario: Different scene does not trigger LoRA
- **WHEN** a new scene is rendered (different topology hash)
- **THEN** use the base model without LoRA fine-tuning
- **AND** start counting renders for this new scene's LoRA trigger

### Requirement: Checkpoint persistence
The system SHALL save model checkpoints every 10 training steps and on session close.

#### Scenario: Checkpoint saved after 10 steps
- **WHEN** 10 training steps have completed since last save
- **THEN** save model weights to `~/.omen/checkpoints/latest.omen`
- **AND** include optimizer state (AdamW m/v moments)
- **AND** include iteration count
- **AND** log "Checkpoint saved at iteration {N}"

#### Scenario: Session close triggers save
- **WHEN** the render session is closing
- **THEN** save the current model state as checkpoint
- **AND** include all LoRA adapter weights if any exist
