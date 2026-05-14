## MODIFIED Requirements

### Requirement: JEPABridge denoise method
The `JEPABridge.denoise()` method SHALL support both inference-only mode (when model is pretrained) and post-training mode (when model was just trained). It SHALL also handle the case where no model is available by returning the raw input.

#### Scenario: Denoise after training
- **WHEN** `bridge.denoise(rgb, aux, scene_graph)` is called after a training step
- **THEN** use the model's current weights (which may have just been updated)
- **AND** encode noisy render + scene graph → latent
- **AND** decode latent → clean RGBA
- **AND** return denoised image as numpy array

#### Scenario: Denoise with no model
- **WHEN** `bridge.denoise(rgb, aux, scene_graph)` is called but `model_available` is False
- **THEN** return the input `rgb` unchanged
- **AND** log "Returning raw render (no model available)"

#### Scenario: Denoise quality check
- **WHEN** denoised output is produced
- **THEN** compute SSIM between denoised and noisy input
- **AND** if SSIM < 0.5 (denoising made it worse), return raw noisy render instead
- **AND** log "Denoising degraded quality, returning raw render"

## ADDED Requirements

### Requirement: JEPABridge SHALL expose checkpoint save
`JEPABridge` SHALL provide a `save_checkpoint()` method that persists model weights, optimizer state, and iteration count.

#### Scenario: Save checkpoint on demand
- **WHEN** `bridge.save_checkpoint()` is called
- **THEN** save model state_dict to `~/.omen/checkpoints/latest.omen`
- **AND** include optimizer state (AdamW m/v moments)
- **AND** include iteration count and architecture hash
- **AND** include any active LoRA adapter weights
- **AND** log "Checkpoint saved at iteration {N}"

### Requirement: JEPABridge SHALL expose scene-specific LoRA management
`JEPABridge` SHALL provide methods to initialize LoRA adapters for a specific scene and save/load scene-specific adapter weights.

#### Scenario: Initialize LoRA for scene
- **WHEN** `bridge.init_lora(scene_hash, rank=8)` is called
- **THEN** freeze base model parameters
- **AND** add LoRA adapters (rank=8) to encoder layers
- **AND** create new optimizer for LoRA parameters only
- **AND** log "LoRA adapters initialized for scene {hash}"

#### Scenario: Save scene-specific checkpoint
- **WHEN** `bridge.save_checkpoint(scene_hash=hash)` is called with a scene hash
- **THEN** save LoRA adapter weights to `~/.omen/checkpoints/{hash}.omen`
- **AND** keep base model weights in `latest.omen` unchanged
