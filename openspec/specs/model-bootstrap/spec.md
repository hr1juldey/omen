## ADDED Requirements

### Requirement: JEPABridge SHALL initialize model from scratch when no checkpoint exists
When `JEPABridge` is constructed and no checkpoint file is found at `~/.omen/checkpoints/latest.omen`, it SHALL create a fresh `OmenJEPA()` model with random weights and a new `OmenTrainer`. The model is ready to train on the first render.

#### Scenario: First-ever render with no checkpoint
- **WHEN** `JEPABridge.__init__()` is called and `~/.omen/checkpoints/latest.omen` does not exist
- **THEN** create directory `~/.omen/checkpoints/` if needed
- **AND** instantiate `OmenJEPA(latent_dim=192)` with random initialization
- **AND** instantiate `OmenTrainer(model, lr=5e-5)`
- **AND** set `self.model_available = True`
- **AND** log "Created fresh JEPA model (no checkpoint found)"

#### Scenario: Checkpoint exists
- **WHEN** `JEPABridge.__init__()` is called and checkpoint file exists
- **THEN** load model weights via `checkpoint.load()`
- **AND** restore optimizer state
- **AND** resume from saved iteration count
- **AND** log "Loaded checkpoint from iteration {N}"

#### Scenario: Nabla not installed
- **WHEN** `JEPABridge.__init__()` is called and Nabla cannot be imported
- **THEN** set `self.model_available = False`
- **AND** log warning "Nabla unavailable, AI pipeline disabled"
- **AND** all subsequent denoise/train calls return the input unchanged

### Requirement: JEPABridge SHALL expose train method
`JEPABridge` SHALL provide a `train_step(noisy_rgb, gt_rgb, scene_graph)` method that delegates to `OmenTrainer.train_step()`.

#### Scenario: Training step invoked
- **WHEN** `bridge.train_step(noisy, gt, scene_graph)` is called
- **THEN** convert numpy arrays to Nabla tensors via DLPack
- **AND** call `self.trainer.train_step(noisy, gt, scene_graph)`
- **AND** return loss dict with `total_loss`, `recon_loss`, `sigreg_loss` values

#### Scenario: Training step with model unavailable
- **WHEN** `bridge.train_step()` is called but `model_available` is False
- **THEN** return empty dict without error
