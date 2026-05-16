## ADDED Requirements

### Requirement: Periodic checkpoint saves during training
The training loop SHALL save checkpoints every N optimizer steps, configurable via `--checkpoint-every N`.

#### Scenario: Checkpoint every 50 steps
- **WHEN** user passes `--checkpoint-every 50`
- **THEN** a checkpoint is saved at iterations 50, 100, 150, etc. to `~/.cache/omen/checkpoints/`

### Requirement: Checkpoint on scene transitions
The curriculum training loop SHALL save a checkpoint after each scene is mastered, before flushing the graph cache.

#### Scenario: Scene transition save
- **WHEN** cornell training completes and foggy training is about to start
- **THEN** a checkpoint is saved as `scene_cornell_iter_N.omen`

### Requirement: Rotating checkpoint history
The system SHALL keep the last 3 checkpoints and delete older ones.

#### Scenario: Rotation with 4th checkpoint
- **WHEN** checkpoint `step_200.omen` is saved and `step_50.omen`, `step_100.omen`, `step_150.omen` already exist
- **THEN** `step_50.omen` is deleted, keeping only the 3 newest

### Requirement: Resume from checkpoint
The CLI SHALL support `--resume` flag to load the latest checkpoint and continue training from the saved iteration.

#### Scenario: Resume training
- **WHEN** user passes `--resume` and a checkpoint exists at iteration 100
- **THEN** model weights and optimizer state are loaded, training continues from iteration 101

### Requirement: Fix _tf multi-rotation support
The `_tf()` helper in scenes.py SHALL accept multiple rotations via `*rotations` varargs.

#### Scenario: Two rotations
- **WHEN** `_tf(translate=[0,5,0], rotate=([1,0,0],90), rotate2=([0,1,0],45), scale=[3,3,1])` is called
- **THEN** the transform chains both rotations correctly without TypeError
