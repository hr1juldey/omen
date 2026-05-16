## ADDED Requirements

### Requirement: Multiple optimizer steps per rendered frame
The training CLI SHALL support `--steps-per-frame N` to run N optimizer iterations on the same rendered GT+noisy pair.

#### Scenario: Multiple steps per frame
- **WHEN** user passes `--steps-per-frame 3`
- **THEN** for each rendered pair, `trainer.train_step_tiled()` is called 3 times with the same data, and the iteration counter increments by 3

#### Scenario: Default single step
- **WHEN** `--steps-per-frame` is not passed
- **THEN** one optimizer step per rendered frame (current behavior)

### Requirement: Steps-per-frame logging
Each sub-step SHALL log its step index within the frame and the loss.

#### Scenario: Sub-step logging
- **WHEN** running steps-per-frame=3 on frame 5
- **THEN** logger outputs "Frame 5, step 1/3: loss=X", "Frame 5, step 2/3: loss=Y", "Frame 5, step 3/3: loss=Z"

### Requirement: GT SPP default raised to 512
The `TrainingDataGenerator` default `gt_spp` SHALL be 512.

#### Scenario: Default SPP
- **WHEN** TrainingDataGenerator is created without explicit gt_spp
- **THEN** gt_spp defaults to 512

#### Scenario: Override SPP
- **WHEN** TrainingDataGenerator is created with `gt_spp=256`
- **THEN** GT renders at 256 SPP
