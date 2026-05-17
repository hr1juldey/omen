## ADDED Requirements

### Requirement: Compiled train step function
The system SHALL provide a function decorated with `@nb.compile` that contains the complete training step: forward pass, backward pass, gradient computation, and per-component optimizer updates. It SHALL return `(new_params, new_states, loss)` with no side effects.

#### Scenario: First call compiles the graph
- **WHEN** `compiled_train_step` is called for the first time with specific tensor shapes
- **THEN** the MAX compiler SHALL trace and compile the entire forward+backward+optimizer graph, taking up to 350s and peaking at ~25GB RAM

#### Scenario: Subsequent calls hit cache
- **WHEN** `compiled_train_step` is called again with the same tensor shapes
- **THEN** the compiled graph SHALL be reused (cache hit) executing in ~5ms with 0 additional RAM

#### Scenario: Stats show cache hits
- **WHEN** `compiled_train_step.stats` is inspected after N calls
- **THEN** it SHALL show `misses=1, hits=N-1` (one compilation, N-1 cache hits)

### Requirement: Per-component AdamW updates inside compiled function
The compiled train step SHALL call `adamw_update` once per optimizer component (encoder, decoder, shared_expert, material_experts, light_experts, geometry_experts, motion_experts, ar_predictor, episodic_correction). Each call SHALL use its own constant `lr` and `weight_decay`.

#### Scenario: Different LRs per component
- **WHEN** the compiled step runs
- **THEN** encoder params SHALL be updated with lr=5e-5, episodic params with lr=2e-2, and each component with its configured constant LR

#### Scenario: All 139 weight tensors updated
- **WHEN** the compiled step completes
- **THEN** all 139 weight tensors in the returned `new_params` dict SHALL have different values from the input `params`

### Requirement: Stable RAM after warmup
After the one-time compilation warmup, the training loop SHALL NOT increase RAM. The compiled graph is reused, not recreated. No `clear_all()` calls between steps.

#### Scenario: RAM stays flat for 100 steps
- **WHEN** 100 consecutive compiled train steps are executed
- **THEN** system RAM SHALL NOT increase more than 500MB from the post-warmup baseline

#### Scenario: No OOM kill
- **WHEN** training runs for 10,000 steps
- **THEN** the process SHALL NOT be OOM-killed — RAM stays under 15GB steady state

### Requirement: Loss convergence
The compiled training loop SHALL produce decreasing loss over steps, proving that gradient flow through the compiled graph is correct and optimizer updates are applied.

#### Scenario: Loss decreases over 100 steps
- **WHEN** 100 compiled train steps are executed on Cornell Box data
- **THEN** the loss at step 100 SHALL be measurably lower than the loss at step 1

#### Scenario: Loss values are unique
- **WHEN** loss values are collected over 10+ steps
- **THEN** more than 1 unique loss value SHALL exist (model is learning, not frozen)

### Requirement: Tiled training with compiled step
The compiled train step SHALL work with tiled input (256x256 tiles from larger images). The compiled graph SHALL handle variable tile counts because the tile size determines tensor shapes — same tile size = same shapes = cache hit regardless of source image size.

#### Scenario: 512x512 image split into 4 tiles
- **WHEN** a 512x512 image is split into 4 tiles of 256x256
- **THEN** each tile SHALL hit the compiled graph cache (same 256x256 shapes) and execute without recompilation

### Requirement: Checkpoint save and resume
The system SHALL support saving and loading checkpoints of the params dict and optimizer states. Checkpoint I/O happens OUTSIDE the compiled function.

#### Scenario: Save and resume training
- **WHEN** training is stopped after step N and resumed from checkpoint
- **THEN** the resumed loss at step N+1 SHALL be consistent with the trend before interruption
