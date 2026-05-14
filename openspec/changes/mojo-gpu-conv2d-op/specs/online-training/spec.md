## MODIFIED Requirements

### Requirement: All 139 params trainable (no frozen conv2d filters)
Previously 4 decoder conv2d filter params were frozen (`CONV2D_BLOCKERS`). With custom matmul-based vjp_rule, all conv2d params receive valid gradients.

#### Scenario: No params frozen in trainer
- **WHEN** `OmenTrainer` is initialized with full model
- **THEN** `CONV2D_BLOCKERS` is removed (deleted entirely)
- **AND** all 139 named parameters appear in optimizer components

#### Scenario: Conv2d filter gradients are non-zero
- **WHEN** `train_step` completes one iteration
- **THEN** gradients for `decoder.e1`, `decoder.e2`, `decoder.e3`, `decoder.e4` are non-zero
- **AND** `decoder.d1`, `decoder.d2`, `decoder.d3`, `decoder.d4` gradients are non-zero
- **AND** `scene_encoder.conv1_filter`, `conv2_filter`, `conv3_filter` gradients are non-zero

### Requirement: Standard gradient realization (no per-tensor workaround)
Previously `_realize_grads` used individual `.to_numpy()` per gradient tensor. With matmul-based vjp_rule producing traceable Nabla tensors, standard realization works.

#### Scenario: _to_real removed from optimizer path
- **WHEN** `_apply_optimizer_updates` runs
- **THEN** no `_to_real` calls on optimizer outputs
- **AND** AdamW outputs are used directly without realization hacks

#### Scenario: Three training steps produce different losses
- **WHEN** trainer runs 3 consecutive `train_step` calls on same data
- **THEN** all 3 loss values are finite
- **AND** at least 2 of 3 loss values differ (params are updating)

### Requirement: All 11 conv2d call sites replaced
Every `nb.conv2d` call in decoder and scene_encoder uses `conv2d_safe`.

#### Scenario: Decoder uses conv2d_safe
- **WHEN** `Decoder.forward` runs
- **THEN** all 8 conv2d operations (e1-e4 encoder, d1-d4 decoder) use `conv2d_safe`
- **AND** output shapes are identical to previous `nb.conv2d` behavior

#### Scenario: RenderFeatureEncoder uses conv2d_safe
- **WHEN** `RenderFeatureEncoder.forward` runs
- **THEN** all 3 conv2d operations (conv1-conv3) use `conv2d_safe`
- **AND** output shapes are identical to previous `nb.conv2d` behavior

### Requirement: 1024×1024 training tile support
Trainer processes 1024×1024 tiles with 47 AOV layers and mixed precision.

#### Scenario: Memory budget at 1024×1024
- **WHEN** train_step runs with 1024×1024 input
- **THEN** total GPU VRAM usage is under 2.25 GB (mixed precision)
- **AND** fits within RTX 3060 12 GB with headroom for batch or larger models

### Requirement: Live render+train pipeline
Mitsuba renders next tile on CPU while Nabla trains on current tile on GPU.

#### Scenario: Double-buffer overlap
- **WHEN** Mitsuba finishes rendering tile N to buffer A
- **THEN** buffer A is transferred to GPU (~200 MB, ~5ms over PCIe)
- **AND** Nabla begins training on buffer A while Mitsuba renders tile N+1 into buffer B
- **AND** GPU is idle only during the ~5ms transfer window

#### Scenario: Geometry node scene variation
- **WHEN** geometry nodes generate a new scene variation (camera, materials, lights)
- **THEN** Mitsuba renders it to the next available buffer
- **AND** the training loop sees diverse data, not temporal frames
