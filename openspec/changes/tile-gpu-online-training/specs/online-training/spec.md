## MODIFIED Requirements

### Requirement: Training SHALL run at reduced resolution
Training pairs SHALL be rendered at full resolution and split into 512x512 tiles for training. The tile size ensures each training batch fits within 12GB VRAM. Edge tiles smaller than 512x512 are processed as-is. The system SHALL encode the scene graph once per full render and share the scene latent across all tiles.

#### Scenario: 4K render triggers tiled training
- **WHEN** the user renders at 3840x2160
- **THEN** training pairs SHALL be generated at 3840x2160
- **AND** the images SHALL be split into 512x512 tiles (40 tiles)
- **AND** scene graph SHALL be encoded once into a (1, 1024) latent
- **AND** each tile SHALL be trained with the shared scene latent
- **AND** the final denoised output SHALL be at the full 3840x2160 resolution

#### Scenario: 512x512 render is single tile
- **WHEN** the user renders at 512x512
- **THEN** training SHALL produce 1 tile containing the full image
- **AND** no tiling overhead SHALL be incurred

#### Scenario: 1024x1024 render produces 4 tiles
- **WHEN** the user renders at 1024x1024
- **THEN** training SHALL produce 4 tiles (2x2), all 512x512
- **AND** scene latent SHALL be shared across all 4 tiles

## ADDED Requirements

### Requirement: OmenTrainer SHALL support tiled training
The `OmenTrainer` class SHALL provide `train_step_tiled(noisy_full, gt_full, scene_graph)` that extracts tiles, encodes the scene graph once, trains on each tile, and aggregates loss across tiles.

#### Scenario: train_step_tiled produces valid training
- **WHEN** `train_step_tiled()` is called with (3840, 2160, 3) noisy and GT arrays
- **THEN** scene graph SHALL be encoded exactly once
- **AND** all 40 tiles SHALL be processed
- **AND** returned metrics SHALL include `total_loss`, `num_tiles`, and `iteration`
- **AND** model weights SHALL be updated after each tile

### Requirement: Loss function SHALL accept pre-encoded scene latent
The `compute_training_loss` function SHALL accept a pre-encoded `scene_latent` tensor instead of a raw `scene_graph` dict. This enables scene latent reuse across tiles without redundant encoding.

#### Scenario: Tile uses shared scene latent
- **WHEN** `compute_training_loss(params, model, tile_noisy, tile_gt, scene_latent, config)` is called
- **THEN** the loss function SHALL use the pre-encoded scene latent directly
- **AND** SHALL NOT call `SceneGraphEncoder` internally
