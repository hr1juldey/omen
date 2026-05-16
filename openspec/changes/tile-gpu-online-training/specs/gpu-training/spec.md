## ADDED Requirements

### Requirement: GPU training with nabla
The training pipeline SHALL execute forward pass, loss computation, gradient computation, and optimizer updates on GPU when available. The system SHALL set `NABLA_DEFAULT_DEVICE=gpu` before nabla import and move input tensors to GPU via `.cuda()`.

#### Scenario: GPU available and sufficient VRAM
- **WHEN** a CUDA GPU with >= 4GB free VRAM is detected
- **AND** `gpu_budget.can_fit_tiles(tile_size=512, num_tiles=1)` returns sufficient=True
- **THEN** input tensors SHALL be moved to GPU via `.cuda()`
- **AND** `nb.value_and_grad` SHALL execute on GPU
- **AND** gradient computation and AdamW update SHALL execute on GPU
- **AND** loss values SHALL be transferred to CPU for logging

#### Scenario: No GPU available
- **WHEN** no CUDA GPU is detected or `_has_accelerator()` returns False
- **THEN** training SHALL proceed on CPU without error
- **AND** a single info log SHALL note "No GPU detected, training on CPU"

### Requirement: Per-tile GPU OOM fallback
The training loop SHALL catch GPU out-of-memory errors on a per-tile basis. When a CUDA OOM error occurs, the failing tile SHALL be reprocessed on CPU and subsequent tiles SHALL attempt GPU first.

#### Scenario: OOM on tile 5 of 40
- **WHEN** GPU OOM occurs while processing tile 5 of 40
- **THEN** the system SHALL log "GPU OOM at tile 5, falling back to CPU"
- **AND** reprocess tile 5 on CPU
- **AND** attempt tile 6 on GPU again
- **AND** continue until all tiles are processed

#### Scenario: Persistent OOM on all tiles
- **WHEN** GPU OOM occurs on every tile attempt
- **THEN** the system SHALL fall back to CPU for all remaining tiles
- **AND** log a single warning "GPU training unstable, using CPU for remaining tiles"

### Requirement: Scene graph encoded once per full render
The scene graph (geometry, materials, lights) SHALL be encoded by `SceneGraphEncoder` exactly once per `train_step_tiled()` call. The resulting (1, 1024) scene latent SHALL be shared across all tiles from the same render.

#### Scenario: 40 tiles share one scene latent
- **WHEN** `train_step_tiled()` processes a 4K render with 40 tiles
- **THEN** `SceneGraphEncoder` SHALL be called exactly 1 time
- **AND** the scene latent SHALL be broadcast to all 40 tiles
- **AND** each tile SHALL combine its own render features with the shared scene latent

#### Scenario: Different scene produces different latent
- **WHEN** two different scenes are trained in sequence
- **THEN** each scene SHALL produce its own unique scene latent
- **AND** tiles from scene A SHALL NOT use scene B's latent

### Requirement: Nabla graph cache stays bounded
For fixed tile_size training, nabla's `_GRAPH_CACHE` SHALL contain at most 2 entries: one for 512x512 tiles and one for edge tiles (smaller dimensions).

#### Scenario: 40 tiles with 2 unique shapes
- **WHEN** `train_step_tiled()` processes 38 full 512x512 tiles and 2 edge tiles
- **THEN** `get_cache_stats()["size"]` SHALL be at most 2 after the step completes
- **AND** total RAM usage SHALL NOT exceed 16GB on a 32GB system

### Requirement: Tile-aware memory estimation
The `gpu_budget.py` module SHALL provide `estimate_tile_memory(tile_size)` and `can_fit_tiles(num_tiles, tile_size)` functions that compute VRAM requirements for tiled training.

#### Scenario: Estimate memory for 512x512 tile
- **WHEN** `estimate_tile_memory(tile_size=512)` is called
- **THEN** it SHALL return an estimate in MB accounting for RGBA input, gradients, optimizer state, and working memory
- **AND** `can_fit_tiles(num_tiles=1, tile_size=512)` SHALL compare the estimate against available VRAM
