## 1. Tile Extraction

- [x] 1.1 Create `src/omen/training/tile.py` with `Tile` dataclass (data, coords, is_edge) and `extract_tiles(image, tile_size=512)` function
- [x] 1.2 Add `tile_to_full(tiles, original_shape)` reconstruction function in tile.py
- [x] 1.3 Verify tile coordinate consistency: GT and noisy images produce identical tile coordinates

## 2. Scene Latent Reuse

- [x] 2.1 Add `_encode_with_scene_latent(self, scene_latent, rgba)` method to `OmenJEPA` in `src/omen/model/jepa.py`
- [x] 2.2 Refactor existing `encode()` to call `_encode_with_scene_latent` internally (backward compatible)

## 3. Loss Function Refactor

- [x] 3.1 Modify `compute_training_loss` in `src/omen/training/trainer/loss.py` to accept pre-encoded `scene_latent` instead of raw `scene_graph`
- [x] 3.2 Verify existing `train_step()` still works by calling `model.encode(scene_graph, noisy)` before passing latent to loss

## 4. Tiled Trainer

- [x] 4.1 Add `_encode_scene_once(self, scene_graph)` method to `OmenTrainer` in `src/omen/training/trainer/core.py`
- [x] 4.2 Add `train_step_tiled(self, noisy_full, gt_full, scene_graph, z_score=0.0)` method: extract tiles, encode scene once, loop tiles, aggregate loss
- [x] 4.3 Add `_train_single_tile(self, tile_noisy, tile_gt, scene_latent, z_score)` private method with per-tile optimizer update

## 5. GPU Training

- [x] 5.1 Add `estimate_tile_memory(tile_size)` and `can_fit_tiles(num_tiles, tile_size)` to `src/omen/gpu_budget.py`
- [x] 5.2 Wire GPU tensor creation in `train_step_tiled`: move tiles to GPU via `.cuda()` when GPU available
- [x] 5.3 Add per-tile OOM fallback: catch RuntimeError on each tile, reprocess on CPU, continue GPU for next tile

## 6. Config

- [x] 6.1 Add `tile_size: int = 512`, `gpu_fallback_enabled: bool = True`, `max_tiles_per_step: int = 64` to training config

## 7. Tests

- [ ] 7.1 Test tile extraction: 4K → 40 tiles, edge tiles smaller, exact multiple, image < tile_size
- [ ] 7.2 Test scene latent shared: mock SceneGraphEncoder, verify called once for 40 tiles
- [ ] 7.3 Test `train_step_tiled` with Cornell Box at 512x512 (1 tile) — GPU smoke test
- [ ] 7.4 Test `train_step_tiled` with Cornell Box at 1024x1024 (4 tiles) — multi-tile test
- [ ] 7.5 Test GPU OOM fallback: mock CUDA error, verify CPU fallback and recovery
- [ ] 7.6 Verify nabla cache size stays <= 2 after multi-tile training (512x512 + edge tiles)

## 8. Launcher

- [x] 8.1 Create `scripts/start_training.sh` — tmux session launcher with `NABLA_DEFAULT_DEVICE=gpu`
- [x] 8.2 Create `scripts/start_training.py` — CLI entry point with --scene, --resolution, --tile-size, --steps, --device args
