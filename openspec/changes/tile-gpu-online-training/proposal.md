## Why

Omen's training pipeline feeds full 4K images to the model entirely on CPU, causing a 6-8GB RAM bomb from nabla's `_GRAPH_CACHE` per compilation, zero GPU utilization, and inevitable OOM at production resolutions. The `gpu_budget.py` and `async_pipeline.py` modules already exist but are not wired into training. Real-time denoising requires training WHILE the user renders — the system must fit in 12GB VRAM using tile-based processing and fall back safely to CPU.

## What Changes

- Add 512x512 tile extraction from full-resolution renders (4K = 3840x2160 → 40 tiles)
- Encode scene graph ONCE per full render, share the (1, 1024) latent across all tiles — tiles are for VRAM savings only, not for routing
- Wire GPU training via `NABLA_DEFAULT_DEVICE=gpu` + `.cuda()` on input tensors
- Add per-tile GPU→CPU fallback on OOM (catch CUDA errors, continue on CPU)
- Wire existing `gpu_budget.py` into training with tile-aware memory estimation
- Add `_encode_with_scene_latent()` to OmenJEPA for scene latent reuse across tiles
- Modify loss function to accept pre-encoded scene latent instead of raw scene graph
- Add tmux launcher script for long-running training sessions

## Capabilities

### New Capabilities
- `tile-extraction`: 512x512 non-overlapping tile extraction from full-resolution images, with edge-tile handling and tile-to-full reconstruction
- `gpu-training`: GPU-accelerated training with per-tile VRAM management, CUDA OOM fallback to CPU, and nabla graph cache awareness
- `training-launcher`: tmux-based training session manager for safe long-running training independent of IDE

### Modified Capabilities
- `online-training`: Training pipeline now processes tiles instead of full images, shares scene graph latent across tiles, and uses GPU with CPU fallback

## Impact

- `src/omen/training/` — new tile.py, modified trainer/core.py, modified trainer/loss.py
- `src/omen/model/jepa.py` — new `_encode_with_scene_latent()` method
- `src/omen/gpu_budget.py` — new tile memory estimation functions
- `src/omen/training/online_gen.py` — GPU tensor output support
- `src/omen/config/` — new tile_size, gpu_fallback_enabled fields
- `tests/test_tile_training.py` — new comprehensive tile training tests
- `scripts/start_training.sh` — new tmux launcher
