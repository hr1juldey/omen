## Why

Omen's training pipeline trains on single scenes with hardcoded parameters (GT SPP=256, 1 step per frame, Cornell-only animations). For the model to generalize across scenes and lighting conditions, it needs multi-scene, multi-frame, multi-step training at production resolution (2K) with higher-fidelity ground truth (SPP=512).

## What Changes

- **BREAKING**: GT SPP default raised from 256 to 512 for cleaner ground truth supervision
- Add `--steps-per-frame` CLI parameter: multiple optimizer updates per rendered frame
- Add `--scenes all` mode: cycle through all scenes in SCENE_REGISTRY per training run
- Add generic animation dispatcher: route any scene to its corresponding `_animations()` generator (not just Cornell)
- Add LR warmup schedule: linear warmup over first N steps, then cosine decay
- Wire all 5 scenes' animation generators into the training loop (veach, shaderball, studio, foggy already exist in scenes.py)

## Capabilities

### New Capabilities
- `multi-scene-loop`: Orchestrates training across multiple scenes with per-scene scene graph encoding and tile extraction
- `animation-dispatcher`: Generic dispatcher that maps scene name → animation generator, replacing hardcoded `cornell_animations` import
- `lr-scheduling`: Linear warmup + cosine decay learning rate schedule with configurable warmup steps
- `steps-per-frame`: Multiple optimizer iterations on the same rendered frame before advancing

### Modified Capabilities
<!-- No existing specs to modify -->

## Impact

- `scripts/start_training.py`: New CLI args (--steps-per-frame, --lr-warmup, --scenes), multi-scene loop, generic animation dispatch
- `src/omen/training/online_gen.py`: GT SPP default 256→512
- `src/omen/training/trainer/core.py`: LR warmup schedule integration
- `src/omen/scenes.py`: No changes needed — all 5 animation generators already exist
- Training time increases proportionally with GT SPP=512 (render time ~2x) and steps-per-frame
