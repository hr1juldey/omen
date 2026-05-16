## Why

Training crashes (VSCode crash, shaderball bug) lose all progress because checkpoints only save at end of run. Multi-hour training needs crash recovery, progress visibility, and background execution.

## What Changes

- Fix `_tf()` to accept multiple rotations (fixes shaderball `rotate2` crash)
- Add periodic checkpointing: save every N steps + on scene transitions
- Add checkpoint rotation: keep last 3, delete older
- Add `--resume` flag to continue from latest checkpoint
- Add tmux launcher script: run training in detached session with log file
- Add `--log-file` flag to write training logs to file (for tmux tailing)

## Capabilities

### New Capabilities
- `periodic-checkpointing`: Save model weights every N steps, on scene transitions, and keep rotating checkpoints
- `tmux-training`: Launch training in tmux session with log file for monitoring

### Modified Capabilities
<!-- No existing specs to modify -->

## Impact

- `src/omen/scenes.py`: Fix `_tf()` to chain multiple rotations via `*rotations` varargs
- `src/omen/training/trainer/core.py`: Add `save_checkpoint_rotating()`, checkpoint every N steps
- `scripts/start_training.py`: Add `--resume`, `--checkpoint-every`, `--log-file` flags, periodic saves in training loop
- `scripts/train_tmux.sh` (NEW): tmux launcher with log file
