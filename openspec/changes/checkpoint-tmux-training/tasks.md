## 1. Fix _tf Multi-Rotation Bug

- [x] 1.1 Change `_tf(translate, scale, rotate)` to `_tf(translate, scale, *rotations)` in `src/omen/scenes.py`, chain all rotation tuples in order
- [x] 1.2 Update all `_tf()` call sites in scenes.py to pass rotations as positional tuples instead of `rotate=` keyword

## 2. Periodic Checkpointing

- [x] 2.1 Add `save_checkpoint_rotating(self, base_dir, keep=3)` to `OmenTrainer` — saves `step_{iter}.omen`, deletes oldest
- [x] 2.2 Add `--checkpoint-every` CLI arg (type=int, default=50) to `scripts/start_training.py`
- [x] 2.3 Add checkpoint call in `_train_on_data()` when `trainer.iteration % args.checkpoint_every == 0`
- [x] 2.4 Add checkpoint call in `_run_multi_scene()` after each scene mastered (before flush)

## 3. Resume from Checkpoint

- [x] 3.1 Add `--resume` flag to CLI
- [x] 3.2 Add `_find_latest_checkpoint(dir)` helper to find newest `step_*.omen` file
- [x] 3.3 Wire `--resume` to call `trainer.load_checkpoint(latest)` before training starts

## 4. Log File

- [x] 4.1 Add `--log-file` CLI arg
- [x] 4.2 Add `logging.FileHandler` in `main()` when `--log-file` is provided

## 5. Tmux Launcher

- [x] 5.1 Create `scripts/train_tmux.sh` — creates tmux session, runs training with `--log-file`, prints attach instructions
- [x] 5.2 Create `logs/` dir, add to `.gitignore`

## 6. Testing

- [x] 6.1 Ruff check + format all modified files
- [ ] 6.2 Quick test: `--scenes all --steps 10 --steps-per-frame 100` via tmux, verify checkpoint files created
