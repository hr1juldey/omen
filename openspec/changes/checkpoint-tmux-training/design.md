## Architecture

### _tf() fix
Change signature from `_tf(translate=None, scale=None, rotate=None)` to `_tf(translate=None, scale=None, *rotations, rotate=None)`.
Each `rotations` element is `(axis, angle)` tuple. Chain all rotations in order. Keep `rotate=None` as first element for backwards compat — actually simpler: just accept `*rotations` as positional after scale.

Better approach: `_tf(translate=None, scale=None, *rotations)` — each rotation is `(axis, angle)` tuple. Update all call sites.

### Periodic checkpointing
- `OmenTrainer.save_checkpoint_rotating(base_dir, keep=3)` — saves `step_{iter}.omen`, deletes oldest beyond `keep`
- In training loop: check `if trainer.iteration % args.checkpoint_every == 0`
- In curriculum: save after each scene mastered (before `flush_graph_cache`)
- Checkpoint dir: `~/.cache/omen/checkpoints/`

### Resume
- `--resume` flag: call `trainer.load_checkpoint(latest)` before training starts
- Find latest by sorting `step_*.omen` files by iteration number

### Log file
- `--log-file PATH`: add `logging.FileHandler` to root logger
- Simple: 2 lines in `main()` before any other logging

### Tmux launcher
- `scripts/train_tmux.sh`: wrapper that creates tmux session, runs training with `--log-file`, prints attach/tail instructions
- Log dir: `logs/` in project root
- Session name: `omen-training`
