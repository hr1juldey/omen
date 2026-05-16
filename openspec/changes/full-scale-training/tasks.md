## 1. GT SPP Default Change

- [ ] 1.1 Change `TrainingDataGenerator.__init__` default `gt_spp` from 256 to 512 in `src/omen/training/online_gen.py`
- [ ] 1.2 Update `start_training.py` CLI help text and any hardcoded SPP references

## 2. LR Schedule

- [ ] 2.1 Add `_compute_scheduled_lr(self, step, total_steps, warmup_steps, min_lr)` method to `OmenTrainer` in `src/omen/training/trainer/core.py` — linear warmup then cosine decay
- [ ] 2.2 Replace `_compute_lr()` calls with `_compute_scheduled_lr()` in `_apply_optimizer_updates()`, keeping surprise modulation as a multiplier on top

## 3. Animation Dispatcher

- [ ] 3.1 Add `ANIMATION_REGISTRY` dict in `src/omen/scenes.py` mapping scene names to their animation generators: `{"cornell": cornell_animations, "veach": veach_animations, ...}`
- [ ] 3.2 Add `get_animation_generator(scene_name, base_resolution)` function in `src/omen/scenes.py` that looks up the registry and returns the animation dict, or None if no generator exists
- [ ] 3.3 Update `scripts/start_training.py` `_run_animation()` to use the generic dispatcher instead of hardcoded `cornell_animations` import

## 4. Steps-Per-Frame

- [ ] 4.1 Add `--steps-per-frame` CLI argument to `scripts/start_training.py` (type=int, default=1)
- [ ] 4.2 Add `--lr-warmup` CLI argument (type=int, default=0)
- [ ] 4.3 Modify `_train_on_data()` to loop `args.steps_per_frame` times, logging sub-step index and loss per iteration

## 5. Multi-Scene Loop

- [ ] 5.1 Add `--scenes` CLI argument (choices=["single", "all"], default="single") to `scripts/start_training.py`
- [ ] 5.2 Add `_run_multi_scene()` function that iterates sorted(SCENE_REGISTRY), builds each scene, encodes scene graph, trains on tiles, logs scene transition
- [ ] 5.3 Wire `--scenes all` into `main()` to call `_run_multi_scene()` instead of `_run_static()`

## 6. CLI Integration & Testing

- [ ] 6.1 Add `--total-steps` CLI argument (required for cosine decay denominator) with default=1000
- [ ] 6.2 Pass `warmup_steps` and `total_steps` from CLI args through to `OmenTrainer`
- [ ] 6.3 Run ruff check + format on all modified files
- [ ] 6.4 Smoke test: single scene, single step, 512x512 — verify GT SPP=512 in logs
- [ ] 6.5 Smoke test: `--scenes all --steps 5` — verify all 5 scenes train
- [ ] 6.6 Smoke test: `--scene veach --animation camera_orbit --steps 3` — verify generic animation dispatch
