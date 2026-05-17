## 1. Functional Forward Pass — Model Sub-modules (one file each, <100 LOC)

- [ ] 1.1 Create `src/omen/model/functional/` package with `__init__.py` and param prefix helper (`_extract_prefix(params, prefix)` returns subset dict)
- [ ] 1.2 Implement `scene_encoder.py` — `scene_encoder_fn(params, scene_graph)` functional GNN + linear matching SceneGraphEncoder. Import from `omen.model.scene_encoder` for shape reference only.
- [ ] 1.3 Implement `render_encoder.py` — `render_encoder_fn(params, rgba)` functional conv2d layers. Import `sigmoid_gpu`, `silu_gpu` from `omen.kernels.activations`.
- [ ] 1.4 Implement `cross_attn.py` — `cross_attn_fn(params, render_latent, scene_latent)` functional attention
- [ ] 1.5 Implement `decoder.py` — `decoder_fn(params, latent, noisy_image)` functional U-Net. Import `sigmoid_gpu`, `silu_gpu` from `omen.kernels.activations`.
- [ ] 1.6 Implement `sigreg.py` — `sigreg_fn(predicted_latent, config)` pure ops, 0 params. Import `SIGREG_LAMBDA` from `omen.model.jepa` (single source, no magic number).
- [ ] 1.7 Implement `confidence.py` — `confidence_fn(params, latent, h, w)` functional MLP
- [ ] 1.8 Implement `ar_predictor.py` — `ar_predictor_fn(params, history, current, delta)` functional conditional blocks
- [ ] 1.9 Implement `episodic.py` — `episodic_fn(params, latent)` functional linear layers
- [ ] 1.10 Wire `__init__.py` — import and re-export all functional fns. Verify ruff clean.

## 2. Pure Loss Function (<100 LOC)

- [ ] 2.1 Create `src/omen/training/trainer/pure_loss.py` — implement `pure_loss_fn(params, noisy, gt, scene_latent, config)` chaining functional sub-modules from `omen.model.functional`. Absolute imports only.
- [ ] 2.2 Write test: verify `pure_loss_fn` produces identical output to `compute_training_loss` for same inputs (float32 tolerance)
- [ ] 2.3 Write test: verify `value_and_grad(pure_loss_fn, argnums=0)` produces valid gradients for all 139 params

## 3. Compiled Train Step (<100 LOC)

- [ ] 3.1 Create `src/omen/training/trainer/compiled_step.py` — implement `compiled_train_step` decorated with `@nb.compile`. Import `COMPONENT_LRS` from `omen.training.trainer.optimizers` (DRY, no magic numbers).
- [ ] 3.2 Implement component loop with `sorted(COMPONENT_LRS.keys())`, extracting subset params/grads and calling `adamw_update` per component
- [ ] 3.3 Return `(new_params, new_states, loss)` — no side effects
- [ ] 3.4 Write test: first call compiles (slow), second call hits cache, stats show misses=1 hits=1

## 4. Compiled Trainer (<100 LOC)

- [ ] 4.1 Create `src/omen/training/trainer/compiled_trainer.py` — `CompiledOmenTrainer` class wrapping compiled_step with `__init__` (model, config, creates optimizer states from `COMPONENT_LRS`)
- [ ] 4.2 Implement `train_step_tiled` — splits images into tiles, calls compiled_step per tile, accumulates loss
- [ ] 4.3 Implement `save_checkpoint` / `load_checkpoint` for params dict + optimizer states (outside compiled function)
- [ ] 4.4 Write test: 5-step training on tiny 64x64 input, verify loss decreases and RAM stays flat

## 5. Training Loop Integration

- [ ] 5.1 Update `scripts/start_training.py` to import and use `CompiledOmenTrainer` (absolute import from `omen.training.trainer.compiled_trainer`)
- [ ] 5.2 Add CLI flag `--compiled` (default True) to switch between compiled and eager trainers
- [ ] 5.3 Remove per-tile `clear_all()` calls in compiled mode — graph is reused, not destroyed
- [ ] 5.4 Update RAM guard: warn during warmup (expected ~25GB peak), abort only if steady-state exceeds limit

## 6. Validation

- [ ] 6.1 Run 10-step training at 512x512 with 256x256 tiles — verify compilation completes, cache hits, no OOM
- [ ] 6.2 Monitor RAM: confirm ~25GB peak during warmup, ~10GB steady state after
- [ ] 6.3 Verify loss convergence: loss at step 10 measurably lower than step 1
- [ ] 6.4 Run `uv run ruff check --fix` and `uv run ruff format` on all new/modified files — zero violations
