## 1. Rewrite JEPABridge with bootstrap + training support

- [x] 1.1 Add `model_bootstrap` mode to `JEPABridge.__init__()`: if no checkpoint at `~/.omen/checkpoints/latest.omen`, create fresh `OmenJEPA()` + `OmenTrainer` instead of raising error
- [x] 1.2 Add `train_step(noisy_rgb, gt_rgb, scene_graph)` method to `JEPABridge`: convert numpy→Nabla via DLPack, delegate to `self.trainer.train_step()`, return loss dict
- [x] 1.3 Add `save_checkpoint()` method: persist model state_dict + optimizer state + iteration count to `~/.omen/checkpoints/latest.omen`
- [x] 1.4 Add `init_lora(scene_hash, rank=8)` method: freeze base model, add LoRA adapters to encoder layers, create LoRA-only optimizer
- [x] 1.5 Add `save_checkpoint(scene_hash=)` overload: save LoRA adapter weights to `~/.omen/checkpoints/{hash}.omen` separately from base model
- [x] 1.6 Handle Nabla unavailable gracefully: set `model_available = False`, `train_step()` returns empty dict, `denoise()` returns input unchanged

## 2. Wire training into denoiser pipeline

- [x] 2.1 Add `_generate_training_pair()` helper in `denoiser.py`: render 4spp noisy + 256spp pseudo-GT at reduced resolution (480x270) using the same scene
- [x] 2.2 Add training phase to `render_denoiser()`: before denoising, call `_generate_training_pair()` then `bridge.train_step(noisy, gt, scene_graph)`
- [x] 2.3 Add SSIM quality gate after denoising: if SSIM between denoised and noisy < 0.5, return raw noisy render instead
- [x] 2.4 Add checkpoint save cadence: call `bridge.save_checkpoint()` every 10 training steps (track via bridge.iteration counter)
- [x] 2.5 Pass scene graph (from scene_extractor) to `train_step()` for scene-aware encoding

## 3. Add per-scene LoRA fine-tuning trigger

- [x] 3.1 Add render count tracking to `denoiser.py`: hash scene topology, count renders per hash in a dict
- [x] 3.2 When render count for a hash reaches 3, call `bridge.init_lora(scene_hash)` and run 50 fine-tuning iterations using cached training pairs
- [x] 3.3 After LoRA fine-tuning, call `bridge.save_checkpoint(scene_hash=hash)` to persist scene-specific adapters
- [x] 3.4 On subsequent renders of same scene, load scene-specific LoRA adapter if it exists

## 4. Update session.py to pass training config

- [x] 4.1 Add training config params to `OmenSession.render_scene()`: `train=True`, `train_resolution=(480, 270)`
- [x] 4.2 Ensure `JEPABridge` lifecycle: init on first render, save checkpoint on session close
- [x] 4.3 Add `__del__` or explicit `close()` to `OmenSession` that calls `bridge.save_checkpoint()`

## 5. Verify and test

- [x] 5.1 Verify `jepa_bridge.py` imports cleanly with and without Nabla installed
- [x] 5.2 Test model bootstrap: delete checkpoint, render, verify fresh model created and checkpoint saved
- [x] 5.3 Test training loop: render same scene twice, verify loss decreases between renders
- [x] 5.4 Test LoRA trigger: render same scene 3 times, verify LoRA adapters initialized
- [x] 5.5 Test fallback: run with Nabla uninstalled, verify raw Mitsuba render returned without error
- [x] 5.6 Run ruff check on all modified files
