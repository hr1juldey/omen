# Tasks: fix-training-wiring

## Phase 1: Config Foundation

- [x] 1.1 Create `src/omen/config.py` with OmenConfig dataclass, ComponentSwitches, TrainingSwitches, ModeSwitches, and preset configs (v1_dense, v1_moe, v1_animation, full)
- [x] 1.2 Add config serialization (to_dict/from_dict) for checkpoint compatibility
- [x] 1.3 Add config validation (mutually exclusive switches, dependency checks like ar_predictor needs scene_delta_encoder)

## Phase 2: Part 2 Fixes — Wrong Implementations

- [x] 2.1 **Fix 2.1**: Refactor `trainer.py` to create per-component AdamW optimizers (encoder_opt, decoder_opt, shared_expert_opt, material_opt, light_opt, geo_opt, motion_opt, ar_opt, episodic_opt) — one per component group with independent base lr
- [x] 2.2 **Fix 2.1**: Add `_active_optimizers()` method to OmenTrainer that returns only optimizers for enabled components (respects config switches)
- [x] 2.3 **Fix 2.1**: Refactor `train_step()` to iterate active optimizers, apply per-component lr, handle Nabla's `model = optimizer.step()` return pattern across multiple optimizers
- [x] 2.4 **Fix 2.4**: Modify `temporal.py` to expose z_score as a returnable value from `detect_surprise()` (already returned, ensure it's threaded through call chain)
- [x] 2.5 **Fix 2.4**: Add `_compute_lr(name, z_score)` to OmenTrainer that applies surprise modulation formula: `lr = base_lr * (1.0 + scale * min(z_score, 5.0))`
- [x] 2.6 **Fix 2.4**: Thread z_score through `jepa_bridge.train_step()` → `trainer.train_step()` for lr modulation
- [x] 2.7 **Fix 2.2**: Create `src/omen/modes/replay.py` with StratifiedReplayBuffer class (500 items, per-scene sub-buffers, stratified sampling, 1:1 replay ratio)
- [x] 2.8 **Fix 2.2**: Update `lora_manager.py` to use StratifiedReplayBuffer instead of flat deque; keep as compat shim exporting same interface
- [x] 2.9 **Fix 2.3**: Add `simple_variance_regularization(latent)` function to `sigreg.py` — 3-line `-log(std + eps)` implementation
- [x] 2.10 **Fix 2.3**: Add config switch logic to SIGRegLoss.forward() — return simple_reg or sigreg or 0 based on config

## Phase 3: Part 3 Fixes — Correct but Wrong Way

- [x] 3.1 **Fix 3.5**: Create `src/omen/model/episodic.py` with EpisodicCorrection network (2-layer MLP, dim*2 → hidden → dim, ~100K params)
- [x] 3.2 **Fix 3.5**: Register EpisodicCorrection as submodule in OmenJEPA.__init__() with config switch
- [x] 3.3 **Fix 3.5**: Add episodic optimizer to OmenTrainer (lr=2e-2, 400x higher than base) following Nabla LoRA pattern
- [x] 3.4 **Fix 3.1**: Add scene-graph routing path to `moe.py` — nn.Embedding for material_id + light_type, projection to routing logits
- [x] 3.5 **Fix 3.1**: Add `_route_from_scene_graph()` method to TileMoERouter that builds routing from material_ids and light_type_ids
- [x] 3.6 **Fix 3.1**: Add routing switch to TileMoERouter.forward() — scene_graph_routing=True uses new path, False uses current fingerprint path
- [x] 3.7 **Fix 3.7**: Add config-gated passthrough to ARPredictor.forward() — when `config.ar_predictor = False`, return current_latent unchanged
- [x] 3.8 **Fix 3.7**: Add history management to JEPABridge — only populate history buffer when `config.ar_predictor = True`
- [x] 3.9 **Fix 3.8**: Add mode switches to denoiser.py pipeline — check config.modes.denoiser/adaptive/multires/temporal, raise or passthrough for disabled modes

## Phase 4: Integration — Wire Config Through Full Pipeline

- [x] 4.1 Modify `OmenJEPA.__init__()` to accept OmenConfig, conditionally initialize routing embeddings and episodic network
- [x] 4.2 Modify `OmenJEPA.forward()` to check config switches for MoE, AR, episodic, confidence — identity passthrough when disabled
- [x] 4.3 Modify `OmenJEPA.compute_loss()` to check config switches for SIGReg vs simple_var_reg
- [x] 4.4 Modify `JEPABridge._init_model()` to create and pass OmenConfig to OmenJEPA and OmenTrainer
- [x] 4.5 Modify `OmenTrainer.__init__()` to accept OmenConfig, create per-component optimizers based on enabled switches
- [x] 4.6 Add config to checkpoint save/load — serialize OmenConfig alongside model weights
- [x] 4.7 Ensure backward compatibility — loading old checkpoints without config defaults to v1_dense()

## Phase 5: Validation

- [x] 5.1 Verify V1 dense config (`OmenConfig.v1_dense()`) initializes correctly: MoE OFF, AR OFF, SIGReg OFF, episodic ON
- [x] 5.2 Verify forward pass with V1 config produces valid output (no NaN, correct shapes)
- [x] 5.3 Verify train_step with V1 config trains only enabled components (check gradient flow with requires_grad tracing)
- [x] 5.4 Verify switching MoE ON mid-training doesn't crash (parameters exist, just weren't trained)
- [x] 5.5 Verify switching AR ON mid-training doesn't crash (history buffer starts empty, grows)
- [x] 5.6 Run existing test suite — all tests pass with default config (28 passed)
- [x] 5.7 Verify scene-graph routing produces different expert assignments than pixel fingerprint routing on a test scene
- [x] 5.8 Verify surprise lr modulation changes optimizer lr when z_score > 0
- [x] 5.9 Verify stratified replay buffer maintains per-scene diversity across 10+ scene additions
- [x] 5.10 Verify episodic correction has separate optimizer with lr=2e-2 (vs base 5e-5)
