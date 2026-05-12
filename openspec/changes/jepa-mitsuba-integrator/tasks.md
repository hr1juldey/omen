## 1. Project Structure

- [ ] 1.1 Create `jepa_kernels/` directory for Mojo source
- [ ] 1.2 Create `src/omen_integrator/scene_extractor.py`
- [ ] 1.3 Create `src/omen_integrator/jepa_bridge.py`
- [ ] 1.4 Create `src/omen_integrator/modes/` directory
- [ ] 1.5 Create `src/omen_integrator/modes/__init__.py`
- [ ] 1.6 Create `src/omen_integrator/modes/denoiser.py`
- [ ] 1.7 Create `src/omen_integrator/modes/adaptive.py`
- [ ] 1.8 Create `src/omen_integrator/modes/multires.py`
- [ ] 1.9 Create `jepa_kernels/C_ABI.mojo` with struct definitions
- [ ] 1.10 Create `jepa_kernels/build.sh` for compilation

## 2. Scene Graph Extraction

- [ ] 2.1 Implement `extract_geometry()` from `mi.Scene.shapes()`
- [ ] 2.2 Extract vertex positions using `shape.vertex_positions`
- [ ] 2.3 Extract face indices using `shape.faces`
- [ ] 2.4 Extract material indices per face
- [ ] 2.5 Implement `extract_materials()` from BSDFs
- [ ] 2.6 Support `mi.PrincipledBSDF` parameter extraction
- [ ] 2.7 Support `mi.RoughBSDF` parameter extraction
- [ ] 2.8 Implement `extract_lights()` from `mi.Scene.emitters()`
- [ ] 2.9 Extract point light position, intensity, color
- [ ] 2.10 Extract area light position, normal, size
- [ ] 2.11 Implement `extract_camera()` from `mi.Scene.sensors()[0]`
- [ ] 2.12 Extract camera transform matrix via `sensor.to_world()`
- [  ] 2.13 Extract FOV from `sensor.x_fov()` or equivalent
- [  ] 2.14 Extract clip planes (`near_clip`, `far_clip`)
- [ ] 2.15 Test: Extract Cornell box (2 meshes, 1 light, 1 camera)

## 3. C ABI Bridge

- [ ] 3.1 Create C header `jepa_kernels/omen_bridge.h`
- [ ] 3.2 Define `SceneGraph` struct with geometry/materials/lights/camera arrays
- [ ] 3.3 Define `RenderObservation` struct with RGBA/depth/normal/albedo pointers
- [ ] 3.4 Define `omen_denoise()` function signature
- [ ] 3.5 Define `omen_predict_confidence()` function signature
- [ ] 3.6 Define `omen_merge_multires()` function signature
- [ ] 3.7 Implement `JEPABridge` class in `jepa_bridge.py`
- [ ] 3.8 Load `.so` library via `ctypes.CDLL`
- [ ] 3.9 Define Python ctypes structs matching C headers
- [ ] 3.10 Implement `load()` method with library path resolution
- [ ] 3.11 Implement `denoise()` method calling `omen_denoise()`
- [ ] 3.12 Implement `predict_confidence()` method calling `omen_predict_confidence()`
- [ ] 3.13 Implement `merge_multires()` method calling `omen_merge_multires()`
- [ ] 3.14 Add error handling for library load failures (graceful degradation)
- [ ] 3.15 Test: Load dummy .so, call function, verify return code

## 4. Mojo C ABI Implementation

- [ ] 4.1 Implement `C_ABI.mojo` with struct definitions
- [ ] 4.2 Define `SceneGraph` Mojo struct matching C header
- [ ] 4.3 Define `RenderObservation` Mojo struct matching C header
- [ ] 4.4 Add `@register_function` decorator to `omen_denoise`
- [ ] 4.5 Add `@register_function` decorator to `omen_predict_confidence`
- [ ] 4.6 Add `@register_function` decorator to `omen_merge_multires`
- [ ] 4.7 Implement `UnsafePointer` wrapping for input buffers
- [ ] 4.8 Create `DeviceBuffer` wrappers with `owning=False` for zero-copy
- [  ] 4.9 Compile to `libomen.so` via build.sh
- [ ] 4.10 Test: Load .so from Python, verify symbols resolve

## 5. Mode 1 - Denoiser

- [ ] 5.1 Implement `render_denoiser()` in `modes/denoiser.py`
- [ ] 5.2 Call `scene_extractor.extract()` on Mitsuba scene
- [ ] 5.3 Render preview at 4 spp: `mi.render(scene, spp=4)`
- [ ] 5.4 Extract RGBA tensor from render result
- [ ] 5.5 Call `jepa_bridge.denoise(scene, noisy_rgba)`
- [ ] 5.6 Return denoised output
- [ ] 5.7 Test: Denoise Cornell box at 4 spp
- [ ] 5.8 Verify: Output has less noise than input
- [ ] 5.9 Verify: No artifacts or hallucinations

## 6. Mode 2 - Adaptive

- [ ] 6.1 Implement `render_adaptive()` in `modes/adaptive.py`
- [ ] 6.2 PASS 1: Render preview at 4 spp
- [ ] 6.3 Extract scene graph
- [ ] 6.4 Call `jepa_bridge.predict_confidence(scene, preview)`
- [ ] 6.5 Receive confidence map [H, W, 1]
- [ ] 6.6 PASS 2: Render at 128 spp
- [ ] 6.7 Merge: high-conf pixels = JEPA, low-conf pixels = path-traced
- [ ] 6.8 Implement `merge_by_confidence()` function
- [ ] 6.9 Calculate sample reduction ratio
- [ ] 6.10 Test: Adaptive render on Cornell box
- [ ] 6.11 Verify: 4-8× sample reduction vs uniform 128 spp
- [  ] 6.12 Verify: Quality matches uniform 128 spp (SSIM > 0.95)

## 7. Mode 3 - Multi-Resolution

- [ ] 7.1 Implement `render_multires()` in `modes/multires.py`
- [ ] 7.2 PASS 1: Set film size to [H/4, W/4]
- [ ] 7.3 Render low-res at 256 spp
- [ ] 7.4 PASS 2: Set film size to [H, W]
- [ ] 7.5 Render high-res at 4 spp
- [ ] 7.6 Extract scene graph
- [ ] 7.7 Call `jepa_bridge.merge_multires(scene, low_res, high_res, scale=4)`
- [  ] 7.8 Return merged high-res output
- [ ] 7.9 Calculate effective speedup
- [ ] 7.10 Test: Multi-res render on Cornell box
- [ ] 7.11 Verify: 8-16× speedup vs uniform 256 spp
- [ ] 7.12 Verify: Edges are sharp (no blur)
- [ ] 7.13 Verify: No DLSS-style hallucination artifacts

## 8. Mojo JEPA Model Architecture

- [ ] 8.1 Create `jepa_kernels/scene_encoder.mojo`
- [ ] 8.2 Implement `SceneEncoder` struct with transformer layers
- [ ] 8.3 Encode geometry tokens: vertices → embeddings
- [ ] 8.4 Encode material features: BSDF params → embeddings
- [ ] 8.5 Encode light features: position/intensity → embeddings
- [ ] 8.6 Create `jepa_kernels/image_encoder.mojo`
- [ ] 8.7 Implement `ImageEncoder` struct with CNN
- [ ] 8.8 Extract patches from image (8×8 pixels)
- [ ] 8.9 Strided convolutions to encode patches
- [  ] 8.10 Create `jepa_kernels/jepa.mojo`
- [ ] 8.11 Implement `JEPAModel` struct
- [ ] 8.12 Integrate scene_encoder and image_encoder
- [ ] 8.13 Implement cross-attention: image queries scene
- [ ] 8.14 Implement decoder: latent → RGB
- [ ] 8.15 Test: Forward pass with dummy data (no training yet)

## 9. Mojo Confidence Head

- [ ] 9.1 Create `jepa_kernels/confidence.mojo`
- [ ] 9.2 Implement `ConfidenceHead` struct
- [ ] 9.3 MLP layers: latent → confidence (0-1)
- [ ] 9.4 Sigmoid activation for [0, 1] output
- [ ] 9.5 Integrate into `jepa.mojo` as second output head
- [ ] 9.6 Test: Predict confidence on random latents

## 10. Mojo Multi-Resolution Merge

- [ ] 10.1 Create `jepa_kernels/multires.mojo`
- [ ] 10.2 Implement `MultiResMerge` struct
- [ ] 10.3 Upsample low-res to high-res (bilinear + refinement)
- [ ] 10.4 Extract high-res features from noisy pass
- [ ] 10.5 Fuse using scene graph edges (geometry-aware)
- [ ] 10.6 Fuse material boundaries (material-aware)
- [ ] 10.7 Test: Merge 25% + 100% renders

## 11. Model Checkpointing & Storage

- [ ] 11.1 Create `src/omen_integrator/model_store.py` module
- [ ] 11.2 Implement `ModelStore` class with cache directory management
- [ ] 11.3 Create `~/.cache/omen/models/` directory if not exists
- [ ] 11.4 Implement `save_checkpoint()` function in Mojo
- [ ] 11.5 Serialize model weights via Nabla `state_dict()`
- [ ] 11.6 Serialize optimizer state (Adam moments, learning rate)
- [ ] 11.7 Save training iteration number and metrics
- [ ] 11.8 Write checkpoint to `checkpoint_iter_<N>.omen`
- [ ] 11.9 Update symlink `latest.omen` to most recent checkpoint
- [ ] 11.10 Implement `load_checkpoint()` function in Mojo
- [ ] 11.11 Load model weights and optimizer state from file
- [ ] 11.12 Resume training from saved iteration
- [ ] 11.13 Implement checkpoint metadata JSON creation
- [ ] 11.14 Include architecture hash, Omen version, Nabla version in metadata
- [ ] 11.15 Implement checkpoint validation before loading
- [ ] 11.16 Verify architecture hash matches current model
- [ ] 11.17 Implement graceful error on incompatible checkpoint
- [ ] 11.18 Create `jepa_kernels/checkpoint.mojo` for save/load operations
- [ ] 11.19 Test: Save checkpoint, crash, resume from checkpoint
- [ ] 11.20 Implement scene hash computation (geometry + materials + lights)
- [ ] 11.21 Use SHA256 for scene hash, exclude camera position
- [ ] 11.22 Implement scene-specific model cache lookup
- [ ] 11.23 Check `~/.cache/omen/models/scenes/<hash>/fine_tuned.omen`
- [ ] 11.24 Implement base model download on first use
- [ ] 11.25 Download from `https://omen-render.org/models/base_v0.omen`
- [ ] 11.26 Verify SHA256 checksum after download
- [ ] 11.27 Implement scene similarity detection
- [ ] 11.28 Compute scene feature vector (mesh count, material types, etc.)
- [ ] 11.29 Query `scene_index.json` for similar scenes (cosine > 0.85)
- [ ] 11.30 Implement quick adaptation from similar scene model (10 iters)
- [ ] 11.31 Implement model aggregation (local learning)
- [ ] 11.32 Load base model + fine-tuned models for federated averaging
- [ ] 11.33 Save aggregated model as `base_v1_local.omen`
- [ ] 11.34 Implement anonymous model contribution (opt-in)
- [ ] 11.35 De-identify scene data before upload (remove coordinates, textures)
- [ ] 11.36 Upload only weight deltas (difference from base model)
- [ ] 11.37 Test: Full lifecycle (base → fine-tune → aggregate → reuse)
- [ ] 11.38 Implement GPU rendering backend detection
- [ ] 11.39 Query `mi.variant()` to detect CPU vs GPU rendering
- [ ] 11.40 Parse variant: `cuda_ad_*` → NVIDIA, `metal_ad_*` → Apple, `cpu_ad_*` → CPU
- [ ] 11.41 Configure zero-copy buffer passing when both on GPU
- [ ] 11.42 Implement `UnsafePointer` wrapping for GPU tensors
- [ ] 11.43 Implement memcpy fallback for CPU→GPU path
- [ ] 11.44 Implement GPU memory detection via CUDA/HIP/Metal API
- [ ] 11.45 Query total memory, free memory, compute available budget
- [ ] 11.46 Estimate memory requirements for model + scene + buffers
- [ ] 11.47 Implement graceful degradation on insufficient memory
- [ ] 11.48 Reduce batch size if training memory insufficient
- [ ] 11.49 Fall back to CPU if GPU memory exhausted
- [ ] 11.50 Test: Zero-copy on GPU, memcpy fallback on CPU

## 12. Temporal Coherence & JEPA World Model

- [ ] 12.1 Create `jepa_kernels/world_model.mojo` with OmenWorldModel struct
- [ ] 12.2 Implement ARPredictor struct (autoregressive next-step predictor from LeWM)
- [ ] 12.3 Implement history window circular buffer (configurable, default 3 frames)
- [ ] 12.4 Implement SceneDelta struct: camera delta, object deltas, light deltas, birth events
- [ ] 12.5 Implement SceneDeltaEncoder (replaces LeWM's action_encoder)
- [ ] 12.6 Encode camera movement delta: translation + rotation quaternion → embedding
- [ ] 12.7 Encode per-object animation deltas: transform delta per object → aggregated embedding
- [ ] 12.8 Encode fluid/smoke introduction as birth event: type + position + size → embedding
- [ ] 12.9 Encode light parameter deltas: intensity + color + position changes → embedding
- [ ] 12.10 Encode material animation deltas: parameter changes → embedding
- [ ] 12.11 Implement SIGReg loss (Sketch Isotropic Gaussian Regularizer from LeWM)
- [ ] 12.12 Implement prediction loss: MSE(predicted_latent, target_latent)
- [ ] 12.13 Total loss: L = L_pred + λ * L_sigreg (λ configurable, default 1.0)
- [ ] 12.14 Implement surprise detection: MSE(predicted, actual) > threshold (2σ)
- [ ] 12.15 Detect birth events (new fluid, smoke, light) as auto-surprise (skip prediction)
- [ ] 12.16 Detect camera jump cuts: translation > 1 unit or rotation > 45° → clear history
- [ ] 12.17 Implement periodic validation: every N predicted frames, render + compare
- [ ] 12.18 Implement topology-based scene hash (face connectivity, not vertex positions)
- [ ] 12.19 Verify: rotating object does NOT change topology hash
- [ ] 12.20 Verify: adding new light DOES change topology hash
- [ ] 12.21 Implement Mode 4 animation pipeline in `modes/animation.py`
- [ ] 12.22 Frame 0: render 4spp → JEPA denoise → store as anchor latent
- [ ] 12.23 Frames 1..N: ALWAYS render 1spp → encode dirty + scene graph → predict clean
- [ ] 12.24 Never predict from nothing - always conditioned on 1spp dirty render + scene graph
- [ ] 12.25 On surprise: re-render at 4spp → JEPA denoise → update history anchor
- [ ] 12.26 On jump cut: clear history → render 4spp → new anchor
- [ ] 12.27 Implement scene graph diffing between frames
- [ ] 12.28 Detect new/deleted scene elements (emitters, objects, lights)
- [ ] 12.29 Test: Render Cornell box animation with camera orbit (100 frames)
- [ ] 12.30 Test: Verify temporal coherence (no flickering between frames)
- [ ] 12.31 Test: Introduce new light mid-animation, verify surprise detection
- [ ] 12.32 Test: Jump cut mid-animation, verify history clear + re-anchor
- [ ] 12.33 Benchmark: 256x256 animation speed, target >30fps with 90% predicted frames

## 13. Mitsuba JEPA Gym (Differentiable Training)

- [ ] 13.1 Verify Mitsuba variant `cuda_ad_rgb` works with Dr.Jit autodiff
- [ ] 13.2 Create `training/jepa_gym.py` for differentiable training loop
- [ ] 13.3 Implement gym loop: 1spp render → JEPA predict → 256spp GT → loss → backward
- [ ] 13.4 Use `dr.backward(loss)` to propagate gradients through JEPA
- [ ] 13.5 Use `drjit.opt.Adam` optimizer for JEPA weight updates
- [ ] 13.6 Verify gradient flow: loss → JEPA weights (not through Mitsuba C++ path tracer)
- [ ] 13.7 Implement self-supervised data generation: random camera + light variations
- [ ] 13.8 Generate Cornell box animation training data: 500 frames with camera orbit
- [ ] 13.9 Generate surprise training data: random light on/off, object spawn/despawn
- [ ] 13.10 Generate fluid introduction training data: volume emitter appears mid-sequence
- [ ] 13.11 Train temporal predictor: 500 iterations on animation data
- [ ] 13.12 Validate: predicted frame SSIM > 0.85 vs 256spp ground truth
- [ ] 13.13 Validate: surprise detection catches >90% of actual surprises
- [ ] 13.14 Validate: false positive rate <10% (don't over-trigger path tracing)
- [ ] 13.15 Implement FLIP-style forward prediction: frame N + delta → predict frame N+1
- [ ] 13.16 Implement closed-loop validation: predict 10 frames, render actual, compare drift
- [ ] 13.17 Test: Train on Cornell box animation, verify loss decreases over 500 iterations

## 14. Training Infrastructure

- [ ] 14.1 Create `training/cornell_box_trainer.py`
- [ ] 14.2 Implement `generate_training_pair()` for denoiser
- [ ] 14.3 Render Cornell box at 4spp + 256spp (same seed)
- [ ] 14.4 Save training pairs to disk
- [ ] 14.5 Implement `generate_variance_pairs()` for confidence
- [ ] 14.6 Render Cornell box 8× at 4spp (different seeds)
- [ ] 14.7 Compute variance across renders → uncertainty labels
- [ ] 14.8 Implement `generate_multires_pairs()` for merge
- [ ] 14.9 Render 25% res 256spp + 100% res 4spp + 100% res 256spp
- [ ] 14.10 Create training loop in Mojo (Nabla autograd)
- [ ] 14.11 Implement L1 loss for denoiser training
- [ ] 14.12 Implement MSE loss for confidence training
- [ ] 14.13 Implement L1 loss for multires training
- [ ] 14.14 Test: Train for 100 iterations, verify loss decreases

## 15. Testing & Validation

- [ ] 15.1 Create `tests/test_scene_extractor.py`
- [ ] 15.2 Test: Extract Cornell box scene, verify geometry/materials/lights
- [ ] 15.3 Create `tests/test_jepa_bridge.py`
- [ ] 15.4 Test: Load .so, call functions, verify return codes
- [ ] 15.5 Create `tests/test_cornell_denoise.py`
- [ ] 15.6 Test: Mode 1 on Cornell box, compare SSIM
- [ ] 15.7 Create `tests/test_cornell_adaptive.py`
- [ ] 15.8 Test: Mode 2 on Cornell box, measure sample reduction
- [ ] 15.9 Create `tests/test_cornell_multires.py`
- [ ] 15.10 Test: Mode 3 on Cornell box, measure speedup
- [ ] 15.11 Benchmark: Time each mode, verify targets met
- [ ] 15.12 Verify: Mode 2 achieves 4-8× sample reduction
- [ ] 15.13 Verify: Mode 3 achieves 8-16× speedup
- [ ] 15.14 Verify: All modes produce clean renders (no artifacts)
