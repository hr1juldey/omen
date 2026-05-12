## 1. Project Structure

- [ ] 1.1 Create `jepa_kernels/` directory for Mojo source
- [ ] 1.2 Create `src/omen_integrator/scene_extractor.py`
- [ ] 1.3 Create `src/omen_integrator/jepa_bridge.py`
- [ ] 1.4 Create `src/omen_integrator/modes/` directory
- [ ] 1.5 Create `src/omen_integrator/modes/__init__.py`
- [ ] 1.6 Create `src/omen_integrator/modes/denoiser.py`
- [ ] 1.7 Create `src/omen_integrator/modes/adaptive.py`
- [ ] 1.8 Create `src/omen_integrator/modes/multires.py`
- [ ] 1.9 Create `src/omen_integrator/modes/animation.py`
- [ ] 1.10 Create `jepa_kernels/C_ABI.mojo` with struct definitions
- [ ] 1.11 Create `jepa_kernels/build.sh` for compilation to `libomen.so`
- [ ] 1.12 Create `training/cornell_box_trainer.py`
- [ ] 1.13 Create `training/jepa_gym.py`

> Spec: cross-cutting (all specs reference these files)

## 2. Scene Graph Extraction

> Spec: `specs/scene-graph-encoding/`

- [ ] 2.1 Implement `extract_geometry()` iterating `mi.Scene.shapes()`
- [ ] 2.2 Extract vertex positions via `mi.traverse(shape)['vertex_positions']` -> Float array Nx3
- [ ] 2.3 Extract face indices via `shape.face_indices(0)` or `shape.faces_buffer()` -> UInt array Fx3
- [ ] 2.4 Check `shape.has_vertex_normals()` and extract normals if available
- [ ] 2.5 Assign sequential material_id per unique BSDF (first BSDF -> id=0, etc.)
- [ ] 2.6 Implement `extract_materials()` calling `mi.traverse(shape.bsdf())`
- [ ] 2.7 Support `mi.PrincipledBSDF`: extract diffuse_reflectance, roughness, metallic, specular_reflectance
- [ ] 2.8 Support `mi.RoughBSDF`: extract alpha, diffuse_reflectance, specular_reflectance
- [ ] 2.9 Handle unknown BSDF: extract available params via `mi.traverse()`, log warning
- [ ] 2.10 Implement `extract_lights()` iterating `mi.Scene.emitters()`
- [ ] 2.11 Extract point light: position (Point3f) + intensity (Color3f)
- [ ] 2.12 Extract area light: `shape.emitter()` + radiance + surface_area
- [ ] 2.13 Detect environment light via `emitter.is_environment()`
- [ ] 2.14 Implement `extract_camera()` from `mi.Scene.sensors()[0]`
- [ ] 2.15 Extract camera transform, FOV, clip planes, film size
- [ ] 2.16 Serialize scene graph into C ABI flat arrays for ctypes transfer
- [ ] 2.17 Handle variable mesh counts: concatenate with boundary offsets
- [ ] 2.18 Test: Extract Cornell box, verify 2 meshes, 1 area light, 1 camera

## 3. Mojo C ABI Bridge

> Spec: `specs/mojo-cabi-bridge/`

- [ ] 3.1 Create `jepa_kernels/omen_bridge.h` with C struct definitions
- [ ] 3.2 Define SceneGraph C struct and RenderObservation C struct
- [ ] 3.3 Define function signatures with int32 return codes
- [ ] 3.4 Define Mojo structs matching C headers in C_ABI.mojo
- [ ] 3.5 Add @register_function to all 4 functions
- [ ] 3.6 Implement DeviceBuffer zero-copy wrapping and TileTensor creation
- [ ] 3.7 Compile to libomen.so, verify symbols
- [ ] 3.8 Implement JEPABridge class in jepa_bridge.py with ctypes loading
- [ ] 3.9 Implement denoise(), predict_confidence(), merge_multires() methods
- [ ] 3.10 Handle load failure and inference errors (graceful degradation)
- [ ] 3.11 Test: Load libomen.so, call functions, verify return codes

## 4. JEPA Model Architecture (Mojo/Nabla)

> Spec: `specs/jepa-model-architecture/`

- [ ] 4.1 Create scene_encoder.mojo with ViT-Tiny (hidden=192, heads=3, depth=12, patch=14)
- [ ] 4.2 Implement patch embedding, CLS token, positional encoding, 12 transformer layers
- [ ] 4.3 Implement Projector MLP (192 -> 2048 -> 192 with BatchNorm)
- [ ] 4.4 Create scene_delta_encoder.mojo (Conv1d + MLP, 155K params)
- [ ] 4.5 Create arpredictor.mojo with ConditionalBlock (AdaLN-zero, 6 layers, 10.8M params)
- [ ] 4.6 Create sigreg.mojo (Epps-Pulley, 17 knots, 1024 projections, 0 learnable params)
- [ ] 4.7 Implement Decoder (latent -> RGBA) and ConfidenceHead (MLP + Sigmoid)
- [ ] 4.8 Test: Forward pass with dummy data, verify output shapes

## 5. JEPA Inference

> Spec: `specs/jepa-inference/`

- [ ] 5.1 Implement invoke_denoise, invoke_predict_confidence, invoke_merge_multires
- [ ] 5.2 Zero-copy GPU path and CPU-GPU memcpy fallback
- [ ] 5.3 Test: Call each function with Cornell box data

## 6. Mode 1 - Denoiser

> Spec: `specs/denoiser-mode/`

- [ ] 6.1 Implement render_denoiser(scene, spp=4) in modes/denoiser.py
- [ ] 6.2 Render 4spp, extract scene graph, call bridge.denoise, return clean RGBA
- [ ] 6.3 Handle bridge unavailable
- [ ] 6.4 Test: Cornell box at 4spp, SSIM > 0.90 vs 256spp

## 7. Mode 2 - Adaptive

> Spec: `specs/adaptive-guidance/`

- [ ] 7.1 Implement render_adaptive(scene, spp_target=128) in modes/adaptive.py
- [ ] 7.2 PASS 1: 4spp preview + confidence prediction
- [ ] 7.3 PASS 2: 128spp high-spp render
- [ ] 7.4 Confidence-weighted merge
- [ ] 7.5 Test: Cornell box, 4-8x sample reduction, SSIM > 0.95

## 8. Mode 3 - Multi-Resolution

> Spec: `specs/multires-merge/`

- [ ] 8.1 Implement render_multires(scene, scale=4) in modes/multires.py
- [ ] 8.2 PASS 1: 25% res 256spp, PASS 2: 100% res 4spp
- [ ] 8.3 Geometry-aware merge via Mojo kernel
- [ ] 8.4 Test: Cornell box, 8-16x speedup, PSNR > 30dB

## 9. Model Checkpointing & Storage

> Spec: `specs/checkpoint-storage/`

- [ ] 9.1 Implement save/load via Nabla state_dict(), NablaAdamW optimizer state
- [ ] 9.2 Metadata JSON with architecture hash, version validation
- [ ] 9.3 Topology-based scene hashing for animation cache
- [ ] 9.4 Base model download, scene-specific fine-tuning cache
- [ ] 9.5 GPU backend detection, zero-copy config, memory management
- [ ] 9.6 Local federated averaging, opt-in anonymous contribution
- [ ] 9.7 Test: Save checkpoint, crash, resume

## 10. Temporal Coherence & JEPA World Model

> Spec: `specs/temporal-coherence/`

- [ ] 10.1 Create world_model.mojo with FrameState, SceneDelta structs
- [ ] 10.2 Implement scene delta encoding (camera, objects, lights, births, materials)
- [ ] 10.3 Implement surprise detection (MSE z-score > 2 sigma, auto-surprise for new elements)
- [ ] 10.4 Implement jump cut detection (translation > 1 unit, rotation > 45 deg)
- [ ] 10.5 Implement modes/animation.py: frame 0 anchor, frames 1..N predict from 1spp dirty + history
- [ ] 10.6 Total loss: L_pred + 0.09 * L_sigreg (lambda from lewm.yaml)
- [ ] 10.7 Test: Cornell box camera orbit 100 frames, surprise detection, jump cut recovery

## 11. Training Gym

> Spec: `specs/training-gym/`

- [ ] 11.1 Verify AD variant, set up Dr.Jit data generation pipeline
- [ ] 11.2 Implement training pair generators (denoiser, variance, multires, temporal)
- [ ] 11.3 Implement omen_train_step in Mojo (Nabla forward + backward + NablaAdamW)
- [ ] 11.4 Cornell box 4-phase schedule: denoiser(100) -> confidence(100) -> multires(100) -> temporal(200)
- [ ] 11.5 Test: Run all phases, verify loss decreases

## 12. Testing & Validation

> Cross-cutting: references all specs

- [ ] 12.1 Create tests for scene extraction, bridge loading, all 4 modes
- [ ] 12.2 Benchmark all modes vs baselines
- [ ] 12.3 Verify: no artifacts, no hallucinations, temporal coherence
