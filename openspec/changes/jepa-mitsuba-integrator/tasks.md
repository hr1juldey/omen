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

## 11. Training Infrastructure

- [ ] 11.1 Create `training/cornell_box_trainer.py`
- [ ] 11.2 Implement `generate_training_pair()` for denoiser
- [ ] 11.3 Render Cornell box at 4spp + 256spp (same seed)
- [ ] 11.4 Save training pairs to disk
- [ ] 11.5 Implement `generate_variance_pairs()` for confidence
- ] 11.6 Render Cornell box 8× at 4spp (different seeds)
- [ ] 11.7 Compute variance across renders → uncertainty labels
- [ ] 11.8 Implement `generate_multires_pairs()` for merge
- [ ] 11.9 Render 25% res 256spp + 100% res 4spp + 100% res 256spp
- [ ] 11.10 Create training loop in Mojo (Nabla autograd)
- [ ] 11.11 Implement L1 loss for denoiser training
- [ ] 11.12 Implement MSE loss for confidence training
- [ ] 11.13 Implement L1 loss for multires training
- [ ] 11.14 Test: Train for 100 iterations, verify loss decreases

## 12. Testing & Validation

- [ ] 12.1 Create `tests/test_scene_extractor.py`
- [ ] 12.2 Test: Extract Cornell box scene, verify geometry/materials/lights
- [ ] 12.3 Create `tests/test_jepa_bridge.py`
- [ ] 12.4 Test: Load .so, call functions, verify return codes
- [ ] 12.5 Create `tests/test_cornell_denoise.py`
- [ ] 12.6 Test: Mode 1 on Cornell box, compare SSIM
- [ ] 12.7 Create `tests/test_cornell_adaptive.py`
- [ ] 12.8 Test: Mode 2 on Cornell box, measure sample reduction
- [ 12.9 Create `tests/test_cornell_multires.py`
- [ ] 12.10 Test: Mode 3 on Cornell box, measure speedup
- [ ] 12.11 Benchmark: Time each mode, verify targets met
- [ ] 12.12 Verify: Mode 2 achieves 4-8× sample reduction
- [ ] 12.13 Verify: Mode 3 achieves 8-16× speedup
- [ 12.14 Verify: All modes produce clean renders (no artifacts)
