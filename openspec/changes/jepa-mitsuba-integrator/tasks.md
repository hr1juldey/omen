## 1. Project Structure

- [x] 1.1 Create `src/omen/` package directory
- [x] 1.2 Create `src/omen/__init__.py` with Nabla import guard
- [x] 1.3 Create `src/omen/scene_extractor.py` for scene graph extraction
- [x] 1.4 Create `src/omen/jepa_bridge.py` for model loading + DLPack transfer
- [x] 1.5 Create `src/omen/model/` directory for Nabla model modules
- [x] 1.6 Create `src/omen/model/__init__.py`
- [x] 1.7 Create `src/omen/model/scene_encoder.py` (SceneGraphEncoder + RenderFeatureEncoder + CrossAttention)
- [x] 1.8 Create `src/omen/model/arpredictor.py` (ConditionalBlock + ARPredictor)
- [x] 1.9 Create `src/omen/model/decoder.py` (Conv2dTranspose decoder + ConfidenceHead)
- [x] 1.10 Create `src/omen/model/sigreg.py` (SIGReg as custom Nabla kernel)
- [x] 1.11 Create `src/omen/model/jepa.py` (OmenJEPA top-level module combining all components)
- [x] 1.12 Create `src/omen/kernels/` for custom Mojo GPU kernels (SIGReg, merge)
- [x] 1.13 Create `src/omen/modes/` directory for render mode implementations
- [x] 1.14 Create `src/omen/modes/__init__.py`
- [x] 1.15 Create `src/omen/modes/denoiser.py`
- [x] 1.16 Create `src/omen/modes/adaptive.py`
- [x] 1.17 Create `src/omen/modes/multires.py`
- [x] 1.18 Create `src/omen/modes/animation.py`
- [x] 1.19 Create `src/omen/training/` directory
- [x] 1.20 Create `src/omen/training/trainer.py` (Nabla training loop with LoRA)
- [x] 1.21 Create `src/omen/training/data_gen.py` (Dr.Jit training pair generation)

> Spec: cross-cutting (all specs reference these files)

## 2. Scene Graph Extraction

> Spec: `specs/scene-graph-encoding/`

- [x] 2.1 Implement `extract_geometry()` iterating `mi.Scene.shapes()`
- [x] 2.2 Extract vertex positions via `mi.traverse(shape)['vertex_positions']` -> Float array Nx3
- [x] 2.3 Extract face indices via `shape.face_indices(0)` or `shape.faces_buffer()` -> UInt array Fx3
- [x] 2.4 Check `shape.has_vertex_normals()` and extract normals if available
- [x] 2.5 Assign sequential material_id per unique BSDF (first BSDF -> id=0, etc.)
- [x] 2.6 Implement `extract_materials()` calling `mi.traverse(shape.bsdf())`
- [x] 2.7 Support `mi.PrincipledBSDF`: extract diffuse_reflectance, roughness, metallic, specular_reflectance
- [x] 2.8 Support `mi.RoughBSDF`: extract alpha, diffuse_reflectance, specular_reflectance
- [x] 2.9 Handle unknown BSDF: extract available params via `mi.traverse()`, log warning
- [x] 2.10 Implement `extract_lights()` iterating `mi.Scene.emitters()`
- [x] 2.11 Extract point light: position (Point3f) + intensity (Color3f)
- [x] 2.12 Extract area light: `shape.emitter()` + radiance + surface_area
- [x] 2.13 Detect environment light via `emitter.is_environment()`
- [x] 2.14 Implement `extract_camera()` from `mi.Scene.sensors()[0]`
- [x] 2.15 Extract camera transform, FOV, clip planes, film size
- [x] 2.16 Return scene graph as Python dict: `{geometry: np.array, materials: np.array, lights: np.array, camera: np.array}`
- [x] 2.17 Handle variable mesh counts: concatenate with boundary offsets
- [ ] 2.18 Test: Extract Cornell box, verify 2 meshes, 1 area light, 1 camera

## 3. DLPack Tensor Bridge

> Spec: `specs/mojo-cabi-bridge/` (renamed from C ABI to DLPack)

- [ ] 3.1 Implement DLPack transfer: `nb.Tensor.from_dlpack(dr_tensor)` for GPU zero-copy
- [ ] 3.2 Implement numpy fallback: `nb.ndarray(np_array)` for CPU renders
- [ ] 3.3 Convert scene graph dict values to Nabla tensors
- [ ] 3.4 Handle alpha channel addition: `nb.concatenate([render, ones], axis=-1)`
- [ ] 3.5 Implement Nabla -> numpy conversion for output: `tensor.numpy()`
- [ ] 3.6 GPU context detection: check if Nabla and Mitsuba share same CUDA device
- [ ] 3.7 Memory management: track Nabla tensor references, avoid leaks
- [ ] 3.8 Test: Transfer Dr.Jit tensor to Nabla and back, verify values match

## 4. JEPA Model Architecture (Nabla Python)

> Spec: `specs/jepa-model-architecture/`

- [ ] 4.1 Create SceneGraphEncoder in Nabla: geometry (Linear+MHA), materials (Embedding+Linear), lights (Linear), cross-attention fusion -> (1, 192)
- [ ] 4.2 Create RenderFeatureEncoder in Nabla: Conv2d stack (8->32->64->128, stride=2) + global pool + Linear(128, 192)
- [ ] 4.3 Implement cross-attention fusion: `F.scaled_dot_product_attention(render_latent, scene_latent, scene_latent)`
- [ ] 4.4 Create SceneDeltaEncoder in Nabla: Conv1d + MLP (smoothed -> 768 -> 192), ~155K params
- [ ] 4.5 Create ConditionalBlock in Nabla: AdaLN-zero conditioning via SiLU + Linear(192, 1152) -> 6 modulation params
- [ ] 4.6 Create ARPredictor in Nabla: 4 ConditionalBlock layers, 8 heads, input (1, 4, 192)
- [ ] 4.7 Create Decoder in Nabla: Linear + Conv2dTranspose stack (128->64->32->4) + Sigmoid
- [ ] 4.8 Create ConfidenceHead in Nabla: Linear(192,96)->SiLU->Linear(96,48)->SiLU->Linear(48,1)->Sigmoid
- [ ] 4.9 Implement SIGReg loss as custom Nabla kernel via `call_custom_kernel()` (UnaryOperation + vjp_rule)
- [ ] 4.10 Create OmenJEPA top-level module: compose encoder + arpredictor + decoder + confidence head
- [ ] 4.11 Implement model.encode(), model.decode(), model.predict_confidence(), model.merge()
- [ ] 4.12 Test: Forward pass with dummy data, verify output shapes (latent: (1,192), output: (1,H,W,4))

## 5. JEPA Inference

> Spec: `specs/jepa-inference/`

- [ ] 5.1 Implement JEPABridge class: load model via `nb.nn.Module.load_state_dict()`
- [ ] 5.2 Handle Nabla import failure and model load failure (graceful degradation)
- [ ] 5.3 Implement denoise inference: encode -> decode -> numpy output
- [ ] 5.4 Implement confidence prediction: encode -> decode + confidence head
- [ ] 5.5 Implement multires merge: dual encode -> merge -> decode
- [ ] 5.6 Optional `@nb.compile` for production speed (JIT compile model)
- [ ] 5.7 Handle inference failure: catch OOM, shape mismatch, return noisy input
- [ ] 5.8 Test: Run all inference paths with Cornell box data

## 6. Mode 1 - Denoiser

> Spec: `specs/denoiser-mode/`

- [ ] 6.1 Implement render_denoiser(scene, spp=4) in modes/denoiser.py
- [ ] 6.2 Render 4spp, extract scene graph, DLPack transfer, model denoise, return clean RGBA
- [ ] 6.3 Handle model unavailable: return raw render
- [ ] 6.4 Quality validation: SSIM, PSNR, artifact detection
- [ ] 6.5 Test: Cornell box at 4spp, SSIM > 0.90 vs 256spp

## 7. Mode 2 - Adaptive

> Spec: `specs/adaptive-guidance/`

- [ ] 7.1 Implement render_adaptive(scene, spp_target=128) in modes/adaptive.py
- [ ] 7.2 PASS 1: 4spp preview + confidence prediction
- [ ] 7.3 PASS 2: 128spp high-spp render
- [ ] 7.4 Confidence-weighted merge: `output = confidence * clean_preview + (1 - confidence) * high_spp`
- [ ] 7.5 Sample reduction reporting and benchmarking
- [ ] 7.6 Test: Cornell box, 4-8x sample reduction, SSIM > 0.95

## 8. Mode 3 - Multi-Resolution

> Spec: `specs/multires-merge/`

- [ ] 8.1 Implement render_multires(scene, scale=4) in modes/multires.py
- [ ] 8.2 PASS 1: 25% res 256spp, PASS 2: 100% res 4spp
- [ ] 8.3 Geometry-aware merge via model.merge() with scene graph guidance
- [ ] 8.4 Speedup measurement and quality validation (PSNR > 30dB)
- [ ] 8.5 Test: Cornell box, 8-16x speedup

## 9. Model Checkpointing & Scene Caching

> Spec: `specs/checkpoint-storage/`

- [ ] 9.1 Implement save/load via Nabla `state_dict()` / `load_state_dict()`
- [ ] 9.2 Save optimizer state (AdamW moments m, v) alongside model weights
- [ ] 9.3 Metadata JSON with architecture hash, version validation
- [ ] 9.4 Topology-based scene hashing for animation cache
- [ ] 9.5 Base model download on first use (bundled or remote)
- [ ] 9.6 Scene-specific fine-tuned model cache at `~/.cache/omen/models/scenes/<hash>/`
- [ ] 9.7 LoRA fine-tuning via Nabla built-in: `init_lora_adapter`, `lora_linear`, `merge_lora_weight`
- [ ] 9.8 GPU memory management: budget check, CPU fallback
- [ ] 9.9 Test: Save checkpoint, crash, resume from checkpoint

## 10. Temporal Coherence & JEPA World Model

> Spec: `specs/temporal-coherence/`

- [ ] 10.1 Create SceneDeltaEncoder for animation deltas (camera, objects, lights, births, materials)
- [ ] 10.2 Implement scene delta computation from frame-to-frame scene graph diff
- [ ] 10.3 Implement surprise detection: MSE z-score > 2 sigma on latent comparison
- [ ] 10.4 Implement auto-surprise for new scene elements (structural changes in graph)
- [ ] 10.5 Implement jump cut detection: translation > 1 unit or rotation > 45 deg
- [ ] 10.6 Implement modes/animation.py: frame 0 anchor, frames 1..N predict from 1spp dirty + history
- [ ] 10.7 History buffer management: CircularBuffer of size 3, clear on jump cut
- [ ] 10.8 Total loss: `L_pred + 0.09 * L_sigreg` (lambda=0.09 from lewm.yaml)
- [ ] 10.9 Periodic validation: every 5 predicted frames, render 1spp for surprise check
- [ ] 10.10 Test: Cornell box camera orbit 100 frames, surprise detection, jump cut recovery

## 11. Blender Scene Converter

> Spec: `specs/blender-scene-converter/`

- [ ] 11.1 Create `src/omen/converter/` directory
- [ ] 11.2 Implement `blend_to_mitsuba.py`: load .blend via `bpy` headless, iterate objects, extract geometry/materials/lights/camera
- [ ] 11.3 Convert Blender Principled BSDF -> mi.PrincipledBSDF (diffuse, roughness, metallic, specular, transmission, clearcoat, sheen)
- [ ] 11.4 Convert Blender Glass BSDF -> mi.DielectricBSDF, Emission -> mi.AreaLight
- [ ] 11.5 Convert lights: Point, Area, Sun, Spot -> Mitsuba equivalents
- [ ] 11.6 Handle texture maps: extract image paths, UV maps, normal maps, environment maps, packed textures
- [ ] 11.7 Apply modifiers before export (subdivision, mirror, boolean)
- [ ] 11.8 Handle hair/particles: export as curve primitives
- [ ] 11.9 Handle volumetrics: smoke, fire, fog -> mi.HomogeneousVolume or mi.GridVolume
- [ ] 11.10 Test: Convert a complex Blender scene, verify Mitsuba render matches expected output

## 12. Production Training Pipeline

> Spec: `specs/blender-scene-converter/` + `specs/training-gym/`

- [ ] 12.1 Verify AD variant: `mi.variant()` must contain `_ad_` (cuda_ad_rgb or llvm_ad_rgb)
- [ ] 12.2 Implement Dr.Jit training pair generators (4spp noisy + 256spp GT, same seed)
- [ ] 12.3 Implement Nabla PyTorch-style training loop: model.train(), loss.backward(), optimizer.step()
- [ ] 12.4 Use NablaAdamW(lr=5e-5, weight_decay=1e-3), gradient clip=1.0, BF16 precision
- [ ] 12.5 Cornell box bootstrap validation: denoiser(100) -> confidence(100) -> multires(100) -> temporal(200)
- [ ] 12.6 Build scene library: convert 50+ production Blender scenes across 7 categories (interiors, architecture, products, vehicles, characters, nature, volumes)
- [ ] 12.7 Batch training pair generation: per-scene random cameras (20+ angles), light variations (0.5x-2.0x), material perturbations, spp pairs (4/8/16/256)
- [ ] 12.8 Full pre-training: denoiser(5000) -> confidence(2000) -> multires(2000) -> temporal(5000) on production data
- [ ] 12.9 Validate on held-out production scenes (NOT Cornell box): SSIM > 0.92 all categories
- [ ] 12.10 Background fine-tuning: trigger on 3+ renders of same scene, LoRA rank=8, 50 iterations
- [ ] 12.11 Export base model: Nabla state_dict, SHA256, metadata JSON, bundle for distribution (~30MB)

## 13. Testing & Validation

> Cross-cutting: references all specs

- [ ] 13.1 Create tests for scene extraction, DLPack transfer, all 4 modes
- [ ] 13.2 Benchmark all modes vs baselines (uniform path tracing at equivalent quality)
- [ ] 13.3 Per-category quality validation: glass, SSS, volumes, hair, metals on production scenes
- [ ] 13.4 Verify: no artifacts, no hallucinations, temporal coherence across animation
