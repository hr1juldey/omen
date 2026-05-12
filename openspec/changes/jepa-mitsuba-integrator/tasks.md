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
- [x] 1.12 Create `src/omen/kernels/` for custom Mojo GPU kernels (SIGReg, merge, tile fingerprint)
- [x] 1.13 Create `src/omen/modes/` directory for render mode implementations
- [x] 1.14 Create `src/omen/modes/__init__.py`
- [x] 1.15 Create `src/omen/modes/denoiser.py`
- [x] 1.16 Create `src/omen/modes/adaptive.py`
- [x] 1.17 Create `src/omen/modes/multires.py`
- [x] 1.18 Create `src/omen/modes/animation.py`
- [x] 1.19 Create `src/omen/training/` directory
- [x] 1.20 Create `src/omen/training/trainer.py` (Nabla training loop with LoRA)
- [x] 1.21 Create `src/omen/training/data_gen.py` (Dr.Jit training pair generation)
- [x] 1.22 Create `src/python/render_engine.py` (Blender RenderEngine plugin — OmenRenderEngine + OmenProperties + scene graph extraction)
- [x] 1.23 Create `src/python/test_pattern.py` (test patterns + camera animation generators for training data)
- [x] 1.24 Upgrade `src/omen_integrator/core.py` (AOV extraction + JEPA denoise pipeline + DLPack transfer)
- [x] 1.25 Upgrade `src/omen_integrator/jepa.py` (model loading + tile fingerprint computation 23-dim)
- [x] 1.26 Upgrade `src/omen_integrator/gpu.py` (VRAM budget tracking + FP8 detection)
- [x] 1.27 Upgrade `src/omen_integrator/path.py` (AOV-aware path tracing reference)
- [x] 1.28 Create `src/omen/model/mla_skip.py` (MLA skip connection compression — 16× down/up projection)
- [x] 1.29 Create `src/omen/model/moe.py` (TileMoERouter + tile fingerprint + expert FFNs + shared expert)
- [x] 1.30 Create `src/omen/aov.py` (AOV pass reader + graceful degradation for missing passes)
- [x] 1.31 Create `src/omen/kernels/tile_fingerprint.mojo` (GPU-accelerated 8×8 tile histogram + variance + edge density)

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
- [x] 2.18 Test: Extract Cornell box, verify 2 meshes, 1 area light, 1 camera

## 3. DLPack Tensor Bridge

> Spec: `specs/mojo-cabi-bridge/` (renamed from C ABI to DLPack)

- [x] 3.1 Implement DLPack transfer: `nb.Tensor.from_dlpack(dr_tensor)` for GPU zero-copy
- [x] 3.2 Implement numpy fallback: `nb.ndarray(np_array)` for CPU renders
- [x] 3.3 Convert scene graph dict values to Nabla tensors
- [x] 3.4 Handle alpha channel addition: `nb.concatenate([render, ones], axis=-1)`
- [x] 3.5 Implement Nabla -> numpy conversion for output: `tensor.numpy()`
- [x] 3.6 GPU context detection: check if Nabla and Mitsuba share same CUDA device
- [x] 3.7 Memory management: track Nabla tensor references, avoid leaks
- [ ] 3.8 Test: Transfer Dr.Jit tensor to Nabla and back, verify values match

## 4. JEPA Model Architecture (Nabla Python)

> Spec: `specs/jepa-model-architecture/`

- [x] 4.1 Create SceneGraphEncoder in Nabla: geometry (Linear+MHA), materials (Embedding+Linear), lights (Linear), cross-attention fusion -> (1, 192)
- [x] 4.2 Create RenderFeatureEncoder in Nabla: nb.conv2d() functional API (NHWC, HWIO filters), stride=2 + global pool + Linear(128, 192)
- [x] 4.3 Implement cross-attention fusion: MultiHeadAttention + LayerNorm residual
- [x] 4.4 Create SceneDeltaEncoder in Nabla: Linear smoothing + MLP (smoothed -> 768 -> 192), ~155K params
- [x] 4.5 Create ConditionalBlock in Nabla: AdaLN-zero conditioning via SiLU + Linear(192, 1152) -> 6 modulation params
- [x] 4.6 Create ARPredictor in Nabla: 4 ConditionalBlock layers, 8 heads, input (1, 4, 192)
- [x] 4.7 Create Decoder in Nabla: Linear + nb.conv2d_transpose() functional API (NHWC, HWOC transposed filters) + Sigmoid
- [x] 4.8 Create ConfidenceHead in Nabla: Linear(192,96)->SiLU->Linear(96,48)->SiLU->Linear(48,1)->Sigmoid
- [x] 4.9 Implement SIGReg loss as variance proxy (TODO: full Epps-Pulley via custom Mojo kernel + call_custom_kernel)
- [x] 4.10 Create OmenJEPA top-level module: compose encoder + arpredictor + decoder + confidence head
- [x] 4.11 Implement model.encode(), model.decode(), model.predict_confidence(), model.merge(), model.compute_loss()
- [ ] 4.12 Test: Forward pass with dummy data, verify output shapes (latent: (1,192), output: (1,H,W,4))

## 5. MLA Skip Connection Compression

> Spec: `specs/jepa-model-architecture/` (MLA requirement)

- [x] 5.1 Create MLASkipConnection module in `model/mla_skip.py`: down-projection Linear(C, C//16) + up-projection Linear(C//16, C)
- [ ] 5.2 Integrate into U-Net encoder: each encoder level compresses features before storing as skip latent
- [ ] 5.3 Integrate into U-Net decoder: reconstruct skip features from latent before concatenation
- [ ] 5.4 Implement edge-aware compression: detect high normal discontinuity tiles, optionally store full-resolution features for those tiles, compressed for smooth regions
- [ ] 5.5 Memory tracking: verify 4K skip memory drops from ~6GB to ~375MB across all levels
- [ ] 5.6 Train MLA projections end-to-end with U-Net (W_down and W_up are learnable)
- [ ] 5.7 Test: Encode skip features, decode, verify reconstruction quality (cosine similarity > 0.95)

## 6. MoE Tile-Based Routing

> Spec: `specs/moe-tile-routing/`

- [x] 6.1 Create TileMoERouter in `model/moe.py`: tile fingerprint computation + expert selection
- [x] 6.2 Implement `compute_tile_fingerprint()`: reshape (B,H,W,8) aux buffers into 8×8 windows -> compute material histogram (8-dim) + normal variance (3-dim) + depth variance (1-dim) + edge density (1-dim) + dominant material (1-dim) + mean albedo (3-dim) = 17-dim fingerprint per tile
- [x] 6.3 Implement material expert routing: Linear(17, 8) on fingerprint -> top-K selection (K=2 medium, K=3 high)
- [x] 6.4 Implement light expert routing: Linear(17, 5) on fingerprint -> top-1 selection
- [x] 6.5 Implement geometry expert routing: Linear(17, 5) on fingerprint -> top-1 selection
- [x] 6.6 Route ALL 64 tokens in a tile together to selected experts (no per-pixel routing)
- [x] 6.7 Implement 8 material expert FFNs: diffuse, glossy/glass, metal, SSS/skin, volume/smoke, emissive, hair/fur, cloth
- [x] 6.8 Implement 5 light expert FFNs: point/spot, area, sun/directional, environment/HDRI, emissive geometry
- [x] 6.9 Implement 5 geometry expert FFNs: flat, curved/organic, edges/silhouettes, fine detail/hair, transparent
- [x] 6.10 Implement 4 motion expert FFNs: static, linear motion, fast motion/blur, occlusion boundary
- [x] 6.11 Implement 1 shared expert (always active, base denoising) — from DeepSeekMoE shared expert isolation
- [x] 6.12 Implement expert combination: `output = shared_expert(x) + Σ(weight_i × expert_i(x))`
- [x] 6.13 Implement mixed-tile handling: tiles spanning material boundaries activate multiple experts with histogram-weighted blending
- [x] 6.14 Implement motion-aware tile handling: tiles with high velocity variance activate deblur expert, occlusion tiles activate inpainting expert
- [x] 6.15 Implement auxiliary-loss-free load balancing (DeepSeek-V3): per-expert bias vector (now 23 experts), adjusted ±0.001 per training step, NO gradient
- [x] 6.16 Implement tier config: Fast = no MoE, Medium = MoE bottleneck top-2 (+motion top-1), High = MoE bottleneck+encoder top-3 (+motion top-1)
- [x] 6.17 Update all routing projections from Linear(17, n) to Linear(23, n) for 23-dim fingerprints
- [ ] 6.18 Test: Route synthetic tiles (pure diffuse, glass-metal boundary, hair edge, fast motion, occlusion) and verify correct expert activation

## 7. Mojo GPU Tile Fingerprint Kernel

> Uses: Mojo GPU fundamentals (TileTensor, row_major, enqueue_function)
> Spec: `specs/moe-tile-routing/` (tile fingerprint computation)

- [x] 7.1 Create `kernels/tile_fingerprint.mojo`: GPU kernel for computing 8×8 tile fingerprints from auxiliary buffer data
- [x] 7.2 Input: TileTensor[float32, row_major[H, W, 10]] (albedo(3) + normal(3) + depth(1) + material_id(1) + motion(2))
- [x] 7.3 Output: TileTensor[float32, row_major[H//8, W//8, 23]] (one 23-dim fingerprint per tile)
- [x] 7.4 Each GPU block processes one 8×8 tile: load 64 pixels into shared memory via `stack_allocation`
- [x] 7.5 Compute material histogram in shared memory: atomic counter per material_id, normalize by 64
- [x] 7.6 Compute normal variance: sum of squared deviations across 64 pixels for 3 normal channels
- [x] 7.7 Compute depth variance: sum of squared deviations for depth channel
- [x] 7.8 Compute edge density: count pixels where `||normal[i+1] - normal[i]|| > threshold` within tile
- [x] 7.9 Compute velocity mean and variance across tile via warp reduction (`warp.sum`)
- [x] 7.10 Compute velocity max and occlusion fraction (velocity discontinuity > threshold)
- [x] 7.11 Write 23-dim fingerprint to output tensor
- [x] 7.11 Bind and launch: `comptime kernel = tile_fingerprint_kernel[type_of(layout)]`, grid_dim=(W//8, H//8), block_dim=(8, 8)
- [x] 7.12 Expose to Python via `call_custom_kernel()` or DLPack interop (input numpy -> DeviceBuffer -> kernel -> DeviceBuffer -> numpy)
- [x] 7.13 Fallback path: pure numpy tile fingerprint computation if Mojo GPU not available
- [ ] 7.14 Test: Compare Mojo GPU fingerprint output vs numpy reference for 256×256 Cornell box aux buffers

## 8. AOV Auxiliary Buffer Handling

> Spec: `specs/moe-tile-routing/` (Blender-compatible auxiliary buffers)

- [x] 8.1 Create `aov.py` module: read auxiliary render passes from Mitsuba/Blender with graceful degradation
- [x] 8.2 Read albedo pass: Mitsuba `mi.render()` with albedo AOV or Blender Diffuse Color pass (3 channels)
- [x] 8.3 Read normal pass: Mitsuba normal AOV or Blender Normal pass (3 channels, world-space)
- [x] 8.4 Read depth pass: Mitsuba depth AOV or Blender Depth/Z pass (1 channel)
- [x] 8.5 Read material ID pass: Cryptomatte from Blender (integer per pixel), or BSDF type index from Mitsuba scene extraction
- [x] 8.6 Read motion vector pass: Blender `scene.render.use_pass_vector = True` → (H, W, 2), Mitsuba AOV `motion:vector`
- [x] 8.7 **Graceful degradation when AOVs missing**: zero-fill missing channels and flag `aov_available=False` for each pass
  - No albedo → fill zeros, material experts rely more on shared expert
  - No normals → fill zeros, geometry routing disabled, rely on shared + material experts
  - No depth → fill zeros, transparency detection disabled
  - No material_id → all pixels get material_id=0 (diffuse default), histogram becomes uniform, shared expert dominates
  - No motion vectors → fill zeros, motion experts never activated, temporal reprojection disabled, static denoise mode
- [ ] 8.8 **Render-time AOV enabling for Mitsuba**: when calling `mi.render()`, configure integrator to output auxiliary AOVs:
  - `mi.load_dict({'type': 'aov', 'aovs': 'albedo:color,normal:color,depth:color,motion:vector'})` wrapping the path integrator
  - This produces auxiliary channels WITHOUT requiring user to set up custom passes
- [ ] 8.9 **Render-time AOV enabling for Blender**: if integrated with Blender, enable required render passes programmatically:
  - `scene.render.use_pass_normal = True`, `scene.render.use_pass_z = True`, `scene.render.use_pass_vector = True`, etc.
- [x] 8.10 Pack auxiliary buffers into single (H, W, 10) tensor for tile fingerprint computation (8 original + 2 motion)
- [x] 8.11 Log AOV status: "AOV available: albedo=yes, normal=yes, depth=no, material_id=no, motion=no — using degraded mode"
- [ ] 8.12 Test: Run denoiser with ALL AOVs missing → verify it still works (shared expert only), log degradation warning
- [ ] 8.13 Test: Run denoiser with partial AOVs (only albedo, no motion) → verify degraded but functional
- [ ] 8.14 Test: Run denoiser with motion vectors → verify motion experts activate on moving tiles

## 9. JEPA Inference

> Spec: `specs/jepa-inference/`

- [x] 9.1 Implement JEPABridge class: load model via `nb.nn.Module.load_state_dict()`
- [x] 9.2 Handle Nabla import failure and model load failure (graceful degradation)
- [x] 9.3 Implement denoise inference: encode -> decode -> numpy output
- [x] 9.4 Implement confidence prediction: encode -> decode + confidence head
- [x] 9.5 Implement multires merge: dual encode -> merge -> decode
- [ ] 9.6 Optional `@nb.compile` for production speed (JIT compile model)
- [x] 9.7 Handle inference failure: catch OOM, shape mismatch, return noisy input
- [ ] 9.8 Test: Run all inference paths with Cornell box data

## 10. Mode 1 - Denoiser

> Spec: `specs/denoiser-mode/`

- [x] 10.1 Implement render_denoiser(scene, spp=4) in modes/denoiser.py
- [x] 10.2 Render 4spp, configure Mitsuba AOV integrator for albedo/normal/depth passes
- [x] 10.3 Extract scene graph, DLPack transfer render + aux buffers, run tile fingerprint computation
- [x] 10.4 Model denoise with tile-based MoE routing (8×8 cryptomatte masks), return clean RGBA
- [x] 10.5 Handle model unavailable: return raw render
- [x] 10.6 Handle missing AOVs: zero-fill missing channels, log degradation, proceed with shared-expert-only routing
- [x] 10.7 Quality validation: SSIM, PSNR, artifact detection
- [x] 10.8 Test: Cornell box at 4spp, SSIM > 0.90 vs 256spp
- [x] 10.9 Test: Cornell box at 4spp with NO auxiliary passes, verify degraded quality still beats raw 4spp

## 11. Mode 2 - Adaptive

> Spec: `specs/adaptive-guidance/`

- [ ] 11.1 Implement render_adaptive(scene, spp_target=128) in modes/adaptive.py
- [ ] 11.2 PASS 1: 4spp preview + AOV extraction + confidence prediction
- [ ] 11.3 PASS 2: 128spp high-spp render
- [ ] 11.4 Confidence-weighted merge: `output = confidence * clean_preview + (1 - confidence) * high_spp`
- [ ] 11.5 Sample reduction reporting and benchmarking
- [ ] 11.6 Test: Cornell box, 4-8x sample reduction, SSIM > 0.95

## 12. Mode 3 - Multi-Resolution

> Spec: `specs/multires-merge/`

- [ ] 12.1 Implement render_multires(scene, scale=4) in modes/multires.py
- [ ] 12.2 PASS 1: 25% res 256spp, PASS 2: 100% res 4spp
- [ ] 12.3 Geometry-aware merge via model.merge() with scene graph guidance
- [ ] 12.4 Speedup measurement and quality validation (PSNR > 30dB)
- [ ] 12.5 Test: Cornell box, 8-16x speedup

## 13. Model Checkpointing & Scene Caching

> Spec: `specs/checkpoint-storage/`

- [x] 13.1 Implement save/load via `state_dict()` / `load_state_dict()` + numpy .npz serialization
- [ ] 13.2 Save optimizer state (AdamW moments m, v) alongside model weights
- [ ] 13.3 Metadata JSON with architecture hash (e.g., "OmenUNet-C96-5lvl-Swin768-MoE_top2_MLA16_AR-4-16-64-2048"), version validation
- [ ] 13.4 Topology-based scene hashing for animation cache
- [ ] 13.5 Base model download on first use (bundled or remote)
- [ ] 13.6 Scene-specific fine-tuned model cache at `~/.cache/omen/models/scenes/<hash>/`
- [x] 13.7 LoRA fine-tuning via Nabla built-in: `init_lora_adapter`, `lora_linear`, `merge_lora_weight`
- [ ] 13.8 GPU memory management: budget check (700MB inference with MLA, 1.6GB training), CPU fallback
- [ ] 13.9 Test: Save checkpoint, crash, resume from checkpoint

## 14. Temporal Coherence & JEPA World Model

> Spec: `specs/temporal-coherence/`

- [ ] 14.1 Create SceneDeltaEncoder for animation deltas (camera, objects, lights, births, materials)
- [ ] 14.2 Implement scene delta computation from frame-to-frame scene graph diff
- [ ] 14.3 Implement surprise detection: MSE z-score > 2 sigma on latent comparison
- [ ] 14.4 Implement auto-surprise for new scene elements (structural changes in graph)
- [ ] 14.5 Implement jump cut detection: translation > 1 unit or rotation > 45 deg
- [ ] 14.6 Implement modes/animation.py: frame 0 anchor, frames 1..N predict from 1spp dirty + history
- [ ] 14.7 History buffer management: CircularBuffer of size 3, clear on jump cut
- [ ] 14.8 Total loss: `L_pred + 0.1 * L_energy + 0.09 * L_sigreg` (energy conservation added)
- [ ] 14.9 Periodic validation: every 5 predicted frames, render 1spp for surprise check
- [ ] 14.10 Test: Cornell box camera orbit 100 frames, surprise detection, jump cut recovery

## 15. Blender Scene Converter

> Spec: `specs/blender-scene-converter/`

- [ ] 15.1 Create `src/omen/converter/` directory
- [ ] 15.2 Implement `blend_to_mitsuba.py`: load .blend via `bpy` headless, iterate objects, extract geometry/materials/lights/camera
- [ ] 15.3 Convert Blender Principled BSDF -> mi.PrincipledBSDF (diffuse, roughness, metallic, specular, transmission, clearcoat, sheen)
- [ ] 15.4 Convert Blender Glass BSDF -> mi.DielectricBSDF, Emission -> mi.AreaLight
- [ ] 15.5 Convert lights: Point, Area, Sun, Spot -> Mitsuba equivalents
- [ ] 15.6 Handle texture maps: extract image paths, UV maps, normal maps, environment maps, packed textures
- [ ] 15.7 Apply modifiers before export (subdivision, mirror, boolean)
- [ ] 15.8 Handle hair/particles: export as curve primitives
- [ ] 15.9 Handle volumetrics: smoke, fire, fog -> mi.HomogeneousVolume or mi.GridVolume
- [ ] 15.10 Test: Convert a complex Blender scene, verify Mitsuba render matches expected output

## 16. Production Training Pipeline

> Spec: `specs/training-gym/`

- [ ] 16.1 Verify AD variant: `mi.variant()` must contain `_ad_` (cuda_ad_rgb or llvm_ad_rgb)
- [ ] 16.2 Implement Dr.Jit training pair generators (4spp noisy + 256spp GT, same seed) with AOV auxiliary passes
- [ ] 16.3 Implement Nabla PyTorch-style training loop: model.train(), loss.backward(), optimizer.step()
- [ ] 16.4 Use NablaAdamW(lr=5e-5, weight_decay=1e-3), gradient clip=1.0, BF16 precision
- [ ] 16.5 Train MLA skip compression end-to-end: W_down and W_up projections learn with U-Net gradients
- [ ] 16.6 Train MoE tile routing: fingerprint projections learn which tiles belong to which experts
- [ ] 16.7 MoE load balancing: after each training step, update per-expert bias ±0.001 based on tile routing counts
- [ ] 16.8 Energy conservation loss: `L_total = L_denoise + 0.1 * L_energy + 0.09 * L_sigreg`
- [ ] 16.9 Train motion experts: generate training pairs with motion blur enabled (animated camera + objects), motion vectors as AOV
- [ ] 16.10 Train temporal reprojection: supervise warped previous frame blending with ground truth
- [ ] 16.11 Cornell box bootstrap validation: denoiser(100) -> confidence(100) -> multires(100) -> motion(100) -> temporal(200)
- [ ] 16.12 Build scene library: convert 50+ production Blender scenes across 7 categories (interiors, architecture, products, vehicles, characters, nature, volumes), include animated variants
- [ ] 16.13 Batch training pair generation: per-scene random cameras (20+ angles), light variations (0.5x-2.0x), material perturbations, spp pairs (4/8/16/256), motion blur variants
- [ ] 16.14 Full pre-training: denoiser(5000) -> confidence(2000) -> multires(2000) -> motion(1000) -> temporal(5000) on production data
- [ ] 16.15 Validate on held-out production scenes (NOT Cornell box): SSIM > 0.92 all categories
- [ ] 16.16 Background fine-tuning: trigger on 3+ renders of same scene, LoRA rank=8, 50 iterations
- [ ] 16.17 Export base model: Nabla state_dict, SHA256, metadata JSON, bundle for distribution (~30MB)
- [ ] 16.18 Validate with degraded AOVs: train on scenes with full AOVs, test with partial AOVs missing (including no motion vectors) to ensure graceful degradation

## 17. Testing & Validation

> Cross-cutting: references all specs

- [ ] 17.1 Create tests for scene extraction, DLPack transfer, all 4 modes
- [ ] 17.2 Benchmark all modes vs baselines (uniform path tracing at equivalent quality)
- [ ] 17.3 Per-category quality validation: glass, SSS, volumes, hair, metals on production scenes
- [ ] 17.4 Verify: no artifacts, no hallucinations, temporal coherence across animation
- [ ] 17.5 Verify tile-based MoE routing: no seam artifacts at material boundaries (glass-metal edge tiles)
- [ ] 17.6 Verify MLA skip reconstruction: cosine similarity > 0.95 between full and compressed-then-reconstructed features
- [ ] 17.7 Verify AOV degradation: quality degrades gracefully, never crashes when passes missing
- [ ] 17.8 Verify Mojo GPU tile fingerprint kernel matches numpy reference within tolerance
- [ ] 17.9 Memory validation: 4K inference under 700MB with MLA compression, 1.6GB training budget
- [ ] 17.10 Verify motion blur handling: denoise motion-blurred frames, verify motion experts activate correctly, no ghosting
- [ ] 17.11 Verify temporal reprojection: compare with/without previous frame reuse, measure quality delta on animation
- [ ] 17.12 Verify motion vector degradation: denoise motion-blurred scene WITHOUT motion vectors, verify no crash, quality degrades gracefully

## 18. Motion Blur & Temporal Reprojection

> Spec: `specs/motion-blur-handling/`

- [ ] 18.1 Create `motion.py` module: motion vector processing + temporal reprojection
- [ ] 18.2 Implement motion vector reading: extract (H, W, 2) from Blender vector pass or Mitsuba AOV
- [ ] 18.3 Implement temporal reprojection: `bilinear_warp(prev_clean, motion_vectors)` → reprojected frame aligned to current coordinates
- [ ] 18.4 Compute motion coherence per pixel: `coherence = 1.0 - clamp(length(motion) / max_velocity, 0, 1)`
- [ ] 18.5 Compute occlusion mask: detect velocity discontinuity between neighboring pixels
- [ ] 18.6 Compute reprojection weight: `alpha = prev_confidence × coherence × (1 - occluded)`
- [ ] 18.7 Merge reprojected + current: `output = alpha * reprojected + (1 - alpha) * current_noisy`
- [ ] 18.8 Handle first frame: no previous frame → skip reprojection, single-frame denoise
- [ ] 18.9 Handle jump cut: clear prev frame buffer, disable reprojection
- [ ] 18.10 Handle missing motion vectors: fill zeros, disable reprojection, log "static denoise mode"
- [ ] 18.11 Extend tile fingerprint from 17-dim to 23-dim: add velocity_mean(2) + velocity_var(2) + velocity_max(1) + occlusion_frac(1)
- [ ] 18.12 Implement motion expert routing: Linear(23, 4) on fingerprint → select static/linear/fast/occlusion expert
- [ ] 18.13 Test: Animated Cornell box (camera orbit) with motion blur, verify temporal reprojection reduces noise vs single-frame
- [ ] 18.14 Test: Fast-moving object with occlusion, verify no ghosting artifacts in disoccluded regions
- [ ] 18.15 Test: Motion-blurred scene WITHOUT motion vectors, verify graceful fallback to static mode

## 19. Performance Optimizations

> Design: `design.md` Decision 15

### 19a: Async Pipeline (DualPipe)

- [ ] 19a.1 Implement double-buffered render pipeline: Thread 1 = Mitsuba render, Thread 2 = JEPA denoise
- [ ] 19a.2 Bounded queue (size 2) between render and denoise threads
- [ ] 19a.3 GPU memory ping-pong: allocate 2× inference buffers for overlapping frames
- [ ] 19a.4 Only enable async when VRAM > 2× single-frame inference budget, fall back to sequential otherwise
- [ ] 19a.5 Benchmark: measure throughput gain on 100-frame animation sequence (target ~1.8×)

### 19b: Speculative Multi-Frame Prediction (MTP)

- [ ] 19b.1 Implement MTP heads on ARPredictor: predict N+1, N+2, N+3 from shared trunk
- [ ] 19b.2 Implement verification: render 1spp of predicted frame, check SSIM > 0.85
- [ ] 19b.3 If prediction verified: skip full render for that frame, use predicted output
- [ ] 19b.4 If prediction fails: fall back to normal render for remaining predicted frames
- [ ] 19b.5 Only active in Mode 4 (animation) with history buffer populated
- [ ] 19b.6 Benchmark: measure effective frame rate gain on camera orbit sequence (target 1.8×)

### 19c: Scene Latent Caching with Smart Invalidation

- [ ] 19c.1 Create `scene/latent_cache.py`: two-level cache (topology_hash + dynamic_hash)
- [ ] 19c.2 Implement topology_hash: hash of face connectivity + material TYPES + light TYPES (excludes positions, values)
- [ ] 19c.3 Implement dynamic_hash: hash of vertex positions + light intensities + material VALUES
- [ ] 19c.4 Cache strategy: if topology_hash matches cached → incremental update via SceneDeltaEncoder (delta encode ~5ms vs full ~30ms)
- [ ] 19c.5 Smart invalidation: full re-encode on births, material type changes, vertex count changes, light additions
- [ ] 19c.6 Small delta path: object moves, light intensity changes, material param values change → delta update only
- [ ] 19c.7 Test: Static scene, verify latent reused across frames (0ms re-encode after first frame)
- [ ] 19c.8 Test: Animated object (position changes only), verify delta encode works correctly
- [ ] 19c.9 Test: New object added mid-animation, verify cache invalidation triggers full re-encode

### 19d: Progressive Adaptive Sampling

- [ ] 19d.1 Implement 2spp ultra-cheap preview (instead of 4spp)
- [ ] 19d.2 Build per-tile spp map from confidence: high-conf tiles → 0 extra spp, low-conf → variable 4-128 spp
- [ ] 19d.3 Implement multi-pass variable spp rendering (render at different spp levels, composite by confidence mask)
- [ ] 19d.4 Benchmark: target 10-16× sample reduction on scenes with >50% easy regions
- [ ] 19d.5 Test: Cornell box, verify progressive adaptive beats standard adaptive on sample reduction

### 19e: Early Exit in U-Net Decoder

- [ ] 19e.1 Implement per-tile confidence estimation at U-Net bottleneck output
- [ ] 19e.2 Skip deeper decoder levels for high-confidence flat tiles (confidence > 0.9, low normal variance)
- [ ] 19e.3 Continue full decoder depth for complex tiles (edges, mixed materials, motion blur)
- [ ] 19e.4 Only enable for High tier at 4K (decoder is bottleneck only at high res)
- [ ] 19e.5 Benchmark: measure inference speedup on scene with >60% flat surfaces

### 19f: FP8 Mixed Precision Inference

- [ ] 19f.1 Enable FP8 (E4M3) for U-Net encoder Conv2d weights with per-tile dynamic scaling
- [ ] 19f.2 Keep BF16 for Swin attention QKV (softmax needs precision)
- [ ] 19f.3 Enable FP8 for MoE expert FFN weights (experts are small, FP8 saves VRAM)
- [ ] 19f.4 Enable FP8 for U-Net decoder Conv2d weights
- [ ] 19f.5 Auto-detect GPU FP8 support (Ada Lovelace / Hopper+), fall back to BF16 on older GPUs
- [ ] 19f.6 Validate: PSNR drop < 0.5dB vs BF16 baseline on Cornell box and production scenes
- [ ] 19f.7 Benchmark: VRAM (700MB → ~400MB) and matmul speedup (target 1.5-2×)

## 20. Blender Plugin (RenderEngine + Shared Node System)

> Spec: `specs/blender-plugin/` (NEW)
> References: `src/python/render_engine.py`, `src/python/test_pattern.py`
> Blender source: `source/blender/render/RE_engine.h`, `intern/cycles/blender/`

- [ ] 20.1 Register `OmenRenderEngine(bpy.types.RenderEngine)` with `bl_idname = "OMEN_RENDER"`, `bl_use_eevee_viewport = True`
- [ ] 20.2 Implement `update_render_passes()` — declare Combined(4), Depth(1), Diffuse Color(3), Specular Color(3), Normal(3), Vector(4), CryptoMaterial(4)
- [ ] 20.3 Implement `render(depsgraph)` — scene graph extraction → Mitsuba render → JEPA denoise → return result via `self.begin_result()`
- [ ] 20.4 Implement `view_update()` / `view_draw()` — delegate to EEVEE for viewport preview (bl_use_eevee_viewport = True)
- [ ] 20.5 Create `OmenProperties(bpy.types.PropertyGroup)` — spp, spp_gt, use_denoiser, model_tier (Fast/Medium/High), model_path, export_motion_vectors, export_cryptomatte
- [ ] 20.6 Register properties on `bpy.types.Scene.omen_props` via PointerProperty
- [ ] 20.7 Implement `_extract_scene_graph(depsgraph)` — iterate depsgraph.objects, extract meshes/lights/cameras per-object
- [ ] 20.8 Implement `_extract_mesh(obj)` — `obj.to_mesh()` → vertices, faces, normals, UVs, material_indices, transform (4x4 matrix_world)
- [ ] 20.9 Implement `_extract_material(mat)` — read `mat.node_tree` (bNodeTree, NTREE_SHADER) → nodes (bNode) + links (bNodeLink) + input socket values
- [ ] 20.10 Implement `_extract_light(obj)` — light.type, energy, color, transform
- [ ] 20.11 Implement `_extract_camera(obj)` — fov, clip_start, clip_end, transform
- [ ] 20.12 Implement `_get_socket_value(socket)` — extract default values from bNodeSocket (VALUE/RGBA/VECTOR/INT/BOOLEAN types)
- [ ] 20.13 Implement `_render_mitsuba()` — convert scene graph → Mitsuba scene dict → mi.render() → return pixels
- [ ] 20.14 Implement `_denoise()` — load JEPA model, stack noisy RGBA + AOV buffers (14ch), forward pass → clean RGBA
- [ ] 20.15 Create Blender addon `__init__.py` with register()/unregister() functions and panel UI for OmenProperties
- [ ] 20.16 Test: Install addon in Blender, select Omen render engine, render test scene, verify gradient output
- [ ] 20.17 Test: Open Blender demo file (Classroom), verify scene graph extraction produces correct mesh/material/light/camera counts
- [ ] 20.18 Test: Toggle AOV passes (motion vectors, cryptomatte on/off), verify passes appear/disappear correctly

## 21. Blender Demo Files — Training Data Generation

> Spec: `specs/blender-scene-converter/` (UPDATED)
> References: `src/omen/training/data_gen.py`, `src/python/test_pattern.py` (camera animation)
> Scene list: `docs/research/research-paper-grade-evaluation.md` Section 11

- [ ] 21.1 Download Blender demo files (15 scenes): Singularity(670MB), DOGWALK(383MB), Gold(300MB), Barbershop(280MB), Laundromat(230MB), Classroom(72MB), Barcelona(24MB), ItalianFlat(368MB), Charge(1.4GB), MonsterBed, HairStyles, AnimalFur, EmberForest, WaspBot, NishitaSky
- [ ] 21.2 Create `src/omen/training/scene_library/` directory with scene manifest JSON listing all 15 scenes with category, license, expected complexity
- [ ] 21.3 Implement `scene_library/download.py` — automated download of Blender demo files from blender.org (wget/curl)
- [ ] 21.4 Implement `scene_library/validate.py` — open each .blend headless via bpy, verify opens without error, log object/material/light/camera counts
- [ ] 21.5 Implement camera animation generators in `test_pattern.py`: orbit(), dolly(), pan(), flythrough() — return Mitsuba-compatible look_at transforms
- [ ] 21.6 Implement `training/batch_generate.py` — for each scene: iterate 30-50 camera positions, render noisy (1-4 spp) + GT (256-4096 spp) pairs via Mitsuba
- [ ] 21.7 Extract AOV buffers per training pair: albedo(3), normal(3), depth(1), motion_vectors(2), cryptomatte_material(4), cryptomatte_object(4)
- [ ] 21.8 Compute tile fingerprints (23-dim) per training pair using `omen_integrator.jepa.compute_tile_fingerprint()`
- [ ] 21.9 Store training pairs as HDF5: group per scene, datasets per frame (noisy, gt, albedo, normal, depth, motion, cryptomatte, fingerprint)
- [ ] 21.10 Implement scene diversity validation: compute material type distribution, light count distribution, geometric complexity stats across dataset — ensure no single category >40%
- [ ] 21.11 Generate random light intensity variations per render (0.5x-2.0x baseline), random material perturbations (roughness ±0.1, color shift ±5%)
- [ ] 21.12 Generate motion blur variants: enable Blender motion blur, render with animated objects, extract motion vectors as AOV
- [ ] 21.13 Target: 500-750 total training pairs across 15 scenes (30-50 frames per scene × 15 scenes)
- [ ] 21.14 Validate dataset: run scene_extractor on each pair, verify scene graphs are non-empty and consistent with source .blend
- [ ] 21.15 Split dataset: 80% train, 10% validation, 10% hold-out test (stratified by scene category)

## 22. Research Paper — Experimental Validation

> Reference: `docs/research/research-paper-grade-evaluation.md`
> Target venue: EGSR 2026 or HPG 2026 (Paper 1: tile-based MoE routing)

- [ ] 22.1 Implement evaluation metrics: PSNR, SSIM, LPIPS, FLIP (NVIDIA's rendering perceptual metric)
- [ ] 22.2 Run baseline comparisons: render hold-out test set with OIDN 2.x, OptiX denoiser, KPCN — compute all metrics
- [ ] 22.3 Run Omen (full model) on same hold-out test set — compute all metrics
- [ ] 22.4 Ablation 1: Full model vs no MoE (single FFN with same param budget) — quantify MoE routing benefit
- [ ] 22.5 Ablation 2: Tile-based routing (8×8) vs per-pixel routing — measure seam artifacts, routing quality
- [ ] 22.6 Ablation 3: Scene graph conditioning vs raw AOV only — quantify JEPA scene understanding contribution
- [ ] 22.7 Ablation 4: With vs without motion experts — measure temporal flicker reduction on animation sequences
- [ ] 22.8 Ablation 5: With vs without MLA skip compression — quality vs memory at 4K (target: cosine sim > 0.95)
- [ ] 22.9 Generate convergence curves: quality vs spp plots for each baseline and Omen
- [ ] 22.10 Generate speed/quality Pareto: inference time vs SSIM frontier plot
- [ ] 22.11 Generate expert activation visualizations: heatmap showing which MoE experts fire on which scene regions
- [ ] 22.12 Analyze failure cases: caustics, volumetrics, very low spp (<2), complex hair — document where Omen breaks
- [ ] 22.13 Write Paper 1 draft: "Tile-Based Mixture-of-Experts Routing for Monte Carlo Denoising" — target EGSR/HPG
- [ ] 22.14 Create supplementary material: side-by-side comparison images, video of temporal denoising, expert routing animation
