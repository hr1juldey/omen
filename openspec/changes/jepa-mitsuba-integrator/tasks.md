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
- [ ] 1.22 Create `src/omen/model/mla_skip.py` (MLA skip connection compression — 16× down/up projection)
- [ ] 1.23 Create `src/omen/model/moe.py` (TileMoERouter + tile fingerprint + expert FFNs + shared expert)
- [ ] 1.24 Create `src/omen/aov.py` (AOV pass reader + graceful degradation for missing passes)
- [ ] 1.25 Create `src/omen/kernels/tile_fingerprint.mojo` (GPU-accelerated 8×8 tile histogram + variance + edge density)

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

- [ ] 5.1 Create MLASkipConnection module in `model/mla_skip.py`: down-projection Linear(C, C//16) + up-projection Linear(C//16, C)
- [ ] 5.2 Integrate into U-Net encoder: each encoder level compresses features before storing as skip latent
- [ ] 5.3 Integrate into U-Net decoder: reconstruct skip features from latent before concatenation
- [ ] 5.4 Implement edge-aware compression: detect high normal discontinuity tiles, optionally store full-resolution features for those tiles, compressed for smooth regions
- [ ] 5.5 Memory tracking: verify 4K skip memory drops from ~6GB to ~375MB across all levels
- [ ] 5.6 Train MLA projections end-to-end with U-Net (W_down and W_up are learnable)
- [ ] 5.7 Test: Encode skip features, decode, verify reconstruction quality (cosine similarity > 0.95)

## 6. MoE Tile-Based Routing

> Spec: `specs/moe-tile-routing/`

- [ ] 6.1 Create TileMoERouter in `model/moe.py`: tile fingerprint computation + expert selection
- [ ] 6.2 Implement `compute_tile_fingerprint()`: reshape (B,H,W,8) aux buffers into 8×8 windows -> compute material histogram (8-dim) + normal variance (3-dim) + depth variance (1-dim) + edge density (1-dim) + dominant material (1-dim) + mean albedo (3-dim) = 17-dim fingerprint per tile
- [ ] 6.3 Implement material expert routing: Linear(17, 8) on fingerprint -> top-K selection (K=2 medium, K=3 high)
- [ ] 6.4 Implement light expert routing: Linear(17, 5) on fingerprint -> top-1 selection
- [ ] 6.5 Implement geometry expert routing: Linear(17, 5) on fingerprint -> top-1 selection
- [ ] 6.6 Route ALL 64 tokens in a tile together to selected experts (no per-pixel routing)
- [ ] 6.7 Implement 8 material expert FFNs: diffuse, glossy/glass, metal, SSS/skin, volume/smoke, emissive, hair/fur, cloth
- [ ] 6.8 Implement 5 light expert FFNs: point/spot, area, sun/directional, environment/HDRI, emissive geometry
- [ ] 6.9 Implement 5 geometry expert FFNs: flat, curved/organic, edges/silhouettes, fine detail/hair, transparent
- [ ] 6.10 Implement 1 shared expert (always active, base denoising) — from DeepSeekMoE shared expert isolation
- [ ] 6.11 Implement expert combination: `output = shared_expert(x) + Σ(weight_i × expert_i(x))`
- [ ] 6.12 Implement mixed-tile handling: tiles spanning material boundaries activate multiple experts with histogram-weighted blending
- [ ] 6.13 Implement auxiliary-loss-free load balancing (DeepSeek-V3): per-expert bias vector, adjusted ±0.001 per training step, NO gradient
- [ ] 6.14 Implement tier config: Fast = no MoE, Medium = MoE bottleneck top-2, High = MoE bottleneck+encoder top-3
- [ ] 6.15 Test: Route synthetic tiles (pure diffuse, glass-metal boundary, hair edge) and verify correct expert activation

## 7. Mojo GPU Tile Fingerprint Kernel

> Uses: Mojo GPU fundamentals (TileTensor, row_major, enqueue_function)
> Spec: `specs/moe-tile-routing/` (tile fingerprint computation)

- [ ] 7.1 Create `kernels/tile_fingerprint.mojo`: GPU kernel for computing 8×8 tile fingerprints from auxiliary buffer data
- [ ] 7.2 Input: TileTensor[float32, row_major[H, W, 8]] (albedo(3) + normal(3) + depth(1) + material_id(1))
- [ ] 7.3 Output: TileTensor[float32, row_major[H//8, W//8, 17]] (one 17-dim fingerprint per tile)
- [ ] 7.4 Each GPU block processes one 8×8 tile: load 64 pixels into shared memory via `stack_allocation`
- [ ] 7.5 Compute material histogram in shared memory: atomic counter per material_id, normalize by 64
- [ ] 7.6 Compute normal variance: sum of squared deviations across 64 pixels for 3 normal channels
- [ ] 7.7 Compute depth variance: sum of squared deviations for depth channel
- [ ] 7.8 Compute edge density: count pixels where `||normal[i+1] - normal[i]|| > threshold` within tile
- [ ] 7.9 Compute dominant material and mean albedo via warp reduction (`warp.sum`, `warp.max`)
- [ ] 7.10 Write 17-dim fingerprint to output tensor
- [ ] 7.11 Bind and launch: `comptime kernel = tile_fingerprint_kernel[type_of(layout)]`, grid_dim=(W//8, H//8), block_dim=(8, 8)
- [ ] 7.12 Expose to Python via `call_custom_kernel()` or DLPack interop (input numpy -> DeviceBuffer -> kernel -> DeviceBuffer -> numpy)
- [ ] 7.13 Fallback path: pure numpy tile fingerprint computation if Mojo GPU not available
- [ ] 7.14 Test: Compare Mojo GPU fingerprint output vs numpy reference for 256×256 Cornell box aux buffers

## 8. AOV Auxiliary Buffer Handling

> Spec: `specs/moe-tile-routing/` (Blender-compatible auxiliary buffers)

- [ ] 8.1 Create `aov.py` module: read auxiliary render passes from Mitsuba/Blender with graceful degradation
- [ ] 8.2 Read albedo pass: Mitsuba `mi.render()` with albedo AOV or Blender Diffuse Color pass (3 channels)
- [ ] 8.3 Read normal pass: Mitsuba normal AOV or Blender Normal pass (3 channels, world-space)
- [ ] 8.4 Read depth pass: Mitsuba depth AOV or Blender Depth/Z pass (1 channel)
- [ ] 8.5 Read material ID pass: Cryptomatte from Blender (integer per pixel), or BSDF type index from Mitsuba scene extraction
- [ ] 8.6 **Graceful degradation when AOVs missing**: zero-fill missing channels and flag `aov_available=False` for each pass
  - No albedo → fill zeros, material experts rely more on shared expert
  - No normals → fill zeros, geometry routing disabled, rely on shared + material experts
  - No depth → fill zeros, transparency detection disabled
  - No material_id → all pixels get material_id=0 (diffuse default), histogram becomes uniform, shared expert dominates
- [ ] 8.7 **Render-time AOV enabling for Mitsuba**: when calling `mi.render()`, configure integrator to output auxiliary AOVs:
  - `mi.load_dict({'type': 'aov', 'aovs': 'albedo:color,normal:color,depth:color'})` wrapping the path integrator
  - This produces auxiliary channels WITHOUT requiring user to set up custom passes
- [ ] 8.8 **Render-time AOV enabling for Blender**: if integrated with Blender, enable required render passes programmatically:
  - `scene.render.use_pass_normal = True`, `scene.render.use_pass_z = True`, etc.
- [ ] 8.9 Pack auxiliary buffers into single (H, W, 8) tensor for tile fingerprint computation
- [ ] 8.10 Log AOV status: "AOV available: albedo=yes, normal=yes, depth=no, material_id=no — using degraded mode"
- [ ] 8.11 Test: Run denoiser with ALL AOVs missing → verify it still works (shared expert only), log degradation warning
- [ ] 8.12 Test: Run denoiser with partial AOVs (only albedo) → verify degraded but functional

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

- [ ] 10.1 Implement render_denoiser(scene, spp=4) in modes/denoiser.py
- [ ] 10.2 Render 4spp, configure Mitsuba AOV integrator for albedo/normal/depth passes
- [ ] 10.3 Extract scene graph, DLPack transfer render + aux buffers, run tile fingerprint computation
- [ ] 10.4 Model denoise with tile-based MoE routing (8×8 cryptomatte masks), return clean RGBA
- [ ] 10.5 Handle model unavailable: return raw render
- [ ] 10.6 Handle missing AOVs: zero-fill missing channels, log degradation, proceed with shared-expert-only routing
- [ ] 10.7 Quality validation: SSIM, PSNR, artifact detection
- [ ] 10.8 Test: Cornell box at 4spp, SSIM > 0.90 vs 256spp
- [ ] 10.9 Test: Cornell box at 4spp with NO auxiliary passes, verify degraded quality still beats raw 4spp

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
- [ ] 16.9 Cornell box bootstrap validation: denoiser(100) -> confidence(100) -> multires(100) -> temporal(200)
- [ ] 16.10 Build scene library: convert 50+ production Blender scenes across 7 categories (interiors, architecture, products, vehicles, characters, nature, volumes)
- [ ] 16.11 Batch training pair generation: per-scene random cameras (20+ angles), light variations (0.5x-2.0x), material perturbations, spp pairs (4/8/16/256)
- [ ] 16.12 Full pre-training: denoiser(5000) -> confidence(2000) -> multires(2000) -> temporal(5000) on production data
- [ ] 16.13 Validate on held-out production scenes (NOT Cornell box): SSIM > 0.92 all categories
- [ ] 16.14 Background fine-tuning: trigger on 3+ renders of same scene, LoRA rank=8, 50 iterations
- [ ] 16.15 Export base model: Nabla state_dict, SHA256, metadata JSON, bundle for distribution (~30MB)
- [ ] 16.16 Validate with degraded AOVs: train on scenes with full AOVs, test with partial AOVs missing to ensure graceful degradation

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
