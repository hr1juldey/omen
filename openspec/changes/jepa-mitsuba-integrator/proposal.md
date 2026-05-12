## Why

Omen is a **render engine turbocharger** — it sits above ANY path tracer and makes it faster, smarter, and scene-aware. Today it works with Mitsuba 3. Tomorrow: Cycles, EEVEE, any engine.

**Omen fights on three fronts:**

1. **vs DLSS 4.0** (upscale + frame generation): DLSS is pixel-level, NVIDIA-locked, scene-blind. Omen understands 3D scene data and predicts frames from scene knowledge, not pixel interpolation.

2. **vs OIDN / OptiX** (denoising): These are scene-blind denoisers. OIDN balanced model is 460K params — it sees pixels + albedo + normals and guesses. Omen sees exact materials, lights, geometry and KNOWS "this is glass, expect caustics here."

3. **vs Diffusion models** (generation): Text-controlled, stochastic, non-reproducible. Artists type "glass sphere with caustics" and pray. Omen takes EXACT 3D parameters — IOR 1.52, roughness 0.02, light at (3,5,2) 5000K — deterministic output. Artist has full control. No prompt lottery.

**Product vision:** Give artists tools that respect their craft. Deterministic, scene-aware, 3D-controlled. Not prompt-spray-and-pray.

**Why JEPA specifically:** JEPA is the ONLY architecture that unifies all three capabilities — denoising, upscaling/frame generation, AND deterministic 3D-controlled rendering — into one model. A pure U-Net can only denoise. Diffusion can only generate stochastically. JEPA does everything because it UNDERSTANDS the scene via learned latent representations.

**Why now:** Mitsuba integrator skeleton exists (previous change). We have:
- Mitsuba 3.8.0 with Python API (easy to work with, no fighting with Cycles yet)
- Scene access via `mi.Scene` (shapes, BSDFs, emitters, sensors)
- Test environment: Cornell box renders in 2 seconds at 256×256
- Nabla ML for neural operations (Python API, MAX/Mojo backend)
- Mojo compiler for custom GPU kernels

Self-training makes JEPA practical: render same scene at 4spp AND 256spp → perfect supervised pairs. Unlimited training data for YOUR specific scene.

**Render engine portability:** Omen only needs THREE things from any render engine:
1. Noisy render (pixels — every engine produces this)
2. Auxiliary buffers (albedo, normal, depth — standard AOVs)
3. Scene graph (materials, lights, geometry — extractable from any scene format)

Standardize this interface and Omen works above Mitsuba today, Cycles tomorrow, EEVEE later.

## What Changes

- **JEPA scene understanding system** (1024-dim latent, scene graph encoder, ARPredictor for temporal)
- **U-Net denoiser** conditioned by JEPA scene latent (not a latent-to-image generator)
- **Scene graph extractor** from Mitsuba Python API → structured tensors
- **Three model tiers** for different scene complexity:
  - Fast (4M): test the waters, beats OIDN
  - Medium (16M): kills OptiX with scene awareness
  - High (64M): palace of mirrors, prisms, fog, 20K lights at 4K 60fps
- **Four rendering modes** implemented as multi-pass strategies:
  - Mode 1: Denoiser (post-process 4-16 spp → clean)
  - Mode 2: Adaptive (confidence-guided sampling, 4-8x sample reduction)
  - Mode 3: Multi-res (25% res 512spp + 100% res 4spp → clean 4K, DLSS competitor)
  - Mode 4: Temporal (ARPredictor frame generation from scene history, DLSS frame gen competitor)
- **Self-training protocol** using Cornell box as test scene
- **Energy conservation loss** (physics-based, no energy gain during denoising)

## Capabilities

### New Capabilities

- **scene-graph-encoding**: Extract render engine scene graph (geometry, materials, lights, camera) into standardized format for JEPA conditioning. Mitsuba today: vertices/faces from `mi.Shape`, BSDF params from `mi.BSDF`, emitter properties from `mi.Emitter`. Future: same interface for Cycles/EEVEE via Blender Python API.

- **jepa-scene-understanding**: JEPA system that encodes scene semantics into 1024-dim latent. Scene encoder (geometry + materials + lights) + SceneDeltaEncoder (frame-to-frame changes) + ARPredictor (temporal prediction). This is the "brain" — it understands "this is glass, expect caustics" and "this is skin, preserve subsurface scattering."

- **unet-denoiser**: U-Net denoiser conditioned by JEPA scene latent via AdaLN modulation. NOT a latent-to-image generator — takes noisy 4K input directly and restores it with scene-aware guidance. Previous clean frame feeds into U-Net for temporal coherence. Three tiers: Fast (4M), Medium (16M), High (64M).

- **adaptive-guidance**: Multi-pass rendering where JEPA classifies pixels by difficulty. PASS 1: quick preview (4 spp) → confidence prediction. PASS 2: targeted high-spp render (128 spp) → merge based on confidence. Sample allocation: high-conf pixels use JEPA prediction, low-conf pixels use path-traced.

- **multires-merge**: Multi-resolution rendering with scene-guided upsampling (DLSS competitor). PASS 1: low-res high-quality (25%, 256 spp). PASS 2: high-res noisy (100%, 4 spp). JEPA merges using exact geometry edges (from scene graph) and material boundaries.

- **temporal-coherence**: JEPA ARPredictor world model for frame generation (DLSS frame gen competitor). Autoregressive prediction of next-frame latents from history window + scene deltas. Scene delta encoder handles camera moves, object animation. At low movement: near-free quality boost from optical-flow-like scene understanding. At jump cuts: graceful fallback to single-frame denoising. Surprise detection triggers re-render only when prediction is unreliable.

- **checkpoint-storage**: Model checkpointing and continuous learning system. Three model tiers (4M/16M/64M) with scene-specific LoRA fine-tuning. Base model pre-trained on production scenes, per-scene adapters cached by topology hash.

- **energy-conservation**: Physics-based loss term preventing energy gain during denoising. `L_energy = mean(relu(E_out - E_in - epsilon))`. The denoiser can redistribute light but cannot create photons.

### Modified Capabilities

None (new functionality, building on existing `omen-integrator` spec)

## Impact

- **Render engine abstraction**: Scene graph extraction standardized behind a common interface. Mitsuba implementation first, Cycles/EEVEE via Blender Python API later. Omen only needs: noisy pixels + AOVs + scene graph.
- **Nabla ML (Python)**: All neural code in Python using Nabla (`import nabla as nb`). Mojo/MAX backend for execution. Custom GPU kernels via `call_custom_kernel()`. No C ABI bridge needed.
- **File structure**:
  - `src/omen/model/`: JEPA scene encoder, ARPredictor, U-Net denoiser, confidence head (Nabla Python)
  - `src/omen/kernels/`: Custom Mojo GPU kernels (SIGReg loss, merge operations)
  - `src/omen/scene/`: Scene graph extractors (Mitsuba first, Cycles/Blender later)
  - `src/omen/modes/`: Denoiser, adaptive, multires, temporal modes
  - `src/omen/training/`: Training loop, data generation
- **Three model tiers**: Fast (4M params, ~16MB), Medium (16M params, ~64MB), High (64M params, ~256MB). All fit in 6GB VRAM budget at 4K inference except High tier (needs 8GB+).
- **Testing**: Cornell box (`mi.cornell_box()`) as validation scene. Benchmark against OIDN (460K params) and OptiX (~57M params).
- **Dependencies**: Mitsuba 3.8.0, Dr.Jit (included), Nabla ML, Mojo compiler (for custom kernels only)
- **No breaking changes**: Existing `omen` integrator Mode 0 (standard path tracing) unchanged
