## Why

Omen needs JEPA (Joint Embedding Predictive Architecture) for **scene-aware rendering acceleration**, not another 2D denoiser.

**The problem:** Current denoisers (OptiX, OIDN) only see 2D pixels. They cannot:
- Understand that a pixel is on a glass surface (needs caustics sampling)
- Know that a flat wall has uniform BSDF (can be predicted)
- Use exact light positions from the scene

**The opportunity:** Mitsuba gives us exact 3D scene data (geometry, materials, lights, camera). JEPA can learn **"given this exact scene, what should clean pixels look like?"** and predict which regions need more samples.

**Why now:** Mitsuba integrator skeleton exists (previous change). We have:
- Mitsuba 3.8.0 with Python API
- Scene access via `mi.Scene` (shapes, BSDFs, emitters, sensors)
- Test environment: Cornell box renders in 2 seconds at 256×256
- Mojo compiler for GPU kernels

Self-training makes JEPA practical: render same scene at 4spp AND 256spp → perfect supervised pairs. Unlimited training data for YOUR specific scene.

## What Changes

- **Add Mojo JEPA kernels** (`jepa_kernels/`) compiled to `.so` with C ABI interface
- **Scene graph extractor** from Mitsuba Python API → structured tensors
- **Ctypes bridge** (`omen_integrator/jepa_bridge.py`) to call Mojo from Python
- **Three rendering modes** implemented as multi-pass strategies:
  - Mode 1: Denoiser (post-process 4-16 spp → clean)
  - Mode 2: Adaptive (confidence-guided sampling, 4-8x sample reduction)
  - Mode 3: Multi-res (25% res 512spp + 100% res 4spp → clean 1080p)
- **Self-training protocol** using Cornell box as test scene

## Capabilities

### New Capabilities

- **scene-graph-encoding**: Extract Mitsuba scene graph (geometry, materials, lights, camera) into structured format for JEPA conditioning. Vertices, faces from `mi.Shape`; BSDF params from `mi.BSDF`; emitter properties from `mi.Emitter`; sensor transform from `mi.Sensor`.

- **jepa-inference**: Load compiled Mojo JEPA model via C ABI and run inference. Inputs: scene graph + noisy render (4 spp). Outputs: denoised RGB + confidence map (Mode 2). Zero-copy GPU buffer passing via `UnsafePointer` wrapping.

- **adaptive-guidance**: Multi-pass rendering where JEPA classifies pixels by difficulty. PASS 1: quick preview (4 spp) → confidence prediction. PASS 2: targeted high-spp render (128 spp) → merge based on confidence. Sample allocation: high-conf pixels use JEPA prediction, low-conf pixels use path-traced.

- **multires-merge**: Multi-resolution rendering with scene-guided upsampling. PASS 1: low-res high-quality (25%, 256 spp). PASS 2: high-res noisy (100%, 4 spp). JEPA merges using exact geometry edges (from scene graph) and material boundaries. Avoids DLSS-style artifacts.

- **checkpoint-storage**: Model checkpointing and continuous learning system. Saves JEPA weights during training (every 10 iterations), enables resume after crash. Scene-specific fine-tuning cached by hash (geometry + materials + lights). Base model pre-trained on Cornell box variants, automatically improves with usage via local model aggregation. Opt-in anonymous contribution uploads only weight deltas (de-identified). Similar scene detection enables model reuse without retraining.

- **temporal-coherence**: JEPA world model (based on LeWM architecture, LeCun et al. 2026) for animation acceleration. Autoregressive prediction of next-frame latents from history window + scene deltas replaces path tracing for most frames. Scene delta encoder handles camera moves, object animation, fluid/smoke introduction, new lights. Surprise detection triggers path tracing only when prediction is unreliable. Topology-based scene hashing for stable animation cache. Target: 10-50x speedup on animations, path-traced quality at EEVEE/UE5 speeds.

### Modified Capabilities

None (new functionality, building on existing `omen-integrator` spec)

## Impact

- **Mitsuba Python API**: Use `mi.Scene.shapes()`, `mi.Scene.emitters()`, `mi.Scene.sensors()` for extraction. No C++ modification.
- **Mojo toolchain**: Require Mojo compiler, Nabla ML for neural operations. Compile to `.so`/`.dll`/`.dylib`.
- **File structure**:
  - `jepa_kernels/`: Mojo source (scene_encoder.mojo, image_encoder.mojo, jepa.mojo, decoder.mojo)
  - `src/omen_integrator/`: Python modules (scene_extractor.py, jepa_bridge.py, modes/)
- **Testing**: Cornell box (`mi.cornell_box()`) as validation scene. 2s baseline at 256×256 → target: same quality in <500ms with 4-8x fewer samples.
- **Dependencies**: Mitsuba 3.8.0, Dr.Jit (included), Mojo, Nabla ML, ctypes (stdlib)
- **No breaking changes**: Existing `omen` integrator Mode 0 (standard path tracing) unchanged
