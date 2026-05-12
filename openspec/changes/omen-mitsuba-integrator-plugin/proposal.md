## Why

Omen render engine needs JEPA-accelerated path tracing for Mitsuba 3. Current Mitsuba has **no**:
- Scene-aware denoising (OptiX is 2D-only, no 3D knowledge)
- Adaptive sampling (confirmed by GitHub issue #21)
- Neural radiance caching or temporal reuse
- Multi-resolution guidance

Omen provides **three modes** of JEPA acceleration:
1. **Denoiser**: Post-process with 3D scene graph knowledge
2. **Accelerator**: Confidence-guided adaptive sampling (4-8x fewer samples)
3. **Multi-Res**: Scene-guided upsampling (8-16x effective speedup)

**Critical constraint**: Mitsuba's path tracer is **C++ code** (not Python). We cannot inject into the sampling loop. We work **before** and **after** `mi.render()` calls using multi-pass strategies.

## What Changes

- Create Omen as a Mitsuba 3 Python plugin (custom integrator)
- Register the integrator with Mitsuba's plugin system
- Add Omen to Mitsuba-Blender's integrator list for Blender UI exposure
- **Phase 1 (this change)**: Path tracing foundation + scene graph extraction
- **Phase 2-4 (follow-up)**: JEPA modes implemented in Mojo, called via C ABI
- Follow CLAUDE_POLICY.md compliance (absolute imports, 100-line file limits, SOLID principles)

## Capabilities

### New Capabilities

- **omen-integrator**: Mitsuba 3 integrator with three rendering modes
  - Mode 0: Standard path tracing (baseline, uses Mitsuba's C++ path tracer)
  - Mode 1: JEPA denoiser (post-process, 4-16 spp → clean output)
  - Mode 2: Confidence-guided adaptive (tile-based, 4-8x sample reduction)
  - Mode 3: Multi-resolution merge (25% res 512spp + 100% res 4spp → final)

- **scene-extractor**: Extract Mitsuba scene graph → structured tensors for JEPA
  - Geometry: vertices, faces, material indices
  - Materials: BSDF parameters (diffuse, roughness, metallic, etc.)
  - Lights: position, type, intensity, color
  - Camera: transform, FOV, clip planes

- **jepa-bridge**: C ABI interface to Mojo JEPA kernels
  - `omen_denoise()`: Denoise noisy render using scene context
  - `omen_predict_confidence()`: Classify pixels by difficulty
  - `omen_merge_multires()`: Fuse multi-resolution renders

### Modified Capabilities

None (this is new functionality, not changing existing specs)

## Impact

- **Mitsuba-Blender addon**: Add `"omen"` entry to `integrators.json`
- **New directories**:
  - `src/omen_integrator/`: Python modules (core, scene_extractor, jepa_bridge, modes/)
  - `jepa_kernels/`: Mojo source for JEPA model and C ABI
- **Dependencies**: Mitsuba 3.8.0, Dr.Jit, Mojo compiler, Nabla ML
- **No Blender code changes**: Omen runs independently within Mitsuba's process
- **No breaking changes**: Existing integrators (path, direct, etc.) continue to work
- **Multi-pass rendering**: JEPA modes require 2-3 `mi.render()` calls per frame

## Success Criteria

- [ ] Omen integrator registered and visible in Mitsuba-Blender UI
- [ ] Standard path tracing produces correct renders (not test gradient)
- [ ] Scene extractor converts Mitsuba scene to structured tensors
- [ ] JEPA bridge loads Mojo .so and calls functions successfully
- [ ] Mode 1 denoises 4 spp Cornell box to match 256 spp quality
- [ ] Mode 2 reduces samples by 4-8x on test scenes (measured by total spp)
- [ ] Mode 3 merges 25%+100% renders to clean 1080p output
