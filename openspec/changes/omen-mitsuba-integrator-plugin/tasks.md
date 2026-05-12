## 1. Project Structure

- [x] 1.1 Create `src/omen_integrator/` directory
- [x] 1.2 Create `src/omen_integrator/__init__.py` (plugin registration)
- [x] 1.3 Create `src/omen_integrator/core.py` (path tracing logic)
- [x] 1.4 Create placeholder `src/omen_integrator/jepa.py` (future)
- [x] 1.5 Create placeholder `src/omen_integrator/gpu.py` (future)

## 2. Core Integrator Implementation

- [x] 2.1 Implement OmenIntegrator class with __init__ method
- [x] 2.2 Add max_depth, rr_depth, jepa_model, use_gpu parameters
- [x] 2.3 Implement render() method signature matching Integrator base
- [x] 2.4 Implement render_path_tracer() function in core.py
- [x] 2.5 Implement _trace_path() helper for single path sampling
- [x] 2.6 Add Russian roulette termination logic
- [x] 2.7 Implement direct illumination sampling (next event estimation)
- [x] 2.8 Add multiple importance sampling (BSDF + emitter)

## 3. Plugin Registration

- [x] 3.1 Implement register() function in __init__.py
- [x] 3.2 Call mi.register_integrator("omen", lambda props: OmenIntegrator(props))
- [x] 3.3 Test registration from Python: `import mitsuba as mi; mi.register_integrator("omen", ...)`

## 4. Mitsuba-Blender Integration

- [x] 4.1 Add "omen" entry to `mitsuba-blender/mitsuba-blender/engine/integrators.json`
- [x] 4.2 Configure max_depth parameter in JSON
- [x] 4.3 Configure rr_depth parameter in JSON
- [x] 4.4 Configure jepa_model parameter in JSON (string)
- [x] 4.5 Configure use_gpu parameter in JSON (boolean)

## 5. CLAUDE_POLICY.md Compliance

- [x] 5.1 Verify all imports are absolute (no `from .` or `from ..`)
- [x] 5.2 Verify each file under 150 lines of executable code
- [x] 5.3 Run `ruff check --fix` and ensure zero errors
- [x] 5.4 Run `ruff check format` and ensure zero changes
- [x] 5.5 Verify SOLID principles (single responsibility per module)

## 6. Testing

- [x] 6.1 Create simple test scene (Cornell box or sphere)
- [x] 6.2 Test rendering from Python API
- [ ] 6.3 Test rendering from Blender via Mitsuba-Blender addon
- [x] 6.4 Verify output differs from test gradient (real rendering occurs)
- [x] 6.5 Verify parameter passing (max_depth, rr_depth work correctly)

## 7. Documentation

- [x] 7.1 Add docstrings to all classes and functions
- [x] 7.2 Update README.md with Omen integrator usage
- [x] 7.3 Document JEPA/Mojo placeholder status
- [x] 7.4 Add example scene rendering command

## 8. JEPA Integration - Phase 1: Scene Extraction

- [ ] 8.1 Create `src/omen_integrator/scene_extractor.py` - Extract Mitsuba scene graph
- [ ] 8.2 Implement geometry extraction (vertices, faces, material indices)
- [ ] 8.3 Implement material/BSDF parameter extraction
- [ ] 8.4 Implement light/emitter parameter extraction
- [ ] 8.5 Implement camera/sensor parameter extraction
- [ ] 8.6 Create SceneGraph dataclass for structured scene data

## 9. JEPA Integration - Phase 2: C ABI Bridge

- [ ] 9.1 Create `jepa_kernels/` directory for Mojo code
- [ ] 9.2 Create `jepa_kernels/C_ABI.mojo` with C interface definitions
- [ ] 9.3 Create `jepa_kernels/omen_bridge.h` C header file
- [ ] 9.4 Create `src/omen_integrator/jepa_bridge.py` - ctypes bridge to Mojo
- [ ] 9.5 Test: Load dummy .so and call simple function from Python

## 10. JEPA Integration - Phase 3: Mode 1 (Denoiser)

- [ ] 10.1 Create `src/omen_integrator/modes/denoiser.py`
- [ ] 10.2 Implement multi-pass render: low spp → JEPA denoise
- [ ] 10.3 Create Mojo image encoder (strided convolutions)
- [ ] 10.4 Create Mojo scene encoder (transformer)
- [ ] 10.5 Create Mojo JEPA cross-attention module
- [ ] 10.6 Create Mojo decoder (latent → pixels)
- [ ] 10.7 Implement training loop: (4spp + scene) → 256spp
- [ ] 10.8 Test: Denoise Cornell box at 4 spp

## 11. JEPA Integration - Phase 4: Mode 2 (Adaptive)

- [ ] 11.1 Create `src/omen_integrator/modes/adaptive.py`
- [ ] 11.2 Create Mojo confidence head (second output)
- [ ] 11.3 Implement multi-pass render with confidence guidance
- [ ] 11.4 Implement adaptive merge (high-conf vs low-conf pixels)
- [ ] 11.5 Training: variance across renders → confidence labels
- [ ] 11.6 Test: Adaptive sampling on complex scene

## 12. JEPA Integration - Phase 5: Mode 3 (Multi-Res)

- [ ] 12.1 Create `src/omen_integrator/modes/multires.py`
- [ ] 12.2 Create Mojo multi-resolution merge kernel
- [ ] 12.3 Implement resolution change orchestration
- [ ] 12.4 Training: low-res + noisy → high-res clean
- [ ] 12.5 Test: 25% res 512spp + 100% res 4spp → merge
