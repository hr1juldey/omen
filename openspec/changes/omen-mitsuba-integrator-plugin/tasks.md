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
