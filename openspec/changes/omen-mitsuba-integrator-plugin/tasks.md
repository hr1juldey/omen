## 1. Project Structure

- [ ] 1.1 Create `src/omen_integrator/` directory
- [ ] 1.2 Create `src/omen_integrator/__init__.py` (plugin registration)
- [ ] 1.3 Create `src/omen_integrator/core.py` (path tracing logic)
- [ ] 1.4 Create placeholder `src/omen_integrator/jepa.py` (future)
- [ ] 1.5 Create placeholder `src/omen_integrator/gpu.py` (future)

## 2. Core Integrator Implementation

- [ ] 2.1 Implement OmenIntegrator class with __init__ method
- [ ] 2.2 Add max_depth, rr_depth, jepa_model, use_gpu parameters
- [ ] 2.3 Implement render() method signature matching Integrator base
- [ ] 2.4 Implement render_path_tracer() function in core.py
- [ ] 2.5 Implement _trace_path() helper for single path sampling
- [ ] 2.6 Add Russian roulette termination logic
- [ ] 2.7 Implement direct illumination sampling (next event estimation)
- [ ] 2.8 Add multiple importance sampling (BSDF + emitter)

## 3. Plugin Registration

- [ ] 3.1 Implement register() function in __init__.py
- [ ] 3.2 Call mi.register_integrator("omen", lambda props: OmenIntegrator(props))
- [ ] 3.3 Test registration from Python: `import mitsuba as mi; mi.register_integrator("omen", ...)`

## 4. Mitsuba-Blender Integration

- [ ] 4.1 Add "omen" entry to `mitsuba-blender/mitsuba-blender/engine/integrators.json`
- [ ] 4.2 Configure max_depth parameter in JSON
- [ ] 4.3 Configure rr_depth parameter in JSON
- [ ] 4.4 Configure jepa_model parameter in JSON (string)
- [ ] 4.5 Configure use_gpu parameter in JSON (boolean)

## 5. CLAUDE_POLICY.md Compliance

- [ ] 5.1 Verify all imports are absolute (no `from .` or `from ..`)
- [ ] 5.2 Verify each file under 100 lines of executable code
- [ ] 5.3 Run `ruff check --fix` and ensure zero errors
- [ ] 5.4 Run `ruff check format` and ensure zero changes
- [ ] 5.5 Verify SOLID principles (single responsibility per module)

## 6. Testing

- [ ] 6.1 Create simple test scene (Cornell box or sphere)
- [ ] 6.2 Test rendering from Python API
- [ ] 6.3 Test rendering from Blender via Mitsuba-Blender addon
- [ ] 6.4 Verify output differs from test gradient (real rendering occurs)
- [ ] 6.5 Verify parameter passing (max_depth, rr_depth work correctly)

## 7. Documentation

- [ ] 7.1 Add docstrings to all classes and functions
- [ ] 7.2 Update README.md with Omen integrator usage
- [ ] 7.3 Document JEPA/Mojo placeholder status
- [ ] 7.4 Add example scene rendering command
