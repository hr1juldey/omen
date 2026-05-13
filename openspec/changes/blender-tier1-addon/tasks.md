## 1. Clean up wrong-architecture files

- [x] 1.1 Delete `src/omen_blender/exporter.py` (wrong JSON export approach)
- [x] 1.2 Delete `src/omen_blender/client.py` (wrong subprocess approach)
- [x] 1.3 Keep `src/omen_blender/__init__.py` and `src/omen_blender/properties.py` (rewrite contents)

## 2. Create engine module structure

- [x] 2.1 Create `src/omen_engine/__init__.py` with public API exports
- [x] 2.2 Create `src/omen_engine/backends/__init__.py` with `Backend` ABC (load_scene, render, get_aov_buffers)
- [x] 2.3 Create `src/omen_engine/backends/mitsuba_backend.py` implementing Backend ABC
- [x] 2.4 Create `src/omen_engine/sync.py` with OmenSync class (mesh, camera, light extraction from depsgraph)
- [x] 2.5 Create `src/omen_engine/session.py` with OmenSession class (pipeline orchestrator)
- [x] 2.6 Create `src/omen_engine/display.py` (placeholder for future viewport)

## 3. Rewrite addon wrapper

- [x] 3.1 Rewrite `src/omen_blender/__init__.py` with bl_info, register/unregister, LD_LIBRARY_PATH setup
- [x] 3.2 Rewrite `src/omen_blender/properties.py` with OmenSettings PropertyGroup (SPP, mode, tile_size)
- [x] 3.3 Create `src/omen_blender/engine.py` with OmenRenderEngine (render callback, delegates to omen_engine)
- [x] 3.4 Create `src/omen_blender/bridge.py` that imports omen_engine with reload support
- [x] 3.5 Create `src/omen_blender/panel.py` with Omen render settings panel

## 4. Mojo runtime setup

- [x] 4.1 Create `src/omen_engine/mojo_runtime.py` — LD_LIBRARY_PATH setup and modular nightly version check
- [x] 4.2 Add ctypes loader for omen_kernels.so with typed Python wrappers
- [x] 4.3 Add error handling for missing .so files and wrong modular version

## 5. Mitsuba backend implementation

- [x] 5.1 Implement `load_scene()` — numpy arrays to mi.Scene (mesh from vertices/faces, no file I/O)
- [x] 5.2 Implement `render()` — call mi.render() with omen integrator, return AOV buffers as numpy
- [x] 5.3 Implement `get_aov_buffers()` — extract color, albedo, normal, depth from mitsuba render result
- [x] 5.4 Register omen_integrator with mitsuba on backend init

## 6. Depsgraph sync implementation

- [x] 6.1 Implement `sync_mesh()` — extract vertex positions and face indices from evaluated depsgraph
- [x] 6.2 Implement `sync_camera()` — extract camera-to-world matrix, fov, resolution
- [x] 6.3 Implement `sync_lights()` — extract light type, position, color, intensity
- [x] 6.4 Handle geometry nodes (use evaluated mesh from depsgraph, not original)
- [x] 6.5 Handle empty scene gracefully (no crash on missing objects)

## 7. Render pipeline wiring

- [x] 7.1 Wire OmenRenderEngine.render() → OmenSync.sync() → MitsubaBackend.load_scene()
- [x] 7.2 Wire MitsubaBackend.render() → Mojo kernel denoise → clean pixel output
- [x] 7.3 Write clean pixels to Blender's RenderResult via begin_result/end_result
- [x] 7.4 Add basic progress reporting to Blender's render progress bar

## 8. Build and packaging

- [x] 8.1 Create `scripts/build_addon.py` — compiles Mojo .so and produces distributable ZIP
- [x] 8.2 Create `src/omen_blender/installer.py` — auto-installs modular nightly + nabla-ml + mitsuba on first enable
- [ ] 8.3 Test ZIP install workflow on a clean Blender installation
- [x] 8.4 Update `setup.sh` to reflect new uv-based development workflow

## 9. End-to-end verification

- [ ] 9.1 Test F12 render with a simple scene (cube + light + camera) in Blender
- [ ] 9.2 Test engine reload without addon reinstall (modify engine code, F3 reload, re-render)
- [ ] 9.3 Test addon disable/re-enable cycle (no crash, no duplicate registration)
- [ ] 9.4 Test with Flatpak Blender (verify Python version compatibility and site-packages access)
