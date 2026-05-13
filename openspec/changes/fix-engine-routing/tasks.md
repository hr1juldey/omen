## 1. Delete duplicated code

- [x] 1.1 Delete `src/omen_engine/mojo_runtime.py` (duplicated by `src/omen/kernels/*.py` bridges)
- [x] 1.2 Remove any imports of mojo_runtime from other omen_engine files

## 2. Add material extraction to sync.py

- [x] 2.1 Add `_sync_materials()` method to OmenSync — extract base_color, roughness, metallic, emission from Principled BSDF nodes
- [x] 2.2 Handle objects with no material (use default grey diffuse)
- [x] 2.3 Wire `_sync_materials()` into the main `sync()` method return dict

## 3. Rewrite mitsuba_backend.py as scene builder only

- [x] 3.1 Replace MitsubaBackend with `build_scene(vertices, faces, camera_matrix, camera_fov, width, height, lights, materials) -> mi.Scene`
- [x] 3.2 Convert material dicts to Mitsuba BSDFs (roughdiffuse / conductor / principled)
- [x] 3.3 Handle empty mesh with fallback quad scene
- [x] 3.4 Remove render(), get_aov_buffers(), _register_integrator() methods

## 4. Rewrite session.py to route through denoiser pipeline

- [x] 4.1 Import and delegate to `src/omen/modes/denoiser.render_denoiser()` instead of own render loop
- [x] 4.2 Add lazy JEPABridge initialization (create on first render, cache for reuse)
- [x] 4.3 Wire: sync → build_scene → render_denoiser → clean RGBA return
- [x] 4.4 Handle render failures gracefully (log error, return black frame)
- [x] 4.5 Remove old render_scene/render_tile methods that bypassed the AI layer

## 5. Update engine.py if needed

- [x] 5.1 Verify engine.py passes correct settings (mode, tier, spp) to session
- [x] 5.2 Verify _to_rgba handles the (H,W,4) output from render_denoiser correctly

## 6. Verify no broken imports

- [x] 6.1 Grep for any remaining mojo_runtime references and remove them
- [x] 6.2 Verify all omen_engine modules import cleanly (no circular deps)
- [x] 6.3 Rebuild omen_blender.zip and verify it contains the corrected files
