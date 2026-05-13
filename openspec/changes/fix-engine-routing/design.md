## Context

`omen_engine/` was created as the Blender-to-rendering glue layer. It currently duplicates logic from `src/omen/`:
- `mitsuba_backend.py` has its own AOV extraction and render loop (already in `src/omen/modes/denoiser.py`)
- `mojo_runtime.py` has its own ctypes loader (already in `src/omen/kernels/*.py` bridges)
- `session.py` bypasses the entire JEPA AI pipeline, returning raw noisy pixels to Blender

The existing `src/omen/` has the full working pipeline:
- `modes/denoiser.py::render_denoiser()` — orchestrates: mi.render + AOV → scene_extractor → Mojo kernels → JEPA → clean RGBA
- `jepa_bridge.py::JEPABridge` — loads Nabla model, runs encode/decode
- `scene_extractor.py::extract_scene_graph()` — builds {geometry, materials, lights} dict from mi.Scene
- `kernels/` — Mojo GPU bridges (aov_pack, moe_dispatch, mla_compress, ssim_kernel)

## Goals / Non-Goals

**Goals:**
- Route session.py through `denoiser.render_denoiser()` so the full AI pipeline runs
- Strip mitsuba_backend.py to only build mi.Scene from numpy arrays
- Add material extraction to sync.py so SceneGraphEncoder gets material data
- Delete mojo_runtime.py (duplicated by existing kernel bridges)

**Non-Goals:**
- Touching anything in `src/omen/` or `src/omen_integrator/`
- Adding training loop integration (separate change)
- Viewport rendering (still placeholder)
- Adding new backends (Cycles/LuxCore future work)

## Decisions

### Decision 1: session.py delegates to denoiser.render_denoiser()
**Choice**: Call `render_denoiser(mi_scene, bridge, spp, tier)` instead of implementing our own render loop.
**Why**: `render_denoiser()` already has the complete pipeline tested and working: AOV extraction, tile fingerprinting, MoE routing, scene graph extraction, JEPA inference. Duplicating it would be wrong and fragile.
**Alternative considered**: Importing individual steps from `src/omen/` and composing them in session.py — rejected because the composition already exists in `render_denoiser()`.

### Decision 2: mitsuba_backend.py becomes scene builder only
**Choice**: Only `build_scene(vertices, faces, camera, lights) → mi.Scene`. No render, no AOV, no integrator registration.
**Why**: The depsgraph gives us numpy arrays. Mitsuba needs an mi.Scene object. That conversion is the ONLY thing that must happen in the backend. Everything downstream (rendering, AOVs, AI) is handled by `src/omen/`.
**Alternative considered**: Keeping a thin Backend ABC — keeping it for future Cycles/LuxCore backends but removing all render logic.

### Decision 3: Delete mojo_runtime.py
**Choice**: Remove entirely.
**Why**: `src/omen/kernels/` already has Python bridges (aov_pack.py, moe_dispatch.py, mla_compress.py, ssim_kernel.py) that load Mojo .so files. `mojo_runtime.py` duplicates this and is never called by the corrected pipeline.

### Decision 4: Add material extraction to sync.py
**Choice**: Add `_sync_materials()` that extracts BSDF parameters (roughness, metallic, base_color, etc.) from depsgraph objects.
**Why**: `SceneGraphEncoder` expects `scene_graph["materials"]["params"]` — a (M, 5) array of material parameters. Without this, the JEPA model has no material understanding, which defeats the purpose of scene-aware denoising.

## Risks / Trade-offs

- **Risk**: `render_denoiser()` expects a fully constructed mi.Scene with proper materials → Mitigation: mitsuba_backend.build_scene() must handle Blender material extraction and conversion to Mitsuba BSDFs
- **Risk**: JEPABridge may not be initialized when session starts → Mitigation: session.py creates JEPABridge lazily on first render
- **Risk**: sync.py material extraction may miss custom shader nodes → Mitigation: fall back to principled BSDF defaults, log warning
