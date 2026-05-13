## Why

`omen_engine/` duplicates logic that already exists in `src/omen/` and `src/omen_integrator/`. The session pipeline bypasses the entire JEPA AI layer — it renders noisy pixels via Mitsuba and returns them directly to Blender without denoising, scene graph extraction, MoE routing, or JEPA inference. This makes the addon useless: pressing F12 returns raw 4spp noise instead of a clean render.

## What Changes

- **DELETE** `src/omen_engine/mojo_runtime.py` — ctypes loading already exists in `src/omen/kernels/*.py` bridges (aov_pack.py, moe_dispatch.py, mla_compress.py, ssim_kernel.py)
- **REWRITE** `src/omen_engine/session.py` — route through `src/omen/modes/denoiser.py::render_denoiser()` which has the full pipeline: mi.render + AOV extraction → scene_extractor → Mojo GPU kernels (tile fingerprint, AOV pack, MoE dispatch) → JEPA encode/decode → clean RGBA
- **REWRITE** `src/omen_engine/backends/mitsuba_backend.py` — reduce to a thin `build_scene()` function that converts numpy arrays (from depsgraph sync) into an `mi.Scene` object. No render logic, no AOV logic, no integrator logic
- **EDIT** `src/omen_engine/sync.py` — add material parameter extraction from depsgraph (roughness, metallic, color, etc.) so the scene graph has material data for `SceneGraphEncoder`

## Capabilities

### New Capabilities

- `session-routing`: Session routes through existing omen AI pipeline (denoiser.render_denoiser) instead of bypassing it
- `scene-builder`: Thin Mitsuba scene builder that converts numpy arrays to mi.Scene (no render/AOV duplication)

### Modified Capabilities

- `depsgraph-sync`: Add material parameter extraction to sync.py so SceneGraphEncoder gets material data

## Impact

- `src/omen_engine/session.py` — full rewrite of render_scene method
- `src/omen_engine/backends/mitsuba_backend.py` — stripped down to scene builder only
- `src/omen_engine/sync.py` — add _sync_materials() method
- `src/omen_engine/mojo_runtime.py` — deleted
- `src/omen/` — untouched (all existing AI code stays as-is)
- `src/omen_integrator/` — untouched
- `src/omen_blender/` — untouched (addon wrapper, bridge, engine, properties all correct)
