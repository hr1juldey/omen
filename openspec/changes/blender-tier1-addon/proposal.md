## Why

Omen has a working JEPA denoising pipeline with Mojo GPU kernels, MoE routing, MLA compression, and Mitsuba path tracing — but it only runs inside pixi. To make this a real product, users need to open Blender, select "Omen" as their render engine, and get AI-accelerated renders without touching a terminal. The Mojo/Nabla runtime has been verified to work in isolated uv environments across Python 3.11-3.14, proving that in-process Tier 1 integration (like Cycles/Eevee) is possible without pixi.

## What Changes

- Replace the wrong-architecture `src/omen_blender/` files (JSON exporter, subprocess client) with a proper Blender RenderEngine addon
- Create `src/omen_engine/` as a separately-iterable engine layer (no addon reinstall needed for engine changes)
- Implement `OmenRenderEngine` with `render()` callback for F12 final renders
- Add `OmenSync` to extract meshes, camera, lights from Blender's depsgraph into numpy arrays
- Add `mitsuba_backend.py` as the first pluggable path tracer backend
- Wire Mojo GPU kernels (.so via ctypes) into the denoising pipeline
- Set up `LD_LIBRARY_PATH` to `modular/lib/` for Mojo runtime .so loading
- Add ZIP-packaging with auto-installer for zero-terminal user experience

## Capabilities

### New Capabilities
- `blender-engine`: Blender RenderEngine registration with render() callback, depsgraph sync, and pixel output via begin_result/end_result
- `engine-session`: Render pipeline orchestrator that coordinates backend rendering, Mojo kernel denoising, and pixel display
- `depsgraph-sync`: Extract mesh vertices, camera transforms, light parameters, and material nodes from Blender's evaluated depsgraph into numpy arrays
- `mojo-runtime`: Load pre-compiled Mojo .so kernels via ctypes with LD_LIBRARY_PATH setup for modular nightly runtime
- `backend-mitsuba`: Mitsuba 3 path tracer backend that consumes numpy scene data and produces AOV buffers
- `addon-packaging`: ZIP distribution with bundled wheels, auto-installer, and pre-compiled Mojo kernels

### Modified Capabilities

(none — this is a new module)

## Impact

- **New files**: `src/omen_blender/` (rewrite), `src/omen_engine/` (new)
- **Existing untouched**: `src/omen_integrator/`, `src/omen/kernels/`, `src/omen/model/`, `src/omen/modes/`, all tests
- **Dependencies**: modular nightly (pip), nabla-ml (pip), mitsuba (pip), numpy
- **Build step**: `mojo build --emit shared-lib` for kernel .so compilation
- **Architecture doc**: `docs/BLENDER_TIER1_ARCHITECTURE.md` (already updated with verified test results)
