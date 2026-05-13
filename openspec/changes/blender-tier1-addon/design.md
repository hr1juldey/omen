## Context

Omen has a working JEPA denoising pipeline: Mojo GPU kernels in `src/omen/kernels/`, a Mitsuba integrator in `src/omen_integrator/`, and model code in `src/omen/model/`. Everything runs in pixi. The Blender addon at `src/omen_blender/` has wrong-architecture files (JSON exporter, subprocess client) that must be replaced.

Verified findings (see `docs/BLENDER_TIER1_ARCHITECTURE.md`):
- `mojo build --emit shared-lib` produces .so loadable via ctypes in any Python
- modular nightly + nabla-ml work in isolated uv venvs on Python 3.11/3.12/3.13/3.14
- Runtime needs: `LD_LIBRARY_PATH` to `modular/lib/`, 10 .so files bundled in pip package
- nabla-ml requires modular NIGHTLY (checks for `.dev` in version string)

## Goals / Non-Goals

**Goals:**
- Register "Omen" as a Blender render engine visible in the dropdown
- F12 render: depsgraph sync → Mitsuba render → Mojo denoise → pixel output
- Addon/engine split so engine code iterates without reinstalling the addon
- ZIP install workflow with auto-installer for dependencies
- Pluggable backend architecture (Mitsuba today, Cycles/LuxCore tomorrow)

**Non-Goals:**
- Viewport rendering (view_update/view_draw) — Phase 2, after F12 works
- Animation/timeline support — Phase 3
- Geometry nodes, hair, curves, volumes — Phase 4
- Training UI inside Blender — separate concern
- Replacing Mojo with PyTorch/ONNX — non-negotiable, Mojo IS the product

## Decisions

### 1. Addon/engine split via importlib.reload

**Decision**: Thin addon (`src/omen_blender/`) imports engine (`src/omen_engine/`). Engine code can be reloaded with Blender's "Reload Scripts" (F3) without reinstalling the addon.

**Why**: During development, the engine changes constantly. Restarting Blender for every change kills productivity. The addon wrapper (engine.py, properties.py, panel.py) rarely changes once set up.

**Alternative considered**: All code in the addon. Rejected because every engine change requires addon reinstall + Blender restart.

### 2. Mojo .so via ctypes, not PythonModuleBuilder

**Decision**: Use `mojo build --emit shared-lib` to compile kernels, load via `ctypes.CDLL()` at runtime.

**Why**: Proven to work across all Python versions. No dependency on mojo.importer at runtime (which needs the full compiler). The .so is self-contained with PTX GPU code embedded.

**Alternative considered**: `mojo.importer` for JIT compilation at runtime. Rejected because it needs the full mojo compiler in the user's environment. Pre-compiled .so is simpler for distribution.

### 3. Modular nightly via uv, not pixi

**Decision**: Install modular nightly + nabla-ml via uv into Blender's Python site-packages.

**Why**: nabla-ml requires modular nightly (`.dev` version check). uv handles this cleanly. Verified across 4 Python versions. No pixi runtime needed.

**Alternative considered**: Bundle pixi with the addon. Rejected because pixi is a heavy dependency manager, not a runtime. uv/pip is the standard Python packaging workflow.

### 4. Mitsuba as pip-installed backend, not bundled

**Decision**: mitsuba is installed via pip as part of the auto-installer, not bundled in the ZIP.

**Why**: mitsuba wheels are 60MB+ and platform-specific. pip handles platform resolution. The auto-installer in the addon runs on first enable.

### 5. Backend ABC for path tracer abstraction

**Decision**: `Backend` abstract class with `load_scene()`, `render()`, `get_aov_buffers()` methods. Mitsuba implements it today. Cycles/LuxCore implement it tomorrow.

**Why**: Decouples the Mojo AI layer from the specific path tracer. The engine doesn't care where noisy pixels come from — it just denoises them.

## Risks / Trade-offs

- **[Risk] Modular nightly is a moving target** → Pin exact version in auto-installer. Test against specific nightly before release.
- **[Risk] 56MB libNVPTX.so in modular package** → Acceptable for desktop addon. Future: strip debug symbols for distribution.
- **[Risk] Blender's Python may not match tested versions** → Auto-installer checks Python version and refuses to install on unsupported versions with a clear error message.
- **[Risk] ctypes .so has no type safety** → Python bridge layer in `src/omen/kernels/*.py` provides typed wrappers around ctypes calls.
- **[Trade-off] No JIT compilation at runtime** → Users get pre-compiled .so files. Custom kernel development still happens in pixi dev environment.

## Open Questions

- Should the auto-installer prompt the user or install silently? (Leaning: silent with progress bar)
- Should we support AMD GPUs (libNVPTX is NVIDIA-only)? Modular includes AMD support but untested.
- Exact Blender versions to target: 4.0+ (Python 3.11), 4.2+ (Python 3.11)?
