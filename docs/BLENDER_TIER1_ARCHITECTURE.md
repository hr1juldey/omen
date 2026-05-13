# Omen Blender Tier 1 Integration Architecture

## TL;DR

Mojo/Nabla runs **in-process** inside Blender's Python. No PyTorch. No ONNX. No subprocess.
`mojo build --emit shared-lib` → ctypes/cffi load → same process, same GPU, zero-copy.
Users install from ZIP. Zero terminal. Zero pip. Zero code.

---

## Proven Viability (2026-05-13, verified in isolated uv venvs)

### Mojo .so compilation

| Test | Result |
|------|--------|
| `mojo build --emit shared-lib` | Produces 16KB .so (EXIT=0) |
| Load via ctypes with `LD_LIBRARY_PATH` | SUCCESS |

### Full stack (modular + mojo + nabla + ctypes) — tested one Python version at a time in clean uv venvs

| Python | modular | mojo.importer | nabla | ctypes .so | Status |
|--------|---------|---------------|-------|------------|--------|
| 3.11.13 | 26.4.0.dev2026051206 | OK | OK | OK | PASS |
| 3.12.0  | 26.4.0.dev2026051206 | OK | OK | OK | PASS |
| 3.13.5  | 26.4.0.dev2026051206 | OK | OK | OK | PASS |
| 3.14.0b4| 26.4.0.dev2026051206 | OK | OK | OK | PASS |

### System Python: UNTOUCHED (verified after all tests)

### Verified setup recipe (no pixi, no pip, uv only)

```bash
uv init --python 3.12  # or 3.11/3.13/3.14
uv add --pre modular --index https://whl.modular.com/nightly/simple/ --prerelease allow
uv add nabla-ml
```

### Key finding: nabla requires modular NIGHTLY

Nabla checks for `.dev` in `metadata.version("modular")`. Stable modular (26.2.0) fails.
Must use `--pre` flag with nightly index. This is why pixi was always used — it defaults to nightly.

## Runtime Dependencies (all inside `modular` nightly pip package)

```
libKGENCompilerRTShared.so  628K   Mojo compiler runtime
libMSupportGlobals.so        46K   Mojo support globals
libAsyncRTRuntimeGlobals.so 621K   Async runtime
libAsyncRTMojoBindings.so   1.1M   Async-Mojo bridge
libNVPTX.so                  56M   GPU kernel support (NVIDIA PTX)
libMojoLLDB.so                -    Mojo debugger
libMGPRT.so                   -    GPU runtime
libmax.so                     -    MAX engine
```

All .so files live in `site-packages/modular/lib/`. Set `LD_LIBRARY_PATH` at addon startup.

## Addon vs Engine Split (dev ergonomics)

The addon and engine are SEPARATED so the engine can be iterated without
reinstalling/restarting the Blender addon.

```
┌─────────────────────────────────────────────────┐
│  BLENDER ADDON (installed once, stays put)       │
│  src/omen_blender/                               │
│    ├─ __init__.py   ← bl_info, register, LD_PATH │
│    ├─ engine.py     ← OmenRenderEngine (bpy)     │
│    ├─ properties.py ← settings UI                │
│    ├─ panel.py      ← panels                     │
│    └─ bridge.py     ← talks to engine via import │
│                                                   │
│  The addon is THIN. It only:                      │
│  1. Registers as a Blender render engine          │
│  2. Syncs depsgraph → numpy arrays                │
│  3. Calls engine.render() and displays pixels     │
└─────────────────────────────────────────────────┘
         │ import omen_engine (reloadable)
         ↓
┌─────────────────────────────────────────────────┐
│  OMEN ENGINE (iterated freely, no addon reinstall)│
│  src/omen_engine/                                │
│    ├─ session.py    ← render pipeline orchestrator│
│    ├─ sync.py       ← depsgraph data extraction  │
│    ├─ display.py    ← viewport GPU display       │
│    ├─ backends/     ← pluggable path tracers     │
│    │   ├─ mitsuba_backend.py                      │
│    │   └─ ...                                     │
│    └─ kernels/      ← Mojo .so loading (ctypes)  │
│                                                   │
│  Dev workflow: edit engine code → F3 reload script│
│  No addon reinstall. No Blender restart.          │
└─────────────────────────────────────────────────┘
         │ ctypes / nabla
         ↓
┌─────────────────────────────────────────────────┐
│  MOJO/NABLA LAYER (compiled ahead of time)        │
│  src/omen/kernels/*.mojo → mojo build → .so       │
│  + modular nightly (runtime .so files)            │
│  + nabla-ml (@compiler.register GPU kernels)      │
│                                                   │
│  This layer is Mojo, not C/C++, by choice.        │
│  Mojo = GPU-speed + memory-safe + Python-friendly │
│  If bindings needed, use mojo-python-interop.      │
└─────────────────────────────────────────────────┘
```

### Why this split matters

- **Addon** (`src/omen_blender/`): Installed via ZIP into Blender. Rarely changes.
  Restarting Blender to reload = 30 seconds of pain. Minimize changes here.

- **Engine** (`src/omen_engine/`): The actual rendering logic. Changes constantly during dev.
  Reloaded via `importlib.reload()` or Blender's F3 "Reload Scripts". No restart needed.

- **Mojo kernels** (`src/omen/kernels/`): Compiled ahead of time with `mojo build`.
  Changes require recompilation but NOT addon reinstall. Just rebuild the .so and reload.

## Pluggable Path Tracer Backend

Omen is NOT tied to Mitsuba. The architecture separates the AI layer (Mojo/Nabla)
from the path tracer backend. Today Mitsuba, tomorrow Cycles or LuxCore.

```
Blender Plugin (Python - glue/orchestration)
         ↓
Omen Engine (Python - coordinates rendering pipeline)
         ↓
┌──────────────────────────────────┐
│  Path Tracer Backend (swappable) │
│  ├─ Backend A: Mitsuba (today)   │
│  ├─ Backend B: Cycles (future)   │
│  └─ Backend C: LuxCore (future)  │
└──────────────────────────────────┘
         ↓
Mojo/Nabla/Max (ALWAYS at the bottom - the AI layer)
  ├─ JEPA denoising (neural network inference)
  ├─ Tile fingerprinting (GPU kernel)
  ├─ MoE routing (GPU kernel)
  ├─ MLA compression (GPU kernel)
  ├─ SSIM quality scoring (GPU kernel)
  └─ Incremental scene learning (GPU kernel)
```

Mojo is the product. The path tracer is a replaceable input source.
This is why Mojo/Nabla/Max/Modular must run the inference — it IS Omen's core.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Blender Process                                                │
│                                                                 │
│  omen_blender addon (bpy)                                       │
│    │                                                            │
│    ├─ view_update(ctx, depsgraph)   ← scene changed             │
│    │    └─ OmenSync.sync(depsgraph)                             │
│    │        ├─ sync_mesh()    ← numpy from depsgraph verts      │
│    │        ├─ sync_camera()  ← transform matrices              │
│    │        ├─ sync_lights()  ← light params                    │
│    │        └─ sync_materials() ← BSDF node traversal           │
│    │                                                            │
│    ├─ view_draw(ctx, depsgraph)     ← draw viewport frame       │
│    │    └─ OmenSession.render(spp=4)                            │
│    │        ├─ backend.render() → noisy AOV buffers             │
│    │        ├─ numpy → Mojo .so (ctypes) → JEPA denoise         │
│    │        └─ GPU texture → Blender viewport via                │
│    │             engine.bind_display_space_shader + draw         │
│    │                                                            │
│    └─ render(depsgraph)             ← F12 final render          │
│         └─ OmenSession.render(spp=full)                         │
│              ├─ backend.render() → high-spp AOVs                │
│              ├─ Mojo JEPA denoise (MoE routing + MLA)           │
│              └─ begin_result/end_result → pixels                 │
│                                                                 │
│  ═══════ In-process Libraries ═══════                           │
│                                                                 │
│  backend (mitsuba today, cycles/luxcore tomorrow)               │
│  modular          (pip install modular) ← Mojo runtime .so's    │
│  nabla_ml         (pip install nabla-ml)                         │
│  omen_kernels.so  (mojo build --emit shared-lib)                │
│    ├─ tile_fingerprint_kernel                                    │
│    ├─ aov_pack_kernel                                            │
│    ├─ moe_dispatch_kernel                                        │
│    ├─ mla_compress_kernel                                        │
│    └─ ssim_kernel                                                │
│                                                                 │
│  LD_LIBRARY_PATH → modular/lib/ (set in addon __init__.py)      │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Blender depsgraph
  │
  ├─ Mesh vertices → numpy array (bpy mesh.vertices)
  │                   ↓
  │         backend.load_scene(numpy_arrays)
  │                   ↓
  │         backend.render(scene, sensor, spp=N)
  │                   ↓
  │   Noisy AOV buffers (numpy) ──→ Mojo .so kernels
  │       ↓                              │
  │   tile_fingerprint ──→ MoE routing   │
  │       ↓                              │
  │   JEPA latent decode                 │
  │       ↓                              │
  │   SSIM quality check                 │
  │       ↓                              │
  │   Clean pixel buffer                 │
  │       ↓                              │
  └─ Display to viewport OR write to RenderResult
```

## Module Layout

```
src/omen_blender/          ← BLENDER ADDON (thin wrapper, installed once)
  __init__.py               ← bl_info, register/unregister, LD_LIBRARY_PATH
  engine.py                 ← OmenRenderEngine(bpy.types.RenderEngine)
  bridge.py                 ← imports omen_engine, handles reload
  properties.py             ← OmenSettings PropertyGroup
  panel.py                  ← UI panels for render settings
  installer.py              ← Auto-installs bundled wheels on first enable

src/omen_engine/           ← ENGINE (iterated freely, no addon reinstall)
  __init__.py
  session.py                ← OmenSession: render pipeline orchestrator
  sync.py                   ← OmenSync: depsgraph → scene data extraction
  display.py                ← Viewport GPU display
  backends/                 ← Pluggable path tracer backends
    __init__.py              ← Backend ABC (render, load_scene, etc.)
    mitsuba_backend.py       ← Mitsuba integration (today)
    cycles_backend.py        ← Cycles integration (future)
    luxcore_backend.py       ← LuxCore integration (future)

src/omen_integrator/        ← Mitsuba-specific integrator (used by mitsuba_backend)
  __init__.py
  core.py
  path.py
  gpu.py
  jepa.py

src/omen/kernels/           ← Mojo kernels + Python bridges (THE CORE)
  *.mojo                     ← GPU kernels (tile_fingerprint, aov_pack, etc.)
  *.py                       ← Python bridges to load .so via ctypes
```

## Installation: ZIP Only (Users NEVER touch terminal)

### What the user does:
1. Download `omen_blender.zip` from website/store
2. Open Blender → Edit → Preferences → Add-ons → Install → select zip
3. Check the checkbox to enable
4. Select "Omen" from the render engine dropdown
5. Done.

### What happens inside the ZIP:

```
omen_blender.zip
  ├── omen_blender/              ← addon code
  │     ├── __init__.py
  │     ├── engine.py
  │     ├── sync.py
  │     ├── session.py
  │     ├── display.py
  │     ├── properties.py
  │     ├── panel.py
  │     ├── installer.py         ← auto-installs deps on first enable
  │     └── backends/
  ├── wheels/                    ← bundled pip wheels (auto-installed)
  │     ├── modular-*.whl
  │     ├── nabla_ml-*.whl
  │     ├── mitsuba-*.whl
  │     └── numpy-*.whl
  ├── lib/                       ← pre-compiled Mojo kernels
  │     ├── omen_kernels.so
  │     └── (Mojo runtime .so files)
  └── weights/                   ← pre-trained model weights
        └── omen_jepa_v1.npz
```

### installer.py logic (runs on first enable):

```python
"""Auto-installs bundled wheels into Blender's Python on first addon enable."""
import os, subprocess, sys

WHEELS_DIR = os.path.join(os.path.dirname(__file__), "..", "wheels")
MARKER = os.path.join(os.path.dirname(__file__), ".deps_installed")

def ensure_dependencies():
    if os.path.exists(MARKER):
        return  # Already installed

    blender_python = sys.executable
    wheels = [os.path.join(WHEELS_DIR, w) for w in os.listdir(WHEELS_DIR) if w.endswith(".whl")]
    subprocess.check_call([blender_python, "-m", "pip", "install", "--quiet"] + wheels)
    open(MARKER, "w").close()  # Mark as done
```

## Commercial Tiers

### Free / Base: Pre-trained Weights + ZIP Install
- Bundled weights trained on common scene types
- Works out of the box for general rendering
- No learning on user's machine
- Download ZIP, install, render

### Pro: Incremental Scene Learning
- Starts with base weights
- Watches user's scene during viewport rendering
- Learns scene-specific patterns (lighting, materials, geometry)
- Gets better the more the user works on their scene
- Temporal coherence improves over animation frames
- This is the NOVELTY — the AI adapts to YOUR scene

## Python vs Mojo: Why Both

```
Python = the coordinator (high level, easy to write, safe)
  → Tells Blender what's happening
  → Extracts mesh data from Blender's scene graph
  → Wires the pipeline together
  → Handles UI, settings, file I/O

Mojo = the engine (low level, GPU fast, memory safe)
  → Runs JEPA neural network inference
  → Executes tile fingerprinting (one GPU kernel)
  → Does MoE routing (another GPU kernel)
  → Does MLA compression (another GPU kernel)
  → 1000x+ faster than Python, safer than C++
  → This is why we chose Mojo over C++ for the compute layer
```

Same split as Cycles: Python plugin talks to Blender, compiled .so does the heavy lifting.
Except our .so is Mojo, not C++.

## Implementation Phases

### Phase 1: Skeleton + Final Render (F12)
- `OmenRenderEngine` with `render()` callback
- `OmenSync` extracts meshes, camera, lights from depsgraph
- Builds mitsuba scene via `mitsuba_backend.py`, renders
- Denoises via Mojo kernels loaded through ctypes
- Writes to `begin_result`/`end_result`

### Phase 2: Viewport Rendering
- `view_update()` + `view_draw()` callbacks
- Incremental sync (only changed objects)
- Progressive refinement (low spp → high spp)
- GPU display via `bind_display_space_shader`

### Phase 3: Animation + Timeline
- Frame change detection via depsgraph
- Temporal coherence (reuse previous frame's latent)
- Delta/surprise/jump-cut detection from `temporal.py`
- Motion vectors from `motion.py`

### Phase 4: Full Feature Parity
- Geometry nodes (auto via evaluated depsgraph)
- Hair/curves/volumes
- AOV output passes
- Material node graph conversion
- Adaptive/multires render modes

### Phase 5: Packaging & Distribution
- Build script to create distributable ZIP
- Bundle wheels, .so files, and weights
- Auto-installer for zero-config setup
- Test on clean Blender install

## Why This Works

1. **Mojo .so is just a shared library.** Python's ctypes can load it from any Python.
2. **`pip install mojo` works with standard pip.** No pixi needed at runtime.
3. **Runtime deps are finite and known.** 5 .so files, all inside `modular` pip package.
4. **`LD_LIBRARY_PATH` solves loading.** One env var, set once at addon startup.
5. **Same process = same GPU.** No IPC overhead. Zero-copy numpy between bpy → backend → mojo.
6. **Backend is pluggable.** Mitsuba today, Cycles tomorrow. Mojo layer stays the same.
7. **Users install from ZIP.** No terminal, no pip, no code. Standard Blender workflow.

This is architecturally identical to how Cycles works (Python addon → compiled .so → zero-copy),
except the .so is Mojo instead of C++. And users never see the difference.
