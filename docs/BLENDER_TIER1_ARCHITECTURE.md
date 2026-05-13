# Omen Blender Tier 1 Integration Architecture

## TL;DR

Mojo/Nabla runs **in-process** inside Blender's Python. No PyTorch. No ONNX. No subprocess.
`mojo build --emit shared-lib` → ctypes/cffi load → same process, same GPU, zero-copy.

---

## Proven Viability (2026-05-13)

| Test | Result |
|------|--------|
| `mojo build --emit shared-lib` | Produces 16KB .so (EXIT=0) |
| Load in Python 3.13 (system) via ctypes | SUCCESS |
| Load in Python 3.14 (pixi) via ctypes | SUCCESS |
| Load with `LD_LIBRARY_PATH` set to `modular/lib/` | SUCCESS |
| Runtime deps | 5 .so files (~58MB) bundled in `modular` pip package |
| `nabla-ml` requires | `modular` + `numpy` (that's it) |

## Runtime Dependencies (all inside `modular` pip package)

```
libKGENCompilerRTShared.so  628K   Mojo compiler runtime
libMSupportGlobals.so        46K   Mojo support globals
libAsyncRTRuntimeGlobals.so 621K   Async runtime
libAsyncRTMojoBindings.so   1.1M   Async-Mojo bridge
libNVPTX.so                  56M   GPU kernel support (NVIDIA PTX)
```

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
│    │        ├─ mi.render() → noisy AOV buffers                  │
│    │        ├─ numpy → Mojo .so (ctypes) → JEPA denoise         │
│    │        └─ GPU texture → Blender viewport via                │
│    │             engine.bind_display_space_shader + draw         │
│    │                                                            │
│    └─ render(depsgraph)             ← F12 final render          │
│         └─ OmenSession.render(spp=full)                         │
│              ├─ mi.render() → high-spp AOVs                     │
│              ├─ Mojo JEPA denoise (MoE routing + MLA)           │
│              └─ begin_result/end_result → pixels                 │
│                                                                 │
│  ═══════ In-process Libraries ═══════                           │
│                                                                 │
│  mitsuba          (pip install mitsuba)                          │
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
  │              mi.load_dict({"type": "ply", ...})
  │              OR mi.Mesh() from numpy arrays
  │                   ↓
  │              mi.Scene(objects, lights, sensor)
  │                   ↓
  ├─ mi.render(scene, sensor, spp=N)
  │       ↓
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
src/omen_blender/
  __init__.py           ← bl_info, register/unregister, set LD_LIBRARY_PATH
  engine.py             ← OmenRenderEngine(bpy.types.RenderEngine)
  sync.py               ← OmenSync: depsgraph → mitsuba scene
  session.py            ← OmenSession: render pipeline orchestrator
  display.py            ← Viewport GPU display (bind_display_space_shader)
  properties.py         ← OmenSettings PropertyGroup
  panel.py              ← UI panels for render settings

src/omen_integrator/     ← KEEP AS-IS (Mitsuba integrator)
  __init__.py
  core.py
  path.py
  gpu.py
  jepa.py

src/omen/kernels/        ← KEEP AS-IS (Mojo kernels + Python bridges)
  *.mojo
  *.py
```

## Installation (End User)

```bash
# In Blender's Python console or terminal:
/blender/4.x/python/bin/python -m pip install modular nabla-ml mitsuba

# Install addon:
# Edit → Preferences → Add-ons → Install → omen_blender.zip
```

The addon's `__init__.py` handles `LD_LIBRARY_PATH` setup:
```python
import os
from modular import lib as _modular_lib
os.environ.setdefault("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = (
    os.path.dirname(_modular_lib.__file__) + ":" +
    os.environ["LD_LIBRARY_PATH"]
)
```

## Implementation Phases

### Phase 1: Skeleton + Final Render (F12)
- `OmenRenderEngine` with `render()` callback
- `OmenSync` extracts meshes, camera, lights from depsgraph
- Builds mitsuba scene, renders, denoises via Mojo kernels
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

## Why This Works

1. **Mojo .so is just a shared library.** Python's ctypes can load it. It doesn't need pixi.
2. **Runtime deps are finite and known.** 5 .so files, all inside the `modular` pip package.
3. **`LD_LIBRARY_PATH` solves loading.** One env var, set once at addon startup.
4. **Mitsuba is pip-installable.** `pip install mitsuba` works in Blender's Python.
5. **Same process = same GPU.** No IPC overhead. No serialization. Zero-copy numpy arrays flow between bpy → mitsuba → mojo.

This is architecturally identical to how Cycles works (Python addon → compiled .so → zero-copy), except the .so is Mojo instead of C++.
