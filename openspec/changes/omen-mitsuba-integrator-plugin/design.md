## Context

Omen render engine needs JEPA-accelerated path tracing for Mitsuba 3. Research shows Mitsuba has **no**:
- Adaptive sampling (confirmed: github.com/mitsuba-renderer/mitsuba-tutorials/issues/21)
- Scene-aware denoising (OptiX is 2D-only)
- Neural radiance caching or temporal reuse
- Multi-resolution guidance

**Critical Finding**: Mitsuba's path tracer is **C++ code** (src/integrators/path.cpp).
- Python `mi.render()` is a binding to C++ - cannot inject into sampling loop
- We can only work **before** (parameter setup) and **after** (post-process) renders
- Multi-pass rendering is our only lever for JEPA integration

**Current state:**
- Mitsuba 3.8.0 installed via pixi
- Mitsuba-Blender addon at `/mitsuba-blender/` handles Blender integration
- Reference: `mitsuba3/src/integrators/path.cpp` (383 lines C++)

**Constraints:**
- CLAUDE_POLICY.md: absolute imports, 100-line file limit, SOLID, Ruff compliance
- Mitsuba Python API for integrator registration
- Mojo C ABI for JEPA model (separate .so)
- JEPA runs in Mojo **only**, not Python
- No Mitsuba C++ modification

## Goals / Non-Goals

**Goals:**
- Create Omen as Mitsuba Python integrator with JEPA acceleration
- Implement 3 JEPA modes: Denoiser, Adaptive, Multi-Res
- Extract Mitsuba scene graph for JEPA conditioning
- Bridge Python ↔ Mojo via C ABI

**Non-Goals:**
- Modifying Mitsuba C++ path tracer
- Per-pixel adaptive sampling within single render (C++ limitation)
- Real-time viewport rendering
- Material node compilation (use Mitsuba's BSDFs)

## Decisions

### Decision 1: Multi-pass rendering strategy
**Choice:** JEPA works via multiple `mi.render()` calls, not single-pass injection

**Rationale:**
- Mitsuba path tracer is C++ - cannot inject Python into sampling loop
- Can render multiple times with different settings (spp, resolution)
- JEPA fuses/merges results from multiple passes
- Matches confirmed limitation from source code analysis

**Alternatives considered:**
- Custom C++ integrator: More powerful but requires build toolchain
- Single-pass injection: Impossible with current architecture

### Decision 2: Mode 2 uses tile-based adaptive sampling
**Choice:** Confidence guidance at tile granularity (64x64), not per-pixel

**Rationale:**
- Cannot set per-pixel sample counts from Python
- Tile-based is viable: render regions at different spp
- Still achieves 4-8x sample reduction for "easy" regions
- Can be refined in future C++ integrator

**Alternatives considered:**
- Full-image adaptive: Render at 2 spp, then 128 spp, merge per-pixel by confidence
- Per-pixel adaptive: Requires C++ modification

### Decision 3: Scene extraction from Mitsuba Python API
**Choice:** Parse `mi.Scene` object for geometry, materials, lights, camera

**Rationale:**
- Mitsuba Python API exposes all scene data
- No need for custom file format reader
- Works with any Mitsuba scene (XML, Python, Blender-exported)

**Data extracted:**
```python
scene = mi.cornell_box()
# Geometry:
shapes = scene.shapes()  # Mesh attributes: vertex_positions, faces
# Materials:
for shape in shapes:
    bsdf = shape.bsdf()  # BSDF parameters: diffuse_reflectance, roughness, etc.
# Lights:
emitters = scene.emitters()  # Emitter attributes: position, intensity, type
# Camera:
sensor = scene.sensors()[0]  # Sensor attributes: to_world, near_clip, far_clip
```

### Decision 4: Mojo C ABI interface
**Choice:** Compile Mojo to .so, call via ctypes from Python

**Rationale:**
- JEPA runs in Mojo only (user requirement)
- C ABI is stable, version-independent
- No Python runtime dependency in Mojo code
- Can cross-compile for Linux/macOS/Windows

**Interface:**
```c
int omen_denoise(
    SceneGraph* scene,        // Structured scene data
    RenderObservation* obs,   // Noisy image + depth + normal + albedo
    float* output_rgba,       // Denoised output
    int gpu_device_id
);
```

### Decision 5: File structure for 100-line limit
**Choice:** Split into focused modules under 100 lines each

```
src/omen_integrator/
├── __init__.py          # Plugin registration, OmenIntegrator class (~70 lines)
├── core.py              # Multi-pass render orchestration (~100 lines)
├── scene_extractor.py   # Mitsuba scene → structured tensors (~100 lines)
├── jepa_bridge.py       # ctypes bridge to Mojo .so (~80 lines)
└── modes/
    ├── denoiser.py      # Mode 1: Post-process denoising (~60 lines)
    ├── adaptive.py      # Mode 2: Multi-pass adaptive (~80 lines)
    └── multires.py      # Mode 3: Multi-resolution merge (~60 lines)

jepa_kernels/
├── C_ABI.mojo           # C interface definitions
├── scene_encoder.mojo   # Transformer over scene tokens
├── image_encoder.mojo   # Strided convolutions
├── cross_attention.mojo # JEPA core world model
├── confidence.mojo      # Confidence head (Mode 2)
├── decoder.mojo         # Latent → pixels
├── multires.mojo        # Multi-resolution merge (Mode 3)
└── build.sh             # Compile to .so
```

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Blender                                 │
│  ┌──────────────┐          ┌──────────────────────────────┐   │
│  │  User        │          │  Mitsuba-Blender Addon       │   │
│  │  selects     │─────────▶│  - Exports scene to Mitsuba  │   │
│  │  "omen"      │          │  - Calls mi.render()         │   │
│  └──────────────┘          └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Mitsuba 3 (Python)                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Omen Integrator (omen_integrator/)                      │  │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │  │
│  │  │   __init__ │  │ scene_extra- │  │  jepa_bridge    │  │  │
│  │  │            │  │    ctor.py   │  │                 │  │  │
│  │  └────────────┘  └──────────────┘  └─────────────────┘  │  │
│  │  ┌────────────┐  ┌──────────────────────────────────┐  │  │
│  │  │   core.py  │─▶│  modes/                          │  │  │
│  │  │            │  │  ├── denoiser.py                 │  │  │
│  │  └────────────┘  │  ├── adaptive.py                 │  │  │
│  │                  │  └── multires.py                 │  │  │
│  │                  └──────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          │ ctypes (C ABI)                      │
│                          ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  JEPA Mojo Kernels (jepa_kernels/*.mojo → .so)          │  │
│  │  - scene_encoder (transformer)                           │  │
│  │  - image_encoder (convolutions)                          │  │
│  │  - cross_attention (JEPA world model)                    │  │
│  │  - decoder (latent → pixels)                             │  │
│  │  - confidence_head (Mode 2)                              │  │
│  │  - multires_merge (Mode 3)                               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Mode 1: Denoiser (Post-Process)

```
INPUT: Scene (Mitsuba), mode="denoise", spp=4

PASS 1: Render low spp
  result = mi.render(scene, spp=4)
  └─> noisy_rgba [H, W, 4]

EXTRACT: Scene graph
  scene_graph = scene_extractor.extract(scene)
  └─> {geometry, materials, lights, camera}

INFERENCE: JEPA denoise
  denoised = jepa_bridge.denoise(
      scene_graph,
      noisy_rgba,
      depth=optional,
      normal=optional,
      albedo=optional
  )
  └─> clean_rgba [H, W, 4]

OUTPUT: Denoised image
  return denoised
```

### Mode 2: Adaptive (Multi-Pass)

```
INPUT: Scene (Mitsuba), mode="adaptive", spp_target=128

PASS 1: Quick preview
  preview = mi.render(scene, spp=4)
  └─> noisy_rgba [H, W, 4]

EXTRACT + PREDICT: Confidence map
  scene_graph = scene_extractor.extract(scene)
  confidence_map = jepa_bridge.predict_confidence(
      scene_graph,
      preview
  )
  └─> confidence [H, W, 1] (0=uncertain, 1=confident)
       denoised [H, W, 4]

PASS 2: Targeted high-spp render
  # Since we can't do per-pixel adaptive, render full image at high spp
  high_spp = mi.render(scene, spp=128)

MERGE: Combine based on confidence
  final = jepa_bridge.merge_adaptive(
      jepa_denoised,  # From PASS 1 (high confidence pixels)
      high_spp,       # From PASS 2 (low confidence pixels)
      confidence_map
  )
  └─> final_rgba [H, W, 4]

OUTPUT: Adaptive render
  return final
```

**Sample savings:**
- High-confidence regions (background, flat surfaces): Use JEPA prediction
- Low-confidence regions (caustics, SSS, edges): Use path-traced pixels
- Expected: 4-8x fewer samples vs uniform sampling

### Mode 3: Multi-Resolution (Spatial)

```
INPUT: Scene (Mitsuba), mode="multires", scale=4

PASS 1: Low resolution, high quality
  scene.sensor.film().set_size([H//4, W//4])
  low_res = mi.render(scene, spp=256)
  └─> clean_low_res [H/4, W/4, 4]

PASS 2: High resolution, noisy
  scene.sensor.film().set_size([H, W])
  high_res = mi.render(scene, spp=4)
  └─> noisy_high_res [H, W, 4]

EXTRACT: Scene graph
  scene_graph = scene_extractor.extract(scene)

MERGE: Scene-guided upsampling
  final = jepa_bridge.merge_multires(
      scene_graph,
      clean_low_res,
      noisy_high_res,
      scale=4
  )
  └─> final_rgba [H, W, 4]

OUTPUT: Multi-resolution render
  return final
```

**Speedup:**
- Render 25% pixels at 256 spp + 100% pixels at 4 spp
- JEPA knows exact geometry → no DLSS-style artifacts
- Expected: 8-16x effective speedup vs uniform 256 spp

## Data Structures

### SceneGraph (Python → Mojo)

```python
@dataclass
class SceneGraph:
    geometry: List[Geometry]    # Meshes: vertices, faces, material_ids
    materials: List[Material]   # BSDF parameters
    lights: List[Light]         # Emitter parameters
    camera: Camera              # Sensor transform and properties
```

### C ABI Layout

```c
typedef struct {
    float* vertices;    // [N_verts, 3]
    int    n_verts;
    int*   faces;       // [N_faces, 3]
    int    n_faces;
    int*   material_ids; // [N_faces]
} Geometry;

typedef struct {
    float* diffuse_reflectance;  // [3]
    float* roughness;            // [1]
    float* metallic;             // [1]
    // ... more BSDF params
} Material;

typedef struct {
    float* position;    // [3]
    float* direction;   // [3]
    float* intensity;   // [3]
    int    type;        // 0=point, 1=area, 2=directional
} Light;

typedef struct {
    float* position;        // [3]
    float* direction;       // [3]
    float* up_vector;       // [3]
    float  fov;             // radians
    float  near_clip;
    float  far_clip;
} Camera;

typedef struct {
    Geometry*  geometries;
    int        n_geometries;
    Material*  materials;
    int        n_materials;
    Light*     lights;
    int        n_lights;
    Camera     camera;
} SceneGraph;

typedef struct {
    float* noisy_rgba;    // [H, W, 4]
    int    height;
    int    width;
    float* depth;         // [H, W, 1] or NULL
    float* normal;        // [H, W, 3] or NULL
    float* albedo;        // [H, W, 3] or NULL
} RenderObservation;
```

## Implementation Phases

### Phase 1: Foundation (this change)
- [ ] OmenIntegrator with mode parameter (0=std, 1=denoise, 2=adaptive, 3=multires)
- [ ] Scene graph extractor from Mitsuba
- [ ] Multi-pass render orchestration in core.py
- [ ] C ABI header and ctypes skeleton

### Phase 2: Mode 1 - Denoiser
- [ ] Mojo image encoder (Nabla convolutions)
- [ ] Mojo scene encoder (transformer)
- [ ] Mojo JEPA cross-attention
- [ ] Mojo decoder
- [ ] Training: (4spp + scene) → 256spp
- [ ] Python bridge integration

### Phase 3: Mode 2 - Adaptive
- [ ] Mojo confidence head
- [ ] Multi-pass rendering with confidence guidance
- [ ] Adaptive merge kernel
- [ ] Training: variance → confidence labels

### Phase 4: Mode 3 - Multi-Res
- [ ] Mojo multi-resolution merge
- [ ] Resolution change orchestration
- [ ] Training: low-res + noisy → high-res clean

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| C ABI version mismatch | Stable structs, version field, graceful fallback |
| Mojo compilation fails | Provide prebuilt .so for major platforms |
| Multi-pass slow (3x renders) | JEPA speedup >3x makes it worth it |
| Tile artifacts in Mode 2 | Overlap tiles, feather edges |
| Scene extraction incomplete | Support Mitsuba primitives first, extend later |

## Migration from Phase 1 (Current)

**Current state**: Placeholder JEPA parameters only
**Migration path**:
1. Add `mode` parameter to OmenIntegrator
2. Implement scene_extractor.py
3. Create jepa_bridge.py with ctypes skeleton
4. Create modes/ directory structure
5. Phase 2-4: Implement actual Mojo kernels

### Rollback strategy
- Remove `mode` parameter, keep `jepa_model` as placeholder
- Delete scene_extractor.py, jepa_bridge.py, modes/
- Continue with standard path tracing (Mode 0)

## Open Questions

1. **Q:** Which Mojo ML framework?
   **A:** Nabla (nabla-ml) - has autograd, GPU, SPMD

2. **Q:** How to train JEPA model?
   **A:** Self-training: render same scene at 4spp AND 256spp, supervised pairs

3. **Q:** Where to store trained models?
   **A:** User cache dir (~/.cache/omen/models/) or project-local

4. **Q:** Mode 2 tile size?
   **A:** Start with 64x64, autotune based on scene complexity
