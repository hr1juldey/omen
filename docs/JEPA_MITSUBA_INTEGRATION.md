# JEPA Integration for Mitsuba 3 - Architecture & Implementation

## Critical Finding: Mitsuba 3's Architecture

**Mitsuba 3's path tracer is C++ code**, not Python. The Python `mi.render()` function is a binding that calls into C++. This means:

### What WE CANNOT Do (from Python):

- ❌ Inject into the C++ sampling loop
- ❌ Modify per-sample behavior during path tracing
- ❌ Access intermediate path states (ray, throughput, etc.) from Python
- ❌ Implement true per-pixel adaptive sampling within a single render pass

### What WE CAN Do (from Python):

- ✅ Pre-process: Set scene parameters before rendering
- ✅ Post-process: Read and modify the rendered image after `mi.render()`
- ✅ Multi-pass: Render multiple times with different settings
- ✅ Scene extraction: Read Mitsuba's scene graph (geometry, materials, lights)

---

## Revised Omen Modes for Mitsuba 3

### Mode 1: Denoiser (POST-PROCESS)

**Status**: ✅ **FULLY VIABLE** - No competition with Mitsuba

```python
# Flow:
1. Render at low spp (4-16)
   result = mi.render(scene, spp=4)
2. JEPA denoises using scene graph context
   denoised = jepa_model.denoise(result, scene_graph)
3. Output denoised image
```

**Value**: Same as Cycles plan - scene-aware denoising.

---

### Mode 2: Confidence-Guided Multi-Pass (ADAPTIVE)

**Status**: ✅ **VIABLE WITH MODIFICATIONS** - Different from Cycles approach

**Original Cycles Plan** (not possible in Mitsuba):
- Per-pixel confidence → guide adaptive sampling WITHIN a single render

**Revised Mitsuba Plan**:
```python
# PASS 1: Quick preview (4 spp)
preview = mi.render(scene, spp=4)
confidence_map = jepa_model.predict_confidence(preview, scene_graph)

# PASS 2: Targeted rendering
# Since we can't do per-pixel adaptive, we use REGION-BASED adaptive:
# - Split image into tiles (e.g., 64x64)
# - For each tile: if avg_confidence > threshold, render at low spp
#                    else render at high spp

final_image = ImageBlock(scene.sensor.film().size())
for tile in tiles:
    tile_confidence = confidence_map[tile].mean()
    if tile_confidence > 0.8:
        tile_result = mi.render(scene, spp=4, crop=tile.bounds)
    else:
        tile_result = mi.render(scene, spp=128, crop=tile.bounds)
    final_image.put_block(tile_result)

# JEPA fuses the multi-spp tiles
final = jepa_model.fuse_multispp(final_image, confidence_map, scene_graph)
```

**Value**: Still 4-8x sample reduction, but with tile granularity instead of per-pixel.

---

### Mode 3: Multi-Resolution Merge (SPATIAL)

**Status**: ✅ **FULLY VIABLE** - No competition with Mitsuba

```python
# PASS 1: Low resolution, high quality
scene.sensor.film().set_size([height//4, width//4])
low_res_high_qual = mi.render(scene, spp=256)

# PASS 2: High resolution, noisy
scene.sensor.film().set_size([height, width])
high_res_noisy = mi.render(scene, spp=4)

# JEPA merges using scene graph knowledge
final = jepa_model.merge_multires(
    low_res_high_qual,
    high_res_noisy,
    scene_graph
)
```

**Value**: Same as Cycles plan - 8-16x effective speedup.

---

## Implementation Architecture

### Python Side (omen_integrator/)

```
src/omen_integrator/
├── __init__.py          # OmenIntegrator class + registration
├── core.py              # render_path_tracer() - orchestrate multi-pass
├── jepa_bridge.py       # ctypes bridge to Mojo .so (NEW)
├── scene_extractor.py   # Extract Mitsuba scene → structured tensors (NEW)
├── tile_renderer.py     # Tile-based adaptive rendering (NEW)
└── modes/
    ├── denoiser.py      # Mode 1: Post-process denoising
    ├── adaptive.py      # Mode 2: Tile-based adaptive sampling
    └── multires.py      # Mode 3: Multi-resolution merge
```

### Mojo Side (jepa_kernels/)

```
jepa_kernels/
├── jepa.mojo            # Main JEPA model definition
├── scene_encoder.mojo   # Transformer over scene tokens
├── image_encoder.mojo   # Strided convolutions
├── cross_attention.mojo # JEPA core world model
├── confidence.mojo      # Confidence head (Mode 2)
├── decoder.mojo         # Latent → pixels
└── C_ABI.mojo           # C interface for Python ctypes
└── build.sh             # Compile to .so/.dll/.dylib
```

---

## Data Flow: Mitsuba Scene → JEPA

### Scene Graph Extraction

```python
def extract_mitsuba_scene_graph(scene: mi.Scene) -> SceneGraph:
    """Extract structured scene data from Mitsuba scene."""

    # Geometry
    meshes = []
    for shape in scene.shapes():
        if hasattr(shape, 'vertex_positions'):
            vertices = dr.ravel(shape.vertex_positions())  # [N*3]
            faces = shape.faces() if hasattr(shape, 'faces') else None
            meshes.append({'vertices': vertices, 'faces': faces})

    # Materials (BSDF parameters)
    materials = []
    for shape in scene.shapes():
        if hasattr(shape, 'bsdf'):
            bsdf = shape.bsdf()
            # Extract BSDF parameters (diffuse reflectance, roughness, etc.)
            materials.append(extract_bsdf_params(bsdf))

    # Lights (emitters)
    lights = []
    for emitter in scene.emitters():
        # Position, type, intensity, color
        lights.append(extract_emitter_params(emitter))

    # Camera
    sensor = scene.sensors()[0]
    camera = extract_camera_params(sensor)

    return SceneGraph(
        geometry=meshes,
        materials=materials,
        lights=lights,
        camera=camera
    )
```

---

## Mode 2: Tile-Based Adaptive Sampling

### Tiling Strategy

```python
def adaptive_tile_render(scene, sensor, integrator, confidence_map, tile_size=64):
    """Render tiles at different spp based on JEPA confidence."""

    H, W = sensor.film().size()
    final_image = mi.ImageBlock([W, H])

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            tile_bounds = [x, y, min(x+tile_size, W), min(y+tile_size, H)]

            # Get average confidence for this tile
            tile_conf = confidence_map[y:y+tile_size, x:x+tile_size].mean()

            # Allocate samples based on confidence
            if tile_conf > 0.8:
                spp = 4    # High confidence: minimal samples
            elif tile_conf > 0.5:
                spp = 16   # Medium confidence
            else:
                spp = 128  # Low confidence: full samples

            # Render this tile
            # Note: Mitsuba doesn't support crop rendering in Python API
            # Workaround: render full image at different spp, then combine
            tile_result = mi.render(scene, spp=spp)

            # Copy tile region to final image
            final_image.put_block(tile_result, offset=[x, y])

    return final_image
```

**Problem**: Mitsuba's Python API doesn't support crop rendering.

**Solution**: Use full-image multi-pass with different sample allocations:

```python
def adaptive_multPass_render(scene, sensor, integrator, confidence_map):
    """Render multiple passes with different sample allocations."""

    # Pass 1: Base render (4 spp)
    base_render = mi.render(scene, spp=4)

    # Pass 2: High-spp render for low-confidence regions only
    # Strategy: Render full image at high spp, but combine intelligently
    high_spp_render = mi.render(scene, spp=128)

    # JEPA merges per-pixel based on confidence
    # - High confidence pixels: use base_render (JEPA-predicted)
    # - Low confidence pixels: use high_spp_render (path-traced)
    final = jepa_model.merge_adaptive(
        base_render,
        high_spp_render,
        confidence_map
    )

    return final
```

---

## C ABI Interface (Python ↔ Mojo)

```c
// omen_bridge.h
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

// Mode 1: Denoise
int omen_denoise(
    SceneGraph* scene,
    RenderObservation* obs,
    float* output_rgba,   // [H, W, 4]
    int    gpu_device_id
);

// Mode 2: Confidence prediction
int omen_predict_confidence(
    SceneGraph* scene,
    RenderObservation* obs,
    float* output_rgba,      // [H, W, 4] - denoised
    float* output_confidence,// [H, W, 1] - 0=uncertain, 1=confident
    int    gpu_device_id
);

// Mode 3: Multi-resolution merge
int omen_merge_multires(
    SceneGraph* scene,
    float* low_res_high_qual,   // [H/4, W/4, 4]
    float* high_res_noisy,      // [H, W, 4]
    int    scale_factor,
    float* output_merged,       // [H, W, 4]
    int    gpu_device_id
);
```

---

## Implementation Phases

### Phase 1: Skeleton + Scene Extraction (Week 1-2)
- [ ] Scene extractor from Mitsuba scene graph
- [ ] C ABI header + Mojo skeleton
- [ ] ctypes bridge in Python
- [ ] Test: extract scene from Cornell box

### Phase 2: Mode 1 - Denoiser (Week 3-5)
- [ ] Image encoder in Mojo (Nabla)
- [ ] Scene encoder in Mojo
- [ ] JEPA cross-attention
- [ ] Decoder in Mojo
- [ ] Training loop: (4spp + scene) → 256spp
- [ ] Test: denoise Cornell box at 4 spp

### Phase 3: Mode 2 - Confidence (Week 6-8)
- [ ] Confidence head in Mojo
- [ ] Multi-pass render orchestration in Python
- [ ] Adaptive merge in Mojo
- [ ] Training: variance → confidence labels
- [ ] Test: adaptive sampling on complex scene

### Phase 4: Mode 3 - Multi-Res (Week 9-11)
- [ ] Multi-resolution merge in Mojo
- [ ] Resolution change orchestration
- [ ] Training: low-res + noisy → high-res clean
- [ ] Test: 25% res 512spp + 100% res 4spp

### Phase 5: Integration & Distribution (Week 12)
- [ ] Package Mojo .so with addon
- [ ] Blender addon UI (if using Blender-Mitsuba)
- [ ] Documentation
- [ ] Test scenes

---

## Key Differences from Cycles Plan

| Aspect | Cycles Plan | Mitsuba Plan |
|--------|-------------|--------------|
| **Adaptive granularity** | Per-pixel (via Cycles API) | Per-tile or per-pass (Python API limitation) |
| **Integrator modification** | Can set adaptive_threshold per pixel | Cannot modify C++ sampling loop |
| **Multi-pass strategy** | Settings changes between renders | Full re-renders at different spp |
| **Scene extraction** | `bpy.data` (Blender API) | `mi.Scene` (Mitsuba API) |
| **Render passes** | Blender render passes | Mitsuba Film/ImageBlock |
| **Temporal fusion** | Via Vector pass | Need motion vectors (if supported) |

---

## Summary: Where JEPA Adds Value

### ✅ What JEPA Does (No Competition):

1. **Scene-aware denoising** (Mode 1)
   - Mitsuba has NO built-in denoiser
   - OptiX denoiser is separate and 2D-only
   - JEPA knows the 3D scene → better quality

2. **Confidence-based adaptive sampling** (Mode 2)
   - Mitsuba has NO adaptive sampling (confirmed by GitHub issue #21)
   - JEPA classifies "easy" vs "hard" regions
   - Allocate 4-32x fewer samples to "easy" regions

3. **Scene-guided multi-resolution** (Mode 3)
   - Mitsuba has NO upscaling or multi-res guidance
   - JEPA knows exact geometry → avoids DLSS-style artifacts
   - Merge 25% res 512spp + 100% res 4spp → clean 1080p

### ❌ What We Don't Do (Reinventing):

1. **Path tracing algorithm** - Use Mitsuba's (17 years of optimization)
2. **BSDF sampling** - Use Mitsuba's (all material types)
3. **Next event estimation** - Use Mitsuba's (MIS optimized)
4. **Russian roulette** - Use Mitsuba's (proper termination)

---

## References

- Mitsuba 3 Path Tracer: https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/integrators/path.cpp
- Custom Plugin Tutorial: https://mitsuba.readthedocs.io/en/stable/src/others/custom_plugin.html
- Adaptive Sampling Issue: https://github.com/mitsuba-renderer/mitsuba-tutorials/issues/21
- JEPA Denoiser Plan: `/docs/JEPA_DENOISER_ADDON_PLAN.md`
