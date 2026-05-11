# Omen — Scene-Aware JEPA Render Accelerator for Blender

## What Omen Is

Omen is a Blender addon with three escalating modes of operation:

| Mode | What It Does | Impact | Complexity |
|------|-------------|--------|------------|
| **Mode 1: Denoiser** | Post-process denoising with 3D scene knowledge | Replace OptiX/OIDN | Baseline |
| **Mode 2: Accelerator** | Predict "obvious" pixels, only path-trace the "confusing" ones | 4-8x fewer samples needed | Medium |
| **Mode 3: Multi-Resolution** | High spp at low res + low spp at high res, JEPA fills the gap | 8-16x effective speedup | Hard |

All three share the same core: a JEPA world model that understands the 3D scene
from `bpy.data`, not just 2D render passes.

## The Fundamental Advantage

Facebook's 3D-JEPA (Locate 3D, arxiv 2504.14151) asks: *"What's in this scene?"*
They get noisy RGB-D sensor data. No ground truth. They need CLIP, DINO, SAM.
They voxelise at 5cm resolution. 300M params. Months of annotation.

**We ask: "Given this exact scene I built, predict what the clean render looks like."**

Blender gives us:
- Exact geometry from the evaluated BVH (not sensor depth)
- Exact material parameters from Principled BSDF inputs (not pixel inference)
- Exact light positions, types, energies (not estimation)
- **Ground truth on demand** — render same frame at any spp
- **Curriculum control** — render at 1, 4, 16, 64 spp, we set difficulty

## How Omen Tampering With Cycles Works

We do NOT modify Cycles source code. We do NOT crash the render. We use Python
hooks and settings that Blender already exposes:

### The Levers We Control From Python

```python
import bpy

scene = bpy.context.scene
cscene = scene.cycles  # Cycles render settings, all settable from Python

# --- Adaptive Sampling (the key lever) ---
cscene.use_adaptive_sampling = True
cscene.adaptive_threshold = 0.01     # noise threshold (0.1 to 0.001)
cscene.adaptive_min_samples = 0      # min samples before adapting

# --- Sample Count ---
cscene.samples = 1024                # max samples
cscene.preview_samples = 32          # viewport samples

# --- Resolution ---
scene.render.resolution_x = 1920
scene.render.resolution_percentage = 50  # render at 50% then upscale

# --- Render Engine Selection ---
scene.render.engine = 'CYCLES'       # or 'BLENDER_EEVEE_NEXT'

# --- Tile Size ---
cscene.tile_size = 2048              # GPU tile size

# --- Time Limit ---
cscene.time_limit = 0.0              # seconds, 0 = unlimited

# --- Denoiser ---
cscene.use_denoising = True
cscene.denoiser = 'OPTIX'            # 'OIDN', 'OPENIMAGEDENOISE'
```

### The Hooks We Intercept

```python
# These fire at specific render lifecycle points:
bpy.app.handlers.render_pre       # BEFORE render starts
bpy.app.handlers.render_post      # AFTER each frame completes
bpy.app.handlers.render_complete  # AFTER entire render job
bpy.app.handlers.render_cancel    # IF user cancels
bpy.app.handlers.render_stats     # WHEN stats are printed
```

### The Render Passes We Read

```python
# After render_post fires, we access render results:
result = scene.render.layers.active  # RenderLayer
for render_pass in result.passes:
    pass_name = render_pass.name     # "Combined", "Vector", "Depth", etc.
    rect = render_pass.rect          # flat float array, width*height*channels
```

### How Omen Uses These Without Crashing Cycles

**The critical rule**: We NEVER modify Cycles state DURING rendering. We only:
1. **Before render** (`render_pre`): Set `adaptive_threshold`, `samples`, resolution
2. **After render** (`render_post`): Read passes, run Omen JEPA, write back result
3. **Between frames**: Adjust settings for next frame based on Omen's prediction

Cycles sees normal Python settings changes. It has no idea Omen is controlling it.
It thinks it's following rules — but those rules come from a neural world model.

---

## Data Sources — What Blender Knows

### Render Observation (2D, from render passes)

| Pass | Content | Channels |
|------|---------|----------|
| Combined | Noisy RGBA | 4 |
| Vector | Motion vectors (prev_xy + next_xy) | 4 |
| Depth | Z-buffer | 1 |
| Normal | World-space normals | 3 |
| Diffuse Color | Albedo without lighting | 3 |

### Scene Graph (3D, from evaluated depsgraph + BVH)

#### Evaluated Geometry

```python
import bpy, bmesh, mathutils
from mathutils.bvhtree import BVHTree

depsgraph = bpy.context.evaluated_depsgraph_get()

for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    # Evaluated mesh = base mesh + all modifiers applied:
    #   Subdivision Surface, Armature, Shape Keys, Mirror, Boolean, Array,
    #   Geometry Nodes (except displacement — see caveat)
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()

    for vert in mesh.vertices:
        world_pos = obj.matrix_world @ vert.co  # exact float3

    for poly in mesh.polygons:
        vert_indices = [mesh.loops[i].vertex_index
                        for i in range(poly.loop_start, poly.loop_start + poly.loop_total)]
        face_normal = poly.normal
        face_area = poly.area          # for importance sampling
        mat_index = poly.material_index

    # BVH for spatial queries (ray cast, nearest point)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.transform(obj.matrix_world)
    bvh = BVHTree.FromBMesh(bm)

    obj_eval.to_mesh_clear()
```

**Displacement shader caveat**: Cycles micro-displacement creates geometry inside
the render kernel. NOT accessible from `bpy`. Phase 2+ research item.

#### Materials — Direct BSDF Read

```python
# Blender 4.0+ Principled BSDF — 22+ inputs, read directly
BSDF_INPUTS = [
    "Base Color", "Roughness", "Metallic", "IOR",
    "Transmission Weight", "Coat Weight", "Sheen Weight",
    "Emission Color", "Emission Strength", "Alpha",
    "Subsurface Weight", "Subsurface Radius", "Subsurface Scale",
    "Specular IOR Level", "Specular Tint", "Anisotropic",
    "Anisotropic Rotation", "Coat Roughness", "Coat IOR",
    "Coat Tint", "Sheen Roughness", "Sheen Tint",
    "Thin Film IOR", "Thin Film Thickness",
]

for mat in bpy.data.materials:
    if not mat.use_nodes:
        continue
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        continue
    for name in BSDF_INPUTS:
        socket = bsdf.inputs.get(name)
        if socket:
            val = socket.default_value
            # float or RGB/RGBA vector
```

#### Lights

```python
for obj in bpy.data.objects:
    if obj.type != 'LIGHT':
        continue
    light = obj.data
    world_pos = obj.matrix_world.translation
    world_dir = obj.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
    # type: POINT, SUN, SPOT, AREA
    # energy (watts), color (rgb), spot_size, area_size
```

#### Camera

```python
cam_obj = bpy.context.scene.camera
view_matrix = cam_obj.matrix_world.inverted()
projection_matrix = cam_obj.data.calc_matrix_camera(depsgraph)
fov = cam_obj.data.angle
clip_start, clip_end = cam_obj.data.clip_start, cam_obj.data.clip_end
```

---

## Omen Mode 1: Denoiser

Post-processing. Cycles renders normally at low spp. Omen denoises after.

```
FLOW:
  render_pre:  Extract scene graph from depsgraph
  Cycles renders at 4-16 spp (normal pipeline, nothing modified)
  render_post: Read render passes -> Omen JEPA -> denoised output -> write back
```

**Architecture**:
```
Scene Graph Encoder (transformer, ~2-5M params)
  Input: geometry tokens + material table + light tokens + camera state
  Output: scene_embedding [N_tokens, 128]

Image Encoder (strided conv, ~1-3M params)
  Input: noisy_combined + depth + normal + albedo
  Output: image_latent [H/8, W/8, 128]

JEPA Cross-Attention (~1-3M params)
  Query: image_latent (noisy observation)
  Key/Value: scene_embedding (what the scene SHOULD look like)
  Output: scene-conditioned latent

Temporal Fusion (if not first frame / camera cut)
  Warp previous denoised via motion vectors
  Cross-attend current <-> warped previous
  Learnable gate: temporal weight vs scene-only weight

Decoder (conv transpose, ~1-2M params)
  Output: denoised RGBA [H, W, 4]
```

**Self-training**: Render same frame at 16spp (noisy) and 256spp (ground truth).
Perfect supervised pair. As many as we want. No external data.

---

## Omen Mode 2: Accelerator

Omen predicts which pixels are "obvious" and which are "confusing". Cycles only
path-traces the confusing pixels. The "obvious" pixels are filled by the JEPA.

This is **predictive decoding for rendering**, analogous to speculative decoding
in LLMs: a draft model predicts tokens, the full model verifies only the uncertain
ones. Omen predicts pixel values, Cycles verifies only the uncertain ones.

### How It Works — The Multi-Pass Trick

We cannot inject into Cycles' inner sampling loop from Python. But we CAN render
multiple passes and adjust settings between them:

```python
# PASS 1: Quick preview render (4 spp)
scene.cycles.samples = 4
scene.cycles.use_adaptive_sampling = False
bpy.ops.render.render()  # Cycles renders normally
# -> render_post fires
# -> Omen reads passes, runs JEPA
# -> Omen produces: denoised_estimate + per_pixel_confidence_map

# PASS 2: Targeted re-render — only where Omen is uncertain
# Omen sets adaptive_threshold LOW for confident pixels
# (Cycles will stop sampling them quickly)
# and HIGH for uncertain pixels (Cycles will keep sampling them)
scene.cycles.samples = 128
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.001  # tight for ALL pixels initially
# BUT we can't set per-pixel threshold — Cycles uses one global value.
# Instead: render full pass, let adaptive sampling equalize, then Omen
# combines Pass 1 (scene-predicted) + Pass 2 (path-traced) weighted by confidence
bpy.ops.render.render()
# -> render_post fires
# -> Omen merges: high-confidence pixels from JEPA prediction,
#                 low-confidence pixels from Cycles pass 2
```

### Per-Tile Steering (alternative approach)

Cycles renders in tiles. We can detect which tile is being rendered and adjust:

```python
# More aggressive: render in tiles manually
scene.render.tile_x = 256
scene.render.tile_y = 256

# For each tile:
#   1. Omen predicts: "this tile is easy" or "this tile is hard"
#   2. If easy: set samples=4, threshold=0.1
#   3. If hard (glass, caustics, SSS): set samples=256, threshold=0.001
#   4. Render just this tile
#   5. Omen denoises result
```

### What Makes Pixels "Obvious" vs "Confusing"

Omen's scene encoder can classify from `bpy.data`:

```
OBVIOUS (Omen predicts, minimal samples):
  - Flat diffuse surface with known albedo → Lambertian is trivial
  - Surface lit by known point light → direct illumination is analytic
  - Background/sky → no path tracing needed
  - Already-converged regions from previous frames (temporal reuse)

CONFUSING (Cycles path-traces fully):
  - Glass/transmission surfaces → caustic variance
  - Subsurface scattering → stochastic scattering
  - Volumes/heterogeneous media → expensive random walk
  - Sharp specular highlights → view-dependent, hard to predict
  - Shadow boundaries → visibility discontinuity
  - First frame / camera cut → no temporal context
```

The confidence map is a second output head on the same JEPA model:
```
JEPA model → denoised_pixels [H, W, 4]
           → confidence_map [H, W, 1]  (0=uncertain, 1=confident)
```

### Research Validation

This approach is proven:
- **NVIDIA Neural Temporal Adaptive Sampling (2020)**: CNN predicts sample map +
  denoises. Co-trained. 2-3x speedup. But NO scene knowledge.
- **NVIDIA Neural Radiance Caching (2021, Müller et al.)**: Self-trains while
  rendering. Predicts indirect illumination. 2.6ms overhead at 1080p. Zero
  pretraining — generalizes via adaptation. THIS IS THE CLOSEST TO OUR APPROACH.
- **"Forget Superresolution, Sample Adaptively" (2025)**: Prediction-based
  sampling at sub-1-spp. Proves you can allocate samples intelligently.
- **Offline Deep Importance Sampling (2019, Bako et al.)**: Learn sampling
  distributions offline, deploy at runtime. No temporal, no scene semantics.

Nobody has combined: scene graph conditioning + JEPA + adaptive sampling + Blender.

---

## Omen Mode 3: Multi-Resolution Intelligence

Render high spp at low resolution + low spp at high resolution. JEPA fills the gap.

This is NOT upscaling like DLSS/FSR. Those models see 2D pixels and guess detail.
Omen knows the actual 3D geometry, materials, and lighting — it doesn't guess, it
predicts from physics.

```
PASS 1: Low resolution, high samples
  scene.render.resolution_percentage = 25   # 480p from 1920p
  scene.cycles.samples = 512                # high quality convergence
  bpy.ops.render.render()
  -> clean_low_res: [H/4, W/4, 4]          # converged, but aliased

PASS 2: High resolution, low samples
  scene.render.resolution_percentage = 100  # full 1920p
  scene.cycles.samples = 4                  # noisy but full detail
  bpy.ops.render.render()
  -> noisy_high_res: [H, W, 4]             # full geometry detail, noisy

OMEN JEPA:
  Input: clean_low_res (upsampled) + noisy_high_res + scene_graph
  Scene graph provides: exact geometry edges, material boundaries, normals
  Output: final_high_res: [H, W, 4]        # clean AND detailed
```

### Why This Works Better Than DLSS

DLSS/FSR/XeSS see: low-res pixels → neural upscaler → high-res pixels.
They have NO knowledge of:
- The mesh edges that create aliased boundaries
- The material difference between two adjacent pixels
- The light source causing a specific highlight pattern
- Whether noise is from Monte Carlo variance or actual detail

Omen knows ALL of this from `bpy.data`. When upscaling, it can:
- Detect geometry silhouettes from the BVH (not guessing from pixels)
- Know material boundaries exactly (material_index per face)
- Understand that a bright pixel is a specular highlight (material has metallic=0.9)
  not noise
- Separate actual scene detail from rendering noise

### Research Validation

- **AMD Neural Supersampling + Denoising (GPUOpen)**: Single U-Net for joint
  denoise+upscale from 1 spp. But no scene knowledge.
- **"High-Fidelity 4x Neural Reconstruction" (ICCV 2025)**: 4x upscale from
  noisy path-traced video. But no 3D scene context.
- **Omen advantage**: Same task, but the model has the actual 3D scene. It knows
  what edges SHOULD exist. It knows what materials are where. It knows the light
  configuration. This is strictly more information than any existing upscaler.

---

## Temporal System (Shared Across All Modes)

### Camera Cut Detection

```python
def is_camera_cut(cam_prev, cam_curr, dist_thresh=0.5, angle_thresh=30):
    distance = (cam_prev.translation - cam_curr.translation).length
    angle = cam_prev.to_quaternion().rotation_difference(
             cam_curr.to_quaternion()).angle
    return distance > dist_thresh or angle > math.radians(angle_thresh)
```

At a cut: reset ALL temporal state. Scene-only denoising. First frame problem.

### Motion Vector Handling

Blender Vector pass: Red/Green = motion to previous frame, Blue/Alpha = from next.
2D pixel space. Used to warp previous denoised into current frame alignment.

Edge cases: first frame (no prev), last frame (no next), fast motion (inaccurate
warping). Omen's attention learns to downweight temporal context when motion
vectors are unreliable.

---

## Self-Training Protocol (Shared Across All Modes)

```
PHASE 1 — Bootstrap (frames 1-5):
  Render each frame at 16spp AND 256spp
  Train: (low_spp + scene_graph) -> high_spp
  ~5 minutes total on RTX 3060

PHASE 2 — Progressive (frames 6-20):
  Render at 8spp, Omen denoises
  Every 5th frame: also render 128spp as new ground truth
  Online training continues

PHASE 3 — Converged (frames 21+):
  Render at 4spp, Omen denoises with scene + temporal
  Model is expert on THIS scene
  SPEEDUP: 16-64x fewer samples

For Mode 2 (Accelerator):
  Same protocol, but Omen also learns to predict confidence maps
  Ground truth for confidence: variance across multiple low-spp renders
  (render same frame 8x at 4spp, variance = ground truth uncertainty)

For Mode 3 (Multi-Res):
  Low-res high-spp serves as ground truth for the spatial detail prediction
  High-res low-spp provides the detail signal
  Omen learns to merge them using scene graph knowledge
```

---

## Mojo Implementation

### C ABI Interface

```c
typedef struct {
    float* geometry_points;      // [N_points, 8]
    int    n_points;
    float* material_features;    // [N_materials, 16]
    int    n_materials;
    float* light_features;       // [N_lights, 14]
    int    n_lights;
    float* camera_state;         // [36]
    int    is_camera_cut;
    int    has_prev_frame;
    int    has_next_frame;
} SceneGraph;

typedef struct {
    float* noisy_combined;       // [H, W, 4]
    float* motion_vectors;       // [H, W, 4]
    float* depth;                // [H, W, 1]
    float* normal;               // [H, W, 3]
    float* albedo;               // [H, W, 3]
    float* prev_denoised;        // [H, W, 4] or NULL
    // Mode 3 additional inputs:
    float* clean_low_res;        // [H/R, W/R, 4] or NULL
    int    low_res_scale;        // R=4 for 25% resolution
    int    height;
    int    width;
} RenderObservation;

// Mode control
#define OMEN_MODE_DENOISER      0
#define OMEN_MODE_ACCELERATOR   1
#define OMEN_MODE_MULTIRES      2

int omen_process(
    SceneGraph* scene,
    RenderObservation* obs,
    float* output_denoised,      // [H, W, 4]
    float* output_confidence,    // [H, W, 1] — Mode 2 only, NULL for Mode 1
    int    mode,                 // OMEN_MODE_DENOISER / ACCELERATOR / MULTIRES
    int    training_mode,        // 1=train, 0=inference
    float* ground_truth,         // [H, W, 4] or NULL
    int    gpu_device_id
);
```

### Mojo GPU Kernels

```mojo
from std.gpu import global_idx, barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

# Scene encoder — transformer over geometry + material + light tokens
def scene_encoder[dtype: DType, LT: TensorLayout](
    geometry: TileTensor[dtype, LT, MutAnyOrigin],
    materials: TileTensor[dtype, LT, MutAnyOrigin],
    lights: TileTensor[dtype, LT, MutAnyOrigin],
    camera: TileTensor[dtype, LT, MutAnyOrigin],
    output: TileTensor[dtype, LT, MutAnyOrigin],
):
    comptime assert geometry.flat_rank == 2
    # Multi-head self-attention, cross-attention to materials/lights
    ...

# Image encoder — strided convolutions
def image_encoder[dtype: DType](
    noisy: TileTensor[dtype, ..., MutAnyOrigin],
    depth: TileTensor[dtype, ..., MutAnyOrigin],
    normal: TileTensor[dtype, ..., MutAnyOrigin],
    albedo: TileTensor[dtype, ..., MutAnyOrigin],
    output_latent: TileTensor[dtype, ..., MutAnyOrigin],
):
    ...

# JEPA cross-attention — core world model
def jepa_cross_attention[dtype: DType](
    image_latent: TileTensor[dtype, ..., MutAnyOrigin],
    scene_embedding: TileTensor[dtype, ..., MutAnyOrigin],
    is_camera_cut: Int,
    prev_latent: TileTensor[dtype, ..., MutAnyOrigin],
    output: TileTensor[dtype, ..., MutAnyOrigin],
):
    ...

# Confidence head — Mode 2 only, predicts per-pixel certainty
def confidence_head[dtype: DType](
    fused_latent: TileTensor[dtype, ..., MutAnyOrigin],
    scene_embedding: TileTensor[dtype, ..., MutAnyOrigin],
    output_confidence: TileTensor[dtype, ..., MutAnyOrigin],
):
    comptime assert output_confidence.flat_rank == 3
    # [H/8, W/8, 1] — sigmoid output, 0=uncertain, 1=confident
    ...

# Multi-resolution merge — Mode 3 only
def multires_merge[dtype: DType](
    clean_low_res: TileTensor[dtype, ..., MutAnyOrigin],   # upsampled
    noisy_high_res: TileTensor[dtype, ..., MutAnyOrigin],
    scene_embedding: TileTensor[dtype, ..., MutAnyOrigin],
    output: TileTensor[dtype, ..., MutAnyOrigin],
):
    # Scene-guided merge: geometry edges, material boundaries,
    # light highlights — all from scene graph, not pixel guessing
    ...

# Decoder — latent to pixels
def decoder[dtype: DType, H: Int, W: Int](
    latent: TileTensor[dtype, ..., MutAnyOrigin],
    output: TileTensor[dtype, ..., MutAnyOrigin],
):
    ...
```

---

## Blender Addon Structure

```
omen/
    bl_manifest.toml
    __init__.py
    operators.py                  # Render + denoise operator (3 modes)
    handlers.py                   # render_pre, render_post handlers
    properties.py                 # User settings: mode, spp, training toggle
    ui.py                         # Panel in Render properties
    scene_encoder.py              # Encode bpy.data -> structured tensors
    bvh_extractor.py              # Read evaluated geometry from depsgraph
    material_reader.py            # Read Principled BSDF inputs
    light_reader.py               # Read light properties
    motion.py                     # Motion vectors, camera cut detection
    confidence.py                 # Mode 2: confidence map -> sample allocation
    multires.py                   # Mode 3: multi-resolution render orchestration
    denoise_core.py               # ctypes bridge to Mojo .so
    train.py                      # Self-training orchestration
    ground_truth.py               # High-spp render management
    settings_steering.py          # Mode 2: dynamically set Cycles params per frame
    libs/
        linux/x86_64/omen.so
        darwin/arm64/omen.dylib
        windows/amd64/omen.dll
```

## Blender Extension Manifest

```toml
blender_version_min = "4.2.0"
id = "omen"
version = "0.1.0"
name = "Omen — Scene-Aware Render Accelerator"
tagline = "JEPA world model denoiser and accelerator for Cycles"
maintainer = "Contributor"
type = "add-on"
license = ["SPDX:GPL-3.0-or-later"]
tags = ["Render"]
permissions = ["files"]
```

---

## Test Scenes

| Scene | Tests | Mode Focus |
|-------|-------|------------|
| **Classroom** (72MB) | Many small lights, indirect bounce, glossy | Mode 1 baseline |
| **Barcelona Pavilion** (24MB) | Glass, transmission, caustics | Mode 2 confidence |
| **Cosmos Laundromat** | Camera cuts, fur, outdoor lighting | Temporal + cuts |
| **Spring** | Dense vegetation, thousands of lights | Mode 2 speed test |
| **Charge** (1.4GB) | Metallic, hard surface, dramatic lighting | Mode 3 multi-res |

---

## Development Phases

### Phase 1: Skeleton + Scene Extraction (Month 1-2)
- Addon with handlers, UI, manifest
- `bvh_extractor.py`, `material_reader.py`, `light_reader.py`
- Mojo `.so` with bilateral filter (non-ML baseline)
- ctypes bridge end-to-end verified
- Test on Classroom scene

### Phase 2: Mode 1 — Denoiser (Month 3-5)
- Scene encoder + image encoder + JEPA cross-attention + decoder in Nabla
- Training loop: (low_spp + scene_graph) -> high_spp
- Temporal fusion + camera cut handling
- Benchmark vs OptiX/OIDN on SSIM/LPIPS
- Test on all 5 battle scenes

### Phase 3: Mode 2 — Accelerator (Month 5-7)
- Confidence head (second output on same model)
- `confidence.py`: confidence map -> sample allocation
- `settings_steering.py`: dynamically adjust `adaptive_threshold` per frame
- Multi-pass render: preview -> Omen classifies -> targeted re-render
- Measure: how many samples does Omen save vs uniform?

### Phase 4: Mode 3 — Multi-Resolution (Month 7-9)
- `multires.py`: orchestrate low-res/high-spp + high-res/low-spp renders
- Multi-resolution merge kernel with scene graph guidance
- Test: render at 25% res with 512spp + 100% res with 4spp -> merge
- Benchmark vs DLSS-style upscaling (but with scene knowledge)

### Phase 5: Package + Distribute (Month 9-10)
- Cross-platform builds (Linux, macOS, Windows)
- Submit to extensions.blender.org
- GPL-3.0-or-later
- Documentation with battle scene demos

---

## Displacement Shader Research (Phase 2+)

Cycles micro-displacement creates geometry inside the render kernel. NOT accessible
from `bpy`. The evaluated depsgraph gives the base mesh, not the displaced surface.

Research paths:
1. Sample displacement texture from Python, compute displacement manually
2. Patch Cycles to export displaced geometry (custom build)
3. Pass displacement texture params as material features — model learns
   "material with displacement X produces noise pattern Y"
4. Accept approximation — subtle surface detail may not need exact vertices

---

## Comparison Table

| Aspect | OptiX Denoiser | OIDN | NVIDIA Neural Adaptive (2020) | AMD Neural SS (2024) | **Omen** |
|--------|----------------|------|------------------------------|----------------------|----------|
| Scene knowledge | None (2D) | None (2D) | None (2D) | None (2D) | **Full 3D from bpy** |
| Self-training | Pre-trained | Pre-trained | Pre-trained | Pre-trained | **On-scene, free GT** |
| Adaptive sampling | No | No | Yes (blind) | No | **Scene-guided** |
| Multi-resolution | No | No | No | Yes (blind) | **Scene-guided** |
| Temporal | Yes | No | Yes | Yes | **Yes + scene** |
| Runs on | NVIDIA only | Any | NVIDIA only | AMD only | **Any GPU (Mojo)** |
| Open source | No | Yes | No | No | **Yes (GPL)** |
| Model size | ~10M | ~5M | ~20M | ~20M | **5-15M** |

---

## References

### World Model / JEPA
- LeWorldModel: github.com/lucas-maes/le-wm — 15M params, trains in hours
- Facebook 3D-JEPA (Locate 3D): arxiv 2504.14151 — SSL on point clouds (CC-BY-NC)
- V-JEPA 2 (Meta, 2025) — video world model at 8B scale
- I-JEPA (Assran et al., CVPR 2023) — Image JEPA

### Neural Adaptive Sampling + Denoising
- NVIDIA Neural Temporal Adaptive Sampling (Hasselgren et al., 2020) — co-trained sampling + denoising
- NVIDIA Neural Radiance Caching (Müller et al., 2021) — self-trains while rendering, 2.6ms overhead
- "Forget Superresolution, Sample Adaptively" (2025) — prediction-based sampling at sub-1-spp
- Offline Deep Importance Sampling (Bako et al., 2019) — learn sampling distributions offline
- AMD Neural Supersampling + Denoising (GPUOpen) — joint denoise+upscale from 1 spp

### Blender API
- depsgraph: docs.blender.org/api/current/bpy.types.Depsgraph.html
- BVHTree: docs.blender.org/api/current/mathutils.bvhtree.html
- Cycles adaptive sampling: blender/intern/cycles/blender/addon/properties.py (L559-595)
- Render handlers: bpy_app_handlers.cc (L108-116)
- Principled BSDF: docs.blender.org/api/current/bpy.types.ShaderNodeBsdfPrincipled.html

### Mojo Ecosystem
- Nabla: github.com/nabla-ml/nabla — autograd + GPU + SPMD
- Vulkan-Mojo: github.com/Ryul0rd/vulkan-mojo — Vulkan bindings
- Mojo HPC: arXiv 2509.21039 — GPU kernel benchmarks
