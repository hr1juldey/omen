# Omen Architecture — HLD & LLD

## Table of Contents
1. [System Overview (HLD)](#system-overview)
2. [Component Architecture (LLD)](#component-architecture)
3. [Data Flows](#data-flows)
4. [Blender API Interface Map](#blender-api-interface-map)
5. [Mojo ↔ C/C++ FFI Bridge](#mojo--cc-ffi-bridge)
6. [Emissive Material Detection](#emissive-material-detection)
7. [Geometry Node Instance Lights](#geometry-node-instance-lights)
8. [Large Cache Management](#large-cache-management)
9. [JEPA Patch-Based Tile Processing](#jepa-patch-based-tile-processing)
10. [File Structure](#file-structure)

---

## System Overview (HLD)

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Blender 5.1+                                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │   bpy.data   │      │  Depsgraph   │      │   Render     │               │
│  │  (Scene DB)  │──────│  (Evaluated) │──────│   Engine     │               │
│  └──────────────┘      └──────────────┘      └──────────────┘               │
│         │                      │                      │                      │
│         ▼                      ▼                      ▼                      │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    Omen Python Addon                               │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │    │
│  │  │   Scene    │  │  Material  │  │    Light   │  │   Render   │   │    │
│  │  │ Extractor  │  │   Reader   │  │   Reader   │  │  Handler   │   │    │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                    C ABI / ctypes bridge                                    │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    Omen Mojo Core (.so)                             │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │    │
│  │  │    JEPA    │  │   Scene    │  │   Render   │  │   Memory   │   │    │
│  │  │   Model    │  │   Graph    │  │  Kernel    │  │  Manager   │   │    │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                    CUDA / HIP / Vulkan GPU APIs                            │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                        GPU (NVIDIA/AMD/Intel)                       │    │
│  │              TileTensor buffers / Shared memory                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Mojo Core, Python Shell**: All heavy computation (JEPA inference, scene graph processing) lives in Mojo. Python only orchestrates.
2. **Zero-Copy GPU**: CUDA/HIP pointers from Blender → Mojo `DeviceBuffer` via `raw_ptr` wrapping, no `memcpy`.
3. **Evaluated Geometry Only**: Always use `depsgraph.evaluated_get()` to capture modifiers, armatures, subdivision.
4. **JEPA Patch Alignment**: Render tiles align with JEPA patch size (default 16x16 patches of 8x8 pixels = 128x128 tile).

---

## Component Architecture (LLD)

### 1. Python Addon Components

#### `main.py` — Entry Point
```python
import bpy
import ctypes
from pathlib import Path

# Load Mojo shared library
_omen_lib = ctypes.CDLL(Path(__file__).parent / "lib" / "libomen_core.so")

# Register render handlers
def register():
    bpy.app.handlers.render_pre.append(omen_render_pre)
    bpy.app.handlers.render_post.append(omen_render_post)
```

#### `scene_extractor.py` — Scene Graph Extraction
```python
def extract_scene_graph(context):
    """Extract evaluated scene graph for Mojo"""
    depsgraph = context.evaluated_depsgraph_get()

    # BVH extraction (evaluated geometry only)
    bvh_trees = []
    for obj in depsgraph.objects:
        if obj.type == 'MESH':
            obj_eval = obj.evaluated_get(depsgraph)
            bm = bmesh.new()
            bm.from_object(obj_eval, depsgraph)
            bvh = BVHTree.FromBMesh(bm)
            bvh_trees.append({
                'object_id': id(obj),
                'bvh_handle': bvh,
                'vertices': len(bm.verts),
                'faces': len(bm.faces),
            })

    return bvh_trees
```

#### `material_reader.py` — Material Parameter Extraction
```python
def read_material_properties(material):
    """Extract Principled BSDF parameters for Mojo"""
    if not material.use_nodes:
        return None

    nodes = material.node_tree.nodes
    output = material.node_tree.nodes.get('Material Output')
    if not output:
        return None

    bsdf = output.inputs['Surface'].links[0].from_node if output.inputs['Surface'].is_linked else None
    if not isinstance(bsdf, bpy.types.ShaderNodeBsdfPrincipled):
        return None

    return {
        'base_color': bsdf.inputs['Base Color'].default_value[:4],
        'subsurface': bsdf.inputs['Subsurface'].default_value,
        'subsurface_radius': bsdf.inputs['Subsurface Radius'].default_value[:3],
        'metallic': bsdf.inputs['Metallic'].default_value,
        'roughness': bsdf.inputs['Roughness'].default_value,
        'ior': bsdf.inputs['IOR'].default_value,
        'transmission': bsdf.inputs['Transmission'].default_value,
        # ... 22+ parameters total
    }
```

#### `light_reader.py` — Light Source Detection
```python
def detect_light_sources(context):
    """Detect all light sources: light objects + emissive materials + instances"""
    depsgraph = context.evaluated_depsgraph_get()
    lights = []

    # 1. Light objects
    for obj in bpy.data.objects:
        if obj.data and obj.data.type == 'LIGHT':
            lights.append({
                'type': obj.data.type,
                'position': obj.matrix_world.translation,
                'energy': obj.data.energy,
                'color': obj.data.color,
                'is_light_object': True,
            })

    # 2. Emissive materials
    for material in bpy.data.materials:
        if is_material_emissive(material):
            lights.append({
                'type': 'EMISSIVE_MATERIAL',
                'material_name': material.name,
                'emission_strength': get_emission_strength(material),
            })

    # 3. Geometry node instances (CRITICAL)
    for instance in depsgraph.object_instances:
        if instance.is_instance:
            obj = instance.object
            matrix = instance.matrix_world
            # Check if instanced object has emissive material
            if has_emissive_material(obj):
                lights.append({
                    'type': 'GEOMETRY_NODE_INSTANCE',
                    'instance_id': instance.persistent_id,
                    'matrix': matrix,
                    'object': obj.name,
                })

    return lights
```

#### `render_handler.py` — Render Pipeline Control
```python
def omen_render_pre(scene):
    """Before render: configure Cycles adaptive sampling"""
    cscene = scene.cycles

    # Enable adaptive sampling
    cscene.use_adaptive_sampling = True
    cscene.adaptive_threshold = 0.01
    cscene.adaptive_min_samples = 0

    # Set moderate samples (JEPA will guide)
    cscene.samples = 256

def omen_render_post(scene, render_result):
    """After render: process passes through JEPA"""
    # Read render passes
    passes = {}
    for pass_data in render_result.passes:
        passes[pass_data.name] = pass_data.rect

    # Send to Mojo
    denoised = omen_core.process(
        passes['Combined'],
        passes['Depth'],
        passes['Vector'],
        passes['Normal'],
        mode=OMEN_MODE_DENOISER,
    )

    # Write back
    render_result.passes['Combined'].rect = denoised
```

### 2. Mojo Core Components

#### `lib/omen_core.mojo` — Main Entry Point
```mojo
from python import Python
from memory import UnsafePointer

@always_inline
fn c_string_to_mojo(ptr: UnsafePointer[C_char]) -> String:
    """Convert C string to Mojo String"""
    var len = 0
    var p = ptr
    while p.load() != 0:
        p += 1
        len += 1
    return String(ptr, len)

struct SceneGraph:
    var objects: List[MeshObject]
    var lights: List[LightSource]
    var materials: List[Material]

    fn __init__(inout self):
        self.objects = List[MeshObject]()
        self.lights = List[LightSource]()
        self.materials = List[Material]()

struct OmenCore:
    var scene: SceneGraph
    var jepa_model: JEPAModel
    var gpu_ctx: DeviceContext

    fn __init__(inout self, device_id: Int):
        self.scene = SceneGraph()
        self.jepa_model = JEPAModel()
        self.gpu_ctx = DeviceContext(device_id)

    @register_function
    fn process(
        inout self,
        noisy_ptr: UnsafePointer[C_float],
        depth_ptr: UnsafePointer[C_float],
        width: Int, height: Int,
        mode: Int,
    ) -> UnsafePointer[C_float]:
        """Main entry point from Python via ctypes"""
        let noisy = Tensor.from_raw_ptr(noisy_ptr, (width, height, 4))
        let depth = Tensor.from_raw_ptr(depth_ptr, (width, height))

        let denoised = self.jepa_model.denoise(noisy, depth, self.scene)

        return denoised.data.ptr
```

#### `jepa_model.mojo` — JEPA Architecture
```mojo
from tensor import Tensor, TensorShape
from autograd import AutogradContext

struct JEPAModel:
    var encoder: VisionEncoder
    var predictor: SpatialPredictor
    var patch_size: Int  # 8x8 pixel patches

    fn denoise(
        inout self,
        noisy: Tensor[DType.float32, 3],
        depth: Tensor[DType.float32, 2],
        scene: SceneGraph,
    ) -> Tensor[DType.float32, 3]:
        """JEPA denoising with scene context"""
        # 1. Encode to latent space
        let latent = self.encoder.encode(noisy)

        # 2. Extract spatial patches
        let patches = extract_patches(latent, self.patch_size)

        # 3. Predict clean patches (JEPA core)
        let predicted_patches = self.predictor.predict(patches, scene)

        # 4. Reconstruct image
        return reconstruct_from_patches(predicted_patches)
```

#### `memory_manager.mojo` — GPU Memory Control
```mojo
from buffer import DeviceBuffer

struct MemoryManager:
    var allocated_bytes: Int
    var budget_bytes: Int  # e.g., 4GB

    fn allocate_texture[
        T: DType
    ](inout self, width: Int, height: Int) -> DeviceBuffer[T]:
        """Allocate GPU texture with budget checking"""
        let required = width * height * sizeof[T]()
        if self.allocated_bytes + required > self.budget_bytes:
            # Evict least-recently-used textures
            self.evict_lru()

        let buffer = DeviceBuffer[T](width * height)
        self.allocated_bytes += required
        return buffer

    fn wrap_cuda_pointer[
        T: DType
    ](ptr: UnsafePointer[C_void], size: Int) -> DeviceBuffer[T]:
        """Zero-copy wrap of existing CUDA pointer (from Blender)"""
        return DeviceBuffer[T](UnsafePointer[T].bitcast(ptr), size, owning=False)
```

---

## Data Flows

### Flow 1: Scene Graph Extraction

```
Blender Scene (bpy.data)
    │
    ├─> depsgraph.evaluated_get()
    │   └─> obj.evaluated_get(depsgraph)
    │       └─> modifiers applied (subdivision, armature, etc.)
    │
    ├─> BVHTree.FromObject(obj, depsgraph, deform=True)
    │   └─> Spatial acceleration structure
    │
    └─> Python dict (serializable)
        │
        └─> ctypes pointer → Mojo
            └─> SceneGraph struct
                └─> GPU resident
```

### Flow 2: Material Parameter Reading

```
Material Slot
    │
    ├─> material.node_tree.nodes
    │   │
    │   ├─> ShaderNodeBsdfPrincipled
    │   │   ├─> inputs['Base Color'].default_value (float4)
    │   │   ├─> inputs['Roughness'].default_value (float)
    │   │   └─> inputs['Metallic'].default_value (float)
    │   │
    │   └─> ShaderNodeEmission
    │       ├─> inputs['Color'].default_value (float4)
    │       └─> inputs['Strength'].default_value (float)
    │
    └─> Python dict → C struct → Mojo struct
```

### Flow 3: Render Pass Processing

```
Cycles Render Complete
    │
    ├─> render_result.passes
    │   ├─> 'Combined' (RGBA float array)
    │   ├─> 'Depth' (Z float array)
    │   ├─> 'Vector' (motion XY float array)
    │   └─> 'Normal' (XYZ float array)
    │
    ├─> Pass to Mojo via ctypes
    │   └─> UnsafePointer[C_float] (zero-copy)
    │
    ├─> JEPA Processing (on GPU)
    │   ├─> Encode noisy patches
    │   ├─> Predict clean latent patches
    │   └─> Decode to RGB
    │
    └─> Return denoised buffer
        └─> Write back to render_result
```

### Flow 4: Emissive Light Detection

```
Scene Iteration
    │
    ├─> bpy.data.objects (light objects)
    │   └─> obj.data.type == 'LIGHT'
    │
    ├─> bpy.data.materials (emissive materials)
    │   └─> Traverse node_tree for emission nodes
    │
    └─> depsgraph.object_instances (GEOMETRY NODE INSTANCES)
        │
        ├─> instance.is_instance == True
        ├─> instance.matrix_world (instance transform)
        ├─> instance.persistent_id (unique instance ID)
        └─> instance.object (the instanced object)
            └─> Check its materials for emission
```

---

## Blender API Interface Map

### Critical bpy APIs Used

| API | Purpose | Data Returned |
|-----|---------|---------------|
| `context.evaluated_depsgraph_get()` | Get evaluated depsgraph | `Depsgraph` |
| `obj.evaluated_get(depsgraph)` | Get evaluated object | `Object` with modifiers applied |
| `BVHTree.FromObject(obj, depsgraph)` | Build spatial BVH | `BVHTree` with ray queries |
| `depsgraph.object_instances` | Iterate ALL instances | Generator of `DepsgraphObjectInstance` |
| `material.node_tree.nodes` | Access shader nodes | `Nodes` collection |
| `node.inputs['name'].default_value` | Read parameter values | `float` / `float4` |
| `node.inputs['name'].is_linked` | Check if socket connected | `bool` |
| `render_result.passes` | Access render passes | `RenderPass` collection |
| `pass.rect` | Get pixel data | Flat `float` array (W*H*channels) |

### Depsgraph Object Instance Properties

```python
for instance in depsgraph.object_instances:
    # Is this a geometry node instance (not the original object)?
    is_instance = instance.is_instance  # bool

    # Instance transformation in world space
    matrix = instance.matrix_world  # 4x4 Matrix

    # Unique identifier for this instance
    persistent_id = instance.persistent_id  # tuple of ints

    # Random ID for sampling variation
    random_id = instance.random_id  # int

    # The actual object being instanced
    obj = instance.object  # Object
```

### Render Pass Types

```python
# Enable passes before render
scene.view_layers[0].use_pass_combined = True
scene.view_layers[0].use_pass_z = True  # Depth
scene.view_layers[0].use_pass_vector = True  # Motion
scene.view_layers[0].use_pass_normal = True  # World-space normal

# Access after render
for pass_data in render_result.passes:
    name = pass_data.name  # 'Combined', 'Depth', 'Vector', 'Normal'
    channels = pass_data.channels  # 4 for RGBA, 1 for Depth, 2 for Vector, 3 for Normal
    rect = pass_data.rect  # Flat float array: width * height * channels
```

### Cycles Settings Control

```python
cscene = scene.cycles

# Adaptive sampling (CRITICAL for accelerator mode)
cscene.use_adaptive_sampling = True
cscene.adaptive_threshold = 0.01  # Lower = more aggressive stopping
cscene.adaptive_min_samples = 0  # Minimum samples before adaptation

# Sample count
cscene.samples = 1024  # Maximum samples
cscene.preview_samples = 32  # 3D viewport samples

# Resolution multipliers
scene.render.resolution_percentage = 50  # Render at 50% resolution

# Tile size (GPU)
cscene.tile_size = 2048  # Pixels per tile

# Denoiser (we disable this, Omen replaces it)
cscene.use_denoising = False
```

---

## Mojo ↔ C/C++ FFI Bridge

### C ABI Interface Definition

```c
// omen_core.h — C interface for Mojo library

#ifdef __cplusplus
extern "C" {
#endif

// Mode constants
#define OMEN_MODE_DENOISER      0
#define OMEN_MODE_ACCELERATOR   1
#define OMEN_MODE_MULTIRES      2

// Training mode constants
#define OMEN_TRAIN_OFF          0
#define OMEN_TRAIN_INCREMENTAL  1

// Opaque handles (Mojo structs passed as pointers)
typedef struct OmenContext OmenContext;
typedef struct SceneGraph SceneGraph;
typedef struct RenderObservation RenderObservation;

/**
 * Create Omen context
 * @param gpu_device_id CUDA/HIP device index (0 = first GPU)
 * @return Opaque context handle
 */
OmenContext* omen_context_create(int gpu_device_id);

/**
 * Destroy Omen context and release GPU memory
 */
void omen_context_destroy(OmenContext* ctx);

/**
 * Set scene graph from extracted Blender data
 * @param ctx Omen context
 * @param scene_data Serialized scene graph (JSON/binary)
 * @param data_size Size in bytes
 */
void omen_set_scene_graph(
    OmenContext* ctx,
    const void* scene_data,
    size_t data_size
);

/**
 * Process render observation (denoise or accelerate)
 * @param ctx Omen context
 * @param obs Render observation (noisy image + auxiliary passes)
 * @param output_denoised Pre-allocated output buffer (RGBA float)
 * @param output_confidence Pre-allocated confidence buffer (float)
 * @param mode Operation mode (DENOISER/ACCELERATOR/MULTIRES)
 * @param training_mode Training mode (OFF/INCREMENTAL)
 * @param ground_truth Optional ground truth for training (NULL if none)
 * @param gpu_device_id GPU to use
 * @return 0 on success, error code on failure
 */
int omen_process(
    OmenContext* ctx,
    const RenderObservation* obs,
    float* output_denoised,
    float* output_confidence,
    int mode,
    int training_mode,
    const float* ground_truth,
    int gpu_device_id
);

/**
 * Get JEPA model statistics
 * @param ctx Omen context
 * @param param_count Output parameter count
 * @param memory_mb Output memory usage in MB
 */
void omen_get_stats(
    const OmenContext* ctx,
    int* param_count,
    int* memory_mb
);

#ifdef __cplusplus
}
#endif
```

### Mojo Side Implementation

```mojo
# omen_core.mojo

from memory import UnsafePointer
from cffi import c_int, c_float, c_void_p

@always_inline
@register_function
fn omen_context_create(gpu_device_id: c_int) -> UnsafePointer[C_void]:
    """Create context - called from Python via ctypes"""
    var ctx = OmenContext(gpu_device_id)
    return UnsafePointer[C_void].bitcast(addressof(ctx))

@always_inline
@register_function
fn omen_process(
    ctx_ptr: UnsafePointer[C_void],
    obs_ptr: UnsafePointer[RenderObservation],
    output_ptr: UnsafePointer[C_float],
    confidence_ptr: UnsafePointer[C_float],
    mode: c_int,
    training_mode: c_int,
    ground_truth_ptr: UnsafePointer[C_float],
    gpu_device_id: c_int,
) -> c_int:
    """Process render - called from Python via ctypes"""
    # Reconstruct context from pointer
    let ctx = deref_unsafe_alias[OmenContext](ctx_ptr.bitcast())

    # Wrap input pointer as Tensor (zero-copy)
    let obs = deref_unsafe_alias[RenderObservation](obs_ptr)
    let noisy = Tensor.from_raw_ptr(obs.noisy_data, (obs.width, obs.height, 4))

    # Process with JEPA
    let (denoised, confidence) = ctx.jepa_model.process(noisy, mode)

    # Copy results to output pointers (or zero-copy if same GPU)
    memcpy(output_ptr, denoised.data.ptr, denoised.num_elements() * sizeof[C_float]())
    memcpy(confidence_ptr, confidence.data.ptr, confidence.num_elements() * sizeof[C_float]())

    return 0  # Success
```

### Python Side Binding

```python
# ffi_bridge.py

import ctypes
import numpy as np
from pathlib import Path

# Load Mojo shared library
_lib_path = Path(__file__).parent / "lib" / "libomen_core.so"
_omen = ctypes.CDLL(str(_lib_path))

# Define C structures
class RenderObservation(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("noisy_data", ctypes.POINTER(ctypes.c_float)),
        ("depth_data", ctypes.POINTER(ctypes.c_float)),
        ("vector_data", ctypes.POINTER(ctypes.c_float)),
        ("normal_data", ctypes.POINTER(ctypes.c_float)),
    ]

# Configure function signatures
_omen.omen_context_create.argtypes = [ctypes.c_int]
_omen.omen_context_create.restype = ctypes.c_void_p

_omen.omen_process.argtypes = [
    ctypes.c_void_p,  # ctx
    ctypes.POINTER(RenderObservation),  # obs
    ctypes.POINTER(ctypes.c_float),  # output_denoised
    ctypes.POINTER(ctypes.c_float),  # output_confidence
    ctypes.c_int,  # mode
    ctypes.c_int,  # training_mode
    ctypes.POINTER(ctypes.c_float),  # ground_truth
    ctypes.c_int,  # gpu_device_id
]
_omen.omen_process.restype = ctypes.c_int

def process_render(noisy_image, depth, vector, normal, mode=0):
    """Process render through JEPA"""
    height, width = noisy_image.shape[:2]

    # Create observation struct
    obs = RenderObservation()
    obs.width = width
    obs.height = height
    obs.noisy_data = noisy_image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    obs.depth_data = depth.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    obs.vector_data = vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    obs.normal_data = normal.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Allocate output buffers
    output = np.zeros((height, width, 4), dtype=np.float32)
    confidence = np.zeros((height, width), dtype=np.float32)

    # Call Mojo
    ret = _omen.omen_process(
        _ctx,
        ctypes.byref(obs),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        confidence.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mode,
        0,  # training_mode
        None,  # ground_truth
        0,  # gpu_device_id
    )

    if ret != 0:
        raise RuntimeError(f"Omen processing failed with code {ret}")

    return output, confidence
```

### Zero-Copy GPU Buffer Passing

```mojo
# Zero-copy wrapper for existing CUDA pointers (from Blender or other renderers)

from buffer import DeviceBuffer

fn wrap_cuda_pointer[T: DType](
    ptr: UnsafePointer[C_void],
    size: Int,
) -> DeviceBuffer[T]:
    """Wrap existing CUDA pointer without copying

    CRITICAL: Blender may allocate CUDA buffers directly. We can wrap them
    in Mojo's DeviceBuffer without memory duplication using owning=False.
    """
    return DeviceBuffer[T](
        UnsafePointer[T].bitcast(ptr),
        size,
        owning=False,  # Don't free when dropped!
    )
```

---

## Emissive Material Detection

### Node Tree Traversal Algorithm

```python
def is_material_emissive(material: bpy.types.Material) -> bool:
    """Check if material has any emission shader nodes"""
    if not material.use_nodes:
        return False

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Check for ShaderNodeEmission
    for node in nodes:
        if isinstance(node, bpy.types.ShaderNodeEmission):
            # Check if emission is connected to output
            if is_connected_to_output(node, 'Emission', links):
                strength = get_emission_strength(node)
                if strength > 0.0:
                    return True

    # Check Principled BSDF emission inputs
    for node in nodes:
        if isinstance(node, bpy.types.ShaderNodeBsdfPrincipled):
            # Check emission color input
            emission_color_socket = node.inputs.get('Emission Color')
            if emission_color_socket and emission_color_socket.is_linked:
                return True

            # Check emission strength input
            emission_strength_socket = node.inputs.get('Emission Strength')
            if emission_strength_socket:
                if emission_strength_socket.is_linked:
                    return True
                elif emission_strength_socket.default_value > 0.0:
                    return True

    return False


def is_connected_to_output(node, socket_name, links) -> bool:
    """BFS traversal to check if node connects to material output"""
    from collections import deque

    visited = set()
    queue = deque([(node, socket_name)])

    while queue:
        current_node, current_socket = queue.popleft()

        if (current_node, current_socket) in visited:
            continue
        visited.add((current_node, current_socket))

        # Check if this is the material output
        if isinstance(current_node, bpy.types.ShaderNodeOutputMaterial):
            return True

        # Follow links from this socket
        for link in links:
            if link.from_node == current_node and link.from_socket.name == current_socket:
                queue.append((link.to_node, link.to_socket.name))

    return False


def get_emission_strength(node) -> float:
    """Extract emission strength from node"""
    if isinstance(node, bpy.types.ShaderNodeEmission):
        strength_socket = node.inputs.get('Strength')
        if strength_socket:
            if strength_socket.is_linked:
                # Linked value — need to traverse upstream
                return get_linked_value(strength_socket)
            return strength_socket.default_value

    elif isinstance(node, bpy.types.ShaderNodeBsdfPrincipled):
        strength_socket = node.inputs.get('Emission Strength')
        if strength_socket:
            if strength_socket.is_linked:
                return get_linked_value(strength_socket)
            return strength_socket.default_value

    return 0.0


def get_linked_value(socket) -> float:
    """Follow socket link to get actual value (handles mixing, math nodes, etc.)"""
    if not socket.is_linked:
        return socket.default_value

    link = socket.links[0]
    from_node = link.from_node
    from_socket = link.from_socket

    # Handle different node types
    if isinstance(from_node, bpy.types.ShaderNodeMix):
        # Mix node — check factor and mixed values
        factor = get_linked_value(from_node.inputs['Factor'])
        a = get_linked_value(from_socket)
        b = get_linked_value(from_node.inputs[1 if from_socket == from_node.outputs[0] else 0])
        return factor * b + (1.0 - factor) * a

    elif isinstance(from_node, bpy.types.ShaderNodeMath):
        # Math node — compute operation
        a = get_linked_value(from_node.inputs[0])
        b = get_linked_value(from_node.inputs[1]) if len(from_node.inputs) > 1 else 0.0

        op = from_node.operation
        if op == 'ADD':
            return a + b
        elif op == 'SUBTRACT':
            return a - b
        elif op == 'MULTIPLY':
            return a * b
        elif op == 'DIVIDE':
            return a / b if b != 0 else 0.0
        # ... etc

    # Default: return the socket's default value
    return from_socket.default_value if hasattr(from_socket, 'default_value') else 0.0
```

### Cycles Internal Emission Handling (Reference)

```cpp
// From Cycles src/scene/shader.cpp
// This is how Cycles internally detects emission

float Shader::estimate_emission()
{
    if (!has_surface_emission)
        return 0.0f;

    float emission_estimate = 0.0f;

    // Walk shader graph to estimate emission
    for (auto& node : graph->nodes) {
        if (node->type == EmissionNode::get_node_type()) {
            EmissionNode* emission = static_cast<EmissionNode*>(node.get());
            float strength = emission->get_emission_strength();
            if (!emission->get_emission_is_constant()) {
                // Emission is texture-based — need to evaluate
                emission_estimate = max(emission_estimate, strength * 10.0f);
            } else {
                emission_estimate = max(emission_estimate, strength);
            }
        }
    }

    return emission_estimate;
}
```

---

## Geometry Node Instance Lights

### Why This Matters

Geometry nodes can instantiate thousands of objects. Each instance can have:
- Different transformation (matrix_world)
- Different materials (via material index override)
- Different visibility settings

**CRITICAL**: The "master copy" object in the scene is NOT what gets rendered. The instances at their transforms are what Cycles actually sees. We must detect these.

### Instance Detection Algorithm

```python
def extract_geometry_node_lights(context) -> list[dict]:
    """Extract all geometry node instances that emit light"""
    depsgraph = context.evaluated_depsgraph_get()
    emissive_instances = []

    for instance in depsgraph.object_instances:
        # Skip non-instances (the original objects)
        if not instance.is_instance:
            continue

        obj = instance.object

        # Check if any material slot has emission
        has_emission = False
        emission_info = {}

        for mat_slot in obj.material_slots:
            if mat_slot.material and is_material_emissive(mat_slot.material):
                has_emission = True
                emission_info = {
                    'material': mat_slot.material.name,
                    'emission_strength': get_emission_strength(mat_slot.material),
                }
                break

        if not has_emission:
            continue

        # Extract instance data
        instance_data = {
            'type': 'GEOMETRY_NODE_INSTANCE_LIGHT',
            'object_name': obj.name,
            'persistent_id': instance.persistent_id,  # Unique ID tuple
            'random_id': instance.random_id,  # For sampling variation
            'matrix_world': instance.matrix_world.copy(),  # 4x4 transform
            'is_instance': True,
            'parent': instance.parent.name if instance.parent else None,
        }
        instance_data.update(emission_info)

        emissive_instances.append(instance_data)

    return emissive_instances


def instance_key(instance) -> tuple:
    """Create a hashable key for caching instance data"""
    return (
        instance.object.name,
        instance.persistent_id,  # Already a tuple
        instance.random_id,
    )


def build_instance_light_tree(emissive_instances: list[dict]) -> dict:
    """Build acceleration structure for instance light queries

    For large scenes with thousands of instances, we need efficient spatial queries.
    We can use Blender's BVHTree or build our own.
    """
    if not emissive_instances:
        return {'bvh': None, 'instances': []}

    # Collect instance positions
    positions = []
    for inst in emissive_instances:
        # Extract translation from matrix_world
        pos = inst['matrix_world'].translation
        positions.append(pos)

    # Build BVH for spatial queries
    # Note: Blender's BVHTree doesn't natively support point queries,
    # so we create a temporary mesh with points at instance locations
    import bmesh

    bm = bmesh.new()
    for pos in positions:
        bm.verts.new(pos)

    bvh = BVHTree.FromBMesh(bm)

    return {
        'bvh': bvh,
        'instances': emissive_instances,
        'bmesh': bm,  # Keep to prevent GC
    }


def query_nearby_lights(light_tree: dict, position: Vector, radius: float) -> list[dict]:
    """Find emissive instances within radius of position"""
    if not light_tree['bvh']:
        return []

    # Find vertices within radius
    bvh = light_tree['bvh']
    nearest = bvh.find_nearest(position, radius)

    if not nearest:
        return []

    # Map back to instances
    nearby = []
    for idx, dist in nearest:
        nearby.append(light_tree['instances'][idx])

    return nearby
```

### Persistent ID and Random ID

```python
# DepsgraphObjectInstance provides two critical IDs:

# 1. persistent_id: Hierarchical ID for the instance
#    Unique across the entire render, stable across frames
#    Used for: caching, temporal consistency, JEPA prediction
persistent_id = instance.persistent_id
# Example: (3, 0, 5, 2, 1, 0, 0)

# 2. random_id: Random integer for sampling variation
#    Different for each instance, even if same geometry
#    Used for: texture coordinate variation, sampling jitter
random_id = instance.random_id
# Example: 123456789

# Usage in JEPA:
def encode_instance_id(instance) -> np.ndarray:
    """Encode instance ID for JEPA input"""
    # Use persistent_id for temporal tracking
    pid = np.array(instance.persistent_id, dtype=np.float32)

    # Use random_id as a feature
    rid = np.array([instance.random_id], dtype=np.float32)

    return np.concatenate([pid, rid])
```

---

## Large Cache Management

### Problem Statement

Blender scenes can contain:
- 8K+ textures (each 4K RGBA = 64MB)
- 10M+ triangle meshes (each vertex = 12 bytes + normals + UVs)
- 1000+ materials with complex node trees

Loading everything into memory crashes the GPU.

### Solution: Streaming + LRU Cache

#### Architecture

```mojo
struct TextureCache:
    var cache: LRUCache[String, TextureTile]
    var max_memory_mb: Int
    var current_memory_mb: Int
    var tile_size: Int  # e.g., 512x512 tiles

    fn get_tile(inout self, texture_name: String, tile_u: Int, tile_v: Int) -> TextureTile:
        """Get texture tile, loading from disk if necessary"""
        let key = texture_name + "_" + str(tile_u) + "_" + str(tile_v)

        if self.cache.contains(key):
            return self.cache[key]

        # Need to load
        let tile = self.load_tile_from_disk(texture_name, tile_u, tile_v)
        let tile_memory = tile.data.num_elements() * sizeof[float32]()

        # Evict if over budget
        while self.current_memory_mb + tile_memory > self.max_memory_mb:
            self.evict_lru()

        self.cache[key] = tile
        self.current_memory_mb += tile_memory
        return tile

    fn evict_lru(inout self):
        """Evict least-recently-used tile"""
        let (key, _) = self.cache.pop_lru()
        self.current_memory_mb -= self.estimate_tile_size(key)


struct GeometryCache:
    var streaming_meshes: Dict[ObjectID, StreamingMesh]
    var max_vertices: Int
    var loaded_vertices: Int

    fn load_mesh_window(inout self, obj_id: ObjectID, center: Vec3, radius: Float):
        """Load mesh geometry within radius of center point"""
        let mesh = self.streaming_meshes[obj_id]

        # Determine which BVH nodes are within radius
        let nodes = mesh.bvh.query_radius(center, radius)

        for node in nodes:
            if not node.is_loaded:
                # Load vertex data for this BVH node
                let vertex_data = self.load_bvh_node(obj_id, node.index)
                mesh.load_node(node.index, vertex_data)
                self.loaded_vertices += node.vertex_count

    fn unload_distant(inout self, camera_pos: Vec3, view_distance: Float):
        """Unload geometry too far from camera"""
        for (obj_id, mesh) in self.streaming_meshes:
            for node in mesh.bvh.nodes:
                if node.is_loaded:
                    dist = distance(node.center, camera_pos)
                    if dist > view_distance * 2.0:
                        mesh.unload_node(node.index)
                        self.loaded_vertices -= node.vertex_count


struct MaterialCache:
    var compiled_shaders: Dict[MaterialID, CompiledShader]
    var parameter_cache: Dict[MaterialID, MaterialParameters]

    fn get_parameters(inout self, material: bpy.types.Material) -> MaterialParameters:
        """Get cached material parameters"""
        mat_id = material.as_pointer()

        if self.parameter_cache.contains(mat_id):
            return self.parameter_cache[mat_id]

        # Extract all parameters
        let params = self.extract_parameters(material)
        self.parameter_cache[mat_id] = params
        return params
```

#### Python-Mojo Coordination

```python
# texture_streamer.py

class TextureStreamer:
    """Manages texture streaming between Blender and Mojo"""

    def __init__(self, omen_ctx, max_cache_mb=2048):
        self.omen_ctx = omen_ctx
        self.max_cache_mb = max_cache_mb
        self.loaded_tiles = {}  # (image_name, u, v) -> GPU pointer

    def request_tile(self, image_name: str, u: int, v: int, size: int = 512):
        """Request texture tile from Mojo (load if not cached)"""
        key = (image_name, u, v)

        if key in self.loaded_tiles:
            return self.loaded_tiles[key]

        # Load tile from Blender image
        image = bpy.data.images[image_name]
        tile_pixels = self.extract_tile_pixels(image, u, v, size)

        # Upload to Mojo GPU memory
        gpu_ptr = self.omen.upload_texture_tile(
            image_name.encode(),
            u, v, size,
            tile_pixels.ctypes.data,
            tile_pixels.size
        )

        self.loaded_tiles[key] = gpu_ptr
        return gpu_ptr

    def extract_tile_pixels(self, image, u, v, size):
        """Extract tile pixels from Blender image"""
        # Calculate tile boundaries
        tile_u_pixels = u * size
        tile_v_pixels = v * size

        # Ensure within bounds
        width, height = image.size
        tile_u_pixels = min(tile_u_pixels, width - size)
        tile_v_pixels = min(tile_v_pixels, height - size)

        # Extract pixels (this loads into RAM)
        # For very large images, we might want to use image.pixels.foreach_get
        pixels = image.pixels[
            (tile_v_pixels * width + tile_u_pixels) * 4:
            (tile_v_pixels * width + tile_u_pixels + size) * 4
        ]

        return np.array(pixels, dtype=np.float32).reshape(size, size, 4)
```

#### Budget Management

```python
# memory_budget.py

def calculate_memory_budget(gpu_free_mb: int, safety_factor: float = 0.8) -> dict:
    """Calculate memory budget for each cache type"""
    usable_mb = gpu_free_mb * safety_factor

    # Heuristic allocation
    budget = {
        'geometry': usable_mb * 0.40,  # 40% for geometry
        'textures': usable_mb * 0.40,   # 40% for textures
        'materials': usable_mb * 0.10,  # 10% for materials
        'jepa_model': usable_mb * 0.10, # 10% for JEPA
    }

    return budget


def monitor_memory_usage(omen_ctx) -> dict:
    """Monitor current Mojo GPU memory usage"""
    stats = omen_get_memory_stats(omen_ctx)

    usage = {
        'geometry_mb': stats.geometry_allocated_mb,
        'textures_mb': stats.textures_allocated_mb,
        'materials_mb': stats.materials_allocated_mb,
        'jepa_mb': stats.jepa_allocated_mb,
        'total_mb': sum([
            stats.geometry_allocated_mb,
            stats.textures_allocated_mb,
            stats.materials_allocated_mb,
            stats.jepa_allocated_mb,
        ]),
    }

    return usage
```

---

## JEPA Patch-Based Tile Processing

### JEPA Architecture for Rendering

```
Noisy Render Tile (128x128)
    │
    ├─> Extract Patches (16x16 patches of 8x8 pixels)
    │   └─> Patch shape: [16, 16, 8, 8, 4] = [patch_y, patch_x, h, w, c]
    │
    ├─> Vision Encoder
    │   ├─> Per-patch embedding: [16, 16, D] where D = 512
    │   └─> Add positional encoding
    │
    ├─> Context Encoder (Transformer)
    │   ├─> Process all patches jointly
    │   └─> Output: latent context [16, 16, D]
    │
    ├─> Spatial Predictor (JEPA core)
    │   ├─> For each patch, predict neighbors' latent representations
    │   └─> Training: InfoNCE loss on predicted vs target latents
    │
    └─> Decoder
        ├─> Upsample patches to full resolution
        └─> Output: Denoised tile [128, 128, 4]
```

### Mojo Implementation

```mojo
from tensor import Tensor
from random import rand

struct JEPAPredictor:
    var context_encoder: TransformerEncoder
    var predictor: MaskedPredictor
    var patch_size: Int
    var num_patches: Int
    var latent_dim: Int

    fn __init__(inout self, patch_size: Int = 8, latent_dim: Int = 512):
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.num_patches = 128 / patch_size  # 16 for 128x128 tile

        self.context_encoder = TransformerEncoder(
            num_layers=6,
            num_heads=8,
            hidden_dim=latent_dim,
        )

        self.predictor = MaskedPredictor(
            num_patches=self.num_patches * self.num_patches,
            latent_dim=latent_dim,
        )

    fn extract_patches(
        inout self,
        image: Tensor[DType.float32, 3],  # [H, W, C]
    ) -> Tensor[DType.float32, 5]:  # [patch_y, patch_x, ph, pw, c]
        """Extract non-overlapping patches from image"""
        let height = image.shape(0)
        let width = image.shape(1)
        let channels = image.shape(2)

        let num_patches_y = height / self.patch_size
        let num_patches_x = width / self.patch_size

        var patches = Tensor[DType.float32](
            num_patches_y, num_patches_x,
            self.patch_size, self.patch_size,
            channels
        )

        for py in range(num_patches_y):
            for px in range(num_patches_x):
                let h_start = py * self.patch_size
                let h_end = h_start + self.patch_size
                let w_start = px * self.patch_size
                let w_end = w_start + self.patch_size

                patches[py, px, :, :, :] = image[h_start:h_end, w_start:w_end, :]

        return patches

    fn encode_patches(
        inout self,
        patches: Tensor[DType.float32, 5],
    ) -> Tensor[DType.float32, 3]:  # [num_patches_y, num_patches_x, D]
        """Encode patches to latent space"""
        let num_patches_y = patches.shape(0)
        let num_patches_x = patches.shape(1)

        # Flatten each patch and encode
        var latents = Tensor[DType.float32](num_patches_y, num_patches_x, self.latent_dim)

        for py in range(num_patches_y):
            for px in range(num_patches_x):
                let patch = patches[py, px, :, :, :]  # [ph, pw, c]
                let flat = patch.reshape(self.patch_size * self.patch_size * 4)  # Assume RGBA

                # Simple linear projection (in practice: use proper CNN encoder)
                latents[py, px, :] = self.project_patch(flat)

        # Add positional encoding
        latents = self.add_positional_encoding(latents)

        return latents

    fn predict_denoised(
        inout self,
        noisy_latents: Tensor[DType.float32, 3],
        scene: SceneGraph,
    ) -> Tensor[DType.float32, 3]:
        """Predict clean latents from noisy latents using scene context"""
        # Encode context (scene-aware)
        let context = self.context_encoder.forward(noisy_latents, scene)

        # Predict clean latents (JEPA: predict in latent space, not pixel space)
        let clean_latents = self.predictor.predict(context)

        return clean_latents

    fn decode_to_image(
        inout self,
        latents: Tensor[DType.float32, 3],
    ) -> Tensor[DType.float32, 3]:  # [H, W, C]
        """Decode latents back to image"""
        let num_patches_y = latents.shape(0)
        let num_patches_x = latents.shape(1)

        var image = Tensor[DType.float32](
            num_patches_y * self.patch_size,
            num_patches_x * self.patch_size,
            4  # RGBA
        )

        for py in range(num_patches_y):
            for px in range(num_patches_x):
                let latent = latents[py, px, :]
                let patch = self.decode_patch(latent)
                image[
                    py * self.patch_size : (py + 1) * self.patch_size,
                    px * self.patch_size : (px + 1) * self.patch_size,
                    :
                ] = patch

        return image


struct MaskedPredictor:
    """JEPA predictor: predict masked patch latents from context"""
    var projection: Linear
    var num_patches: Int
    var latent_dim: Int

    fn predict(
        inout self,
        context: Tensor[DType.float32, 3],
    ) -> Tensor[DType.float32, 3]:
        """Predict all patches from context (self-supervised)"""
        # In training: mask random patches and predict them
        # In inference: use full context to refine predictions

        var predictions = Tensor[DType.float32](context.shape())

        for i in range(self.num_patches):
            let patch_latent = context[i]

            # Predict this patch from its neighbors
            let neighbor_context = self.get_neighbor_context(context, i)
            let prediction = self.projection.forward(neighbor_context)

            predictions[i, :] = prediction

        return predictions

    fn get_neighbor_context(
        inout self,
        context: Tensor[DType.float32, 3],
        center_idx: Int,
    ) -> Tensor[DType.float32, 1]:
        """Extract context from neighboring patches"""
        # 3x3 neighborhood around center patch
        let num_patches_x = context.shape(1)
        let center_y = center_idx / num_patches_x
        let center_x = center_idx % num_patches_x

        var neighbor_features = Tensor[DType.float32](self.latent_dim * 9)  # 3x3 neighbors
        var count = 0

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                let ny = center_y + dy
                let nx = center_x + dx

                if 0 <= ny < context.shape(0) and 0 <= nx < num_patches_x:
                    neighbor_features[count * self.latent_dim : (count + 1) * self.latent_dim] = context[ny, nx, :]
                    count += 1

        return neighbor_features
```

### Training Protocol (Self-Training)

```python
# training_protocol.py

class SelfTrainingProtocol:
    """Self-train JEPA on scene during rendering"""

    def __init__(self, omen_ctx):
        self.omen_ctx = omen_ctx
        self.training_data = []

    def on_render_complete(self, scene, noisy_render, spp):
        """Generate training pair from render"""
        # Noisy render at current spp
        noisy = self.extract_render_passes(noisy_render)

        # Ground truth: re-render at higher spp (e.g., 4x)
        cscene = scene.cycles
        original_samples = cscene.samples

        # Render at high spp for ground truth
        cscene.samples = original_samples * 4
        gt_render = bpy.ops.render.render(write_still=False)
        ground_truth = self.extract_render_passes(gt_render)

        # Restore original sample count
        cscene.samples = original_samples

        # Add to training set
        self.training_data.append({
            'noisy': noisy,
            'ground_truth': ground_truth,
            'spp': spp,
        })

        # Trigger training step
        if len(self.training_data) >= 8:  # Batch size
            self.train_step()

    def train_step(self):
        """Perform one training iteration"""
        batch = self.training_data[-8:]  # Last 8 frames

        # Send batch to Mojo for training
        for sample in batch:
            self.omen.train_incremental(
                noisy_ptr=sample['noisy'].ctypes.data,
                gt_ptr=sample['ground_truth'].ctypes.data,
                learning_rate=1e-4,
            )

        # Clear old data
        self.training_data = []
```

---

## File Structure

```
omen/
├── ARCHITECTURE.md           # This document
├── README.md
├── pyproject.toml
├── main.py                   # Blender addon entry point
│
├── python/                   # Python addon code
│   ├── __init__.py
│   ├── scene_extractor.py    # BVH, geometry extraction
│   ├── material_reader.py    # Material parameter extraction
│   ├── light_reader.py       # Light source detection
│   ├── render_handler.py     # Render lifecycle hooks
│   ├── texture_streamer.py   # Large texture handling
│   ├── memory_budget.py      # GPU memory management
│   ├── training_protocol.py  # Self-training logic
│   └── ffi_bridge.py         # Ctypes bindings to Mojo
│
├── mojo/                     # Mojo core engine
│   ├── __init__.mojo
│   ├── omen_core.mojo        # Main entry point (C ABI)
│   ├── jepa_model.mojo       # JEPA architecture
│   ├── scene_graph.mojo      # Scene graph representation
│   ├── vision_encoder.mojo   # Patch encoder
│   ├── predictor.mojo        # JEPA predictor
│   ├── memory_manager.mojo   # GPU memory management
│   ├── texture_cache.mojo    # Texture streaming
│   └── training.mojo          # Training loop
│
├── c/                        # C headers and utilities
│   ├── omen_core.h           # C ABI header
│   └── build.sh              # Build Mojo to .so
│
├── lib/                      # Built binaries (gitignored)
│   └── libomen_core.so       # Mojo shared library
│
└── tests/                    # Unit tests
    ├── test_emission.py
    ├── test_instances.py
    ├── test_ffi.py
    └── test_jepa.mojo
```

---

## References

### Blender API Documentation
- [Scene API](https://docs.blender.org/api/current/bpy.types.Scene.html)
- [Depsgraph API](https://docs.blender.org/api/current/bpy.types.Depsgraph.html)
- [RenderLayer/Pass API](https://docs.blender.org/api/current/bpy.types.RenderLayer.html)
- [Material/Node API](https://docs.blender.org/api/current/bpy.types.Material.html)
- [BVHTree API](https://docs.blender.org/api/current/bpy.types.BVHTree.html)

### Cycles Source Code (Reference)
- `src/scene/shader.cpp` — Emission detection
- `src/scene/object.cpp` — Mesh lights
- `blender/shader.cpp` — Material sync
- `blender/object.cpp` — Instance sync

### Mojo Documentation
- [Mojo Manual](https://docs.modular.com/mojo)
- [Mojo GPU](https://docs.modular.com/mojo/gpu)
- [Python Bindings](https://docs.modular.com/mojo/python)

### JEPA Research
- Facebook 3D-JEPA (Locate 3D, arxiv 2504.14151) — CC-BY-NC license
- Original JEPA (IJCV 2024)
