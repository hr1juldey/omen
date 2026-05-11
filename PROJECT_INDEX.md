# Cycles_mojo - Full Project Index & Relationship Map

## Project Overview
This is a fork/port of **Blender Cycles** renderer. Cycles is a physically-based, production-ready path tracing renderer with support for CPU and multiple GPU backends (CUDA, OptiX, HIP, Metal, oneAPI).

- **License**: Apache 2.0
- **Language**: C++ (528 .h, 213 .cpp), OSL shaders (104 .osl), CMake build (37 CMakeLists.txt)
- **Total source lines**: ~200K+ (excluding .git)

---

## Directory Structure

```
cycles_mojo/
├── CMakeLists.txt              # Root build config
├── GNUmakefile                 # Linux/macOS build helper
├── make.bat                    # Windows build helper
├── .clang-format               # Code formatting rules
├── .gitmodules                 # Git submodules
├── src/                        # Main source code
│   ├── app/                    # Standalone application entry
│   ├── bvh/                    # Bounding Volume Hierarchy
│   ├── device/                 # Device abstractions (CPU/GPU)
│   ├── graph/                  # Shader node graph
│   ├── integrator/             # Path tracing integrator
│   ├── kernel/                 # Rendering kernel (runs on device)
│   ├── scene/                  # Scene representation & management
│   ├── session/                # Render session management
│   ├── subd/                   # Subdivision surfaces
│   ├── test/                   # Unit tests
│   ├── util/                   # Utility library
│   └── cmake/                  # CMake find modules
├── third_party/                # Bundled dependencies
├── lib/                        # Pre-compiled kernel binaries
├── tools/                      # Development scripts
├── web/                        # Hugo website
├── examples/                   # XML scene examples
└── cycles_mojo/                # Submodule clone
```

---

## Module Relationship Map

### Dependency Graph (high-level)

```
app ──→ session ──→ integrator ──→ device ──→ kernel
  │         │           │            │
  │         │           │            ├──→ util
  │         │           │            └──→ scene
  │         │           │                 │
  │         │           │                 ├──→ graph ──→ util
  │         │           │                 ├──→ bvh   ──→ util
  │         │           │                 └──→ subd  ──→ util
  │         │           │
  │         │           └──→ util
  │         │
  │         └──→ scene
  │
  └──→ device, scene, session
```

### Cross-Module Dependencies (from #include analysis)

| Module    | Depends On |
|-----------|-----------|
| **device** | cpu, cuda, optix, hip, hiprt, metal, oneapi, dummy, denoise, graphics_interop, buffers, bvh2, geometry, util |
| **kernel** | bvh, camera, closure, film, geom, integrator, sample, svm, util, bake, osl, device |
| **scene**  | bvh, graph, integrator, session(buffers), util, subd, device |
| **integrator** | device, denoise, session(buffers/display), util |
| **session** | device, scene, integrator, util |
| **bvh**    | util, scene(mesh/object/curves/hair/octree) |
| **subd**   | util, scene(mesh/attribute/camera) |
| **graph**  | util (self-contained node graph system) |
| **util**   | (base layer - no internal cross-module deps) |
| **app**    | device, scene, session, util |

---

## Detailed File Index by Module

### 1. src/device/ — Device Abstraction Layer

**Purpose**: Abstract interface for compute devices (CPU, CUDA, OptiX, HIP, Metal, oneAPI).

| File | Purpose |
|------|---------|
| `device.h/cpp` | Base `Device` class interface, factory methods |
| `device_queue.h/cpp` | Base `DeviceQueue` for async kernel execution |
| `device_memory.h/cpp` | `device_ptr`, `DeviceBuffer`, memory management |
| `device_graphics_interop.h/cpp` | Graphics interop (OpenGL/Vulkan) |
| `cpu/device.h` | CPU device interface |
| `cpu/kernel.h/cpp` | CPU kernel thread management |
| `cpu/kernel_thread_globals.h` | Per-thread kernel state |
| `cuda/device_impl.h/cpp` | CUDA device implementation |
| `cuda/device_impl.cpp` | CUDA kernel compile/load, memory ops |
| `cuda/queue.h/cpp` | CUDA stream-based queue |
| `cuda/util.h` | CUDA context scope, error macros |
| `optix/device_impl.h/cpp` | OptiX ray-tracing device (NVidia RTX) |
| `optix/device_impl.cpp` | OptiX pipeline/SBT setup, BVH build |
| `optix/queue.h/cpp` | OptiX launch queue |
| `optix/util.h` | OptiX utility helpers |
| `hip/device_impl.h/cpp` | HIP device (AMD ROCm) |
| `hip/device_impl.cpp` | HIP kernel compile, memory management |
| `hip/queue.h/cpp` | HIP stream queue |
| `hip/util.h` | HIP context helpers |
| `hiprt/device_impl.h/cpp` | HIP RT (AMD ray tracing) |
| `hiprt/device_impl.cpp` | HIP RT BVH build, pipeline setup |
| `hiprt/queue.h/cpp` | HIP RT queue |
| `hiprt/util.h` | HIP RT utilities |
| `metal/device_impl.mm/h` | Metal device (Apple GPU) |
| `metal/device_impl.mm` | Metal pipeline compilation |
| `metal/bvh.mm/h` | Metal BVH acceleration structure |
| `metal/queue.mm/h` | Metal command buffer queue |
| `metal/util.mm/h` | Metal utility functions |
| `oneapi/device_impl.h/cpp` | oneAPI/SYCL device (Intel GPU) |
| `oneapi/device_impl.cpp` | oneAPI kernel compilation, execution |
| `oneapi/queue.h/cpp` | oneAPI queue |
| `dummy/device.h/cpp` | Null device for testing |
| `denoise.h/cpp` | Generic denoiser interface |
| `denoiser_gpu.h/cpp` | GPU-accelerated denoising |
| `denoiser_oidn.h/cpp` | Intel OIDN denoiser |
| `denoiser_oidn_gpu.h/cpp` | OIDN GPU denoiser |

**Key Relations**:
- `device.h` → `device_queue.h`, `device_memory.h`, `denoise.h`
- Each GPU backend inherits `Device` and `DeviceQueue`
- `cuda/` → `optix/` (OptiX extends CUDA)
- `hip/` → `hiprt/` (HIP RT extends HIP)
- All backends → `kernel/device/{backend}/globals.h` (kernel data access)

---

### 2. src/kernel/ — Rendering Kernel

**Purpose**: Core path tracing algorithms that run on the device (CPU/GPU).

#### kernel/device/ — Per-device kernel entry points

| File | Purpose |
|------|---------|
| `cpu/globals.h` | CPU kernel globals (KernelGlobals) |
| `cpu/kernel.h` | CPU kernel entry declarations |
| `gpu/kernel.h` | GPU kernel entry declarations |
| `cuda/globals.h` | CUDA-specific kernel globals |
| `optix/globals.h` | OptiX-specific kernel globals |
| `hip/globals.h` | HIP-specific kernel globals |
| `hiprt/globals.h` | HIP RT kernel globals |
| `metal/globals.h` | Metal-specific kernel globals |
| `oneapi/globals.h` | oneAPI-specific kernel globals |

#### kernel/bvh/ — BVH Traversal

| File | Purpose |
|------|---------|
| `bvh.h` | BVH intersection entry |
| `bvh_nodes.h` | BVH node traversal (aligned/unaligned) |
| `bvh_traversal.h` | Ray-AABB intersection |
| `bvh_volume.h` | Volume intersection |
| `bvh_shadow.h` | Shadow ray intersection |
| `bvh_local.h` | Local (subsurface) intersection |
| `bvh_curve.h` | Curve intersection |
| `bvh_curve_all.h` | All curve types |
| `bvh_point.h` | Point cloud intersection |
| `bvh_aligned_node.h` | Aligned BVH node intersect |
| `bvh_unaligned_node.h` | Unaligned BVH node intersect |
| `bvh_quantized_node.h` | Quantized BVH nodes |

#### kernel/camera/ — Camera

| File | Purpose |
|------|---------|
| `camera.h/cpp` | Camera ray generation |
| `projection.h` | Projection utilities |

#### kernel/closure/ — BSDF/BSSRDF/Volume closures

| File | Purpose |
|------|---------|
| `alloc.h` | Closure allocation |
| `bsdf.h` | BSDF evaluation/sampling |
| `bsdf_diffuse.h` | Lambertian/Oren-Nayar diffuse |
| `bsdf_oren_nayar.h` | Oren-Nayar diffuse |
| `bsdf_diffuse_ramp.h` | Color-ramped diffuse |
| `bsdf_phong_ramp.h` | Color-ramped Phong |
| `bsdf_microfacet.h` | Microfacet GGX/Beckmann |
| `bsdf_microfacet_multi.h` | Multi-scatter microfacet |
| `bsdf_ashikhmin_shirley.h` | Ashikhmin-Shirley |
| `bsdf_ashikhmin_velvet.h` | Velvet/Sheen |
| `bsdf_sheen.h` | Sheen closure |
| `bsdf_hair.h` | Hair BSDF |
| `bsdf_principled_hair_huang.h` | Principled hair (Huang) |
| `bsdf_ray_portal.h` | Ray portal BSDF |
| `bsdf_toon.h` | Toon shading |
| `bsdf_transparent.h` | Transparent BSDF |
| `bsdf_ambient_occlusion.h` | AO closure |
| `bsdf_util.h` | BSDF utilities |
| `bssrdf.h` | BSSRDF subsurface scattering |
| `emissive.h` | Emissive closure |
| `volume.h` | Volume absorption/scattering |
| `volume_henyey_greenstein.h` | Henyey-Greenstein phase |
| `volume_fournier_forand.h` | Fournier-Forand phase |
| `volume_draine.h` | Draine phase function |
| `volume_rayleigh.h` | Rayleigh scattering |
| `volume_util.h` | Volume utilities |

#### kernel/film/ — Film/AOV

| File | Purpose |
|------|---------|
| `film.h` | Film data access |
| `aov_passes.h` | Arbitrary Output Variables |
| `cryptomatte_passes.h` | Cryptomatte ID passes |
| `light_passes.h` | Light pass accumulation |
| `adaptive_sampling.h` | Adaptive sampling convergence |

#### kernel/geom/ — Geometry

| File | Purpose |
|------|---------|
| `geom.h` | Geometry intersection entry |
| `object.h` | Object transforms, attributes |
| `curve.h` | Curve geometry |
| `triangle.h` | Triangle intersection |
| `triangle_intersect.h` | Moeller-Trumbore intersection |
| `triangle_primitive.h` | Triangle primitive data |
| `motion_triangle.h` | Motion blur triangles |
| `motion_triangle_intersect.h` | Motion triangle intersection |
| `motion_curve.h` | Motion blur curves |
| `point_intersect.h` | Point cloud intersection |
| `primitive.h` | Primitive accessors |
| `shader_data.h` | ShaderData struct |
| `volume.h` | Volume geometry |

#### kernel/integrator/ — Path Tracing Integrator

| File | Purpose |
|------|---------|
| `init_from_camera.h` | Primary ray from camera |
| `init_from_bake.h` | Bake ray initialization |
| `intersect_closest.h` | Closest hit intersection |
| `intersect_shadow.h` | Shadow ray intersection |
| `intersect_subsurface.h` | Subsurface intersection |
| `intersect_volume_stack.h` | Volume stack intersection |
| `intersect_dedicated_light.h` | Dedicated light intersection |
| `shade_background.h` | Background shading |
| `shade_light.h` | Light evaluation (NEE) |
| `shade_surface.h` | Surface shading |
| `shade_surface_raycast.h` | Surface ray-tracing (OptiX) |
| `shade_surface_mnee.h` | Manifold Next Event Estimation |
| `shade_volume.h` | Volume shading |
| `shade_shadow.h` | Shadow shading |
| `shade_dedicated_light.h` | Dedicated light shading |
| `state.h` | Integrator state structures |
| `state_template.h` | State template for GPU |
| `state_util.h` | State utility functions |
| `state_flow.h` | Shadow/path state flow |
| `path_state.h` | Path state (bounces, flags) |
| `shadow_state.h` | Shadow state |
| `shadow_state_template.h` | Shadow state template |
| `subsurface.h` | Subsurface dispatch |
| `subsurface_disk.h` | Disk subsurface |
| `subsurface_random_walk.h` | Random walk subsurface |
| `volume_stack.h` | Volume stack management |
| `volume_shader.h` | Volume shader evaluation |
| `surface_shader.h` | Surface shader evaluation |
| `displacement_shader.h` | Displacement evaluation |
| `guiding.h` | Path guiding (PG) |
| `volume_guiding_denoise.h` | Volume guiding + denoise |

#### kernel/sample/ — Sampling

| File | Purpose |
|------|---------|
| `sample.h` | Sample data access |
| `pattern.h` | Sample patterns (Sobol, CMJ, PMJ) |
| `sobol_burley.h` | Burley Sobol sampling |
| `tabulated_sobol.h` | Tabulated Sobol |
| `lcg.h` | Linear Congruential Generator |
| `literals.h` | Sampling utilities |
| `mapping.h` | Texture mapping |
| `mesh_light.h` | Mesh light sampling |
| `lightpick_table.h` | Light picking table |

#### kernel/svm/ — Shader Virtual Machine

| File | Purpose |
|------|---------|
| `svm.h` | SVM main entry |
| `svm_color_util.h` | SVM color ops |
| `math_util.h` | SVM math ops |
| `util.h` | SVM utilities |
| `node_types.h` | SVM node type definitions |
| `node_types_template.h` | Node type templates |
| `closure.h` | SVM → closure bridge |
| `tex.h` | SVM texture coordinate |
| `types.h` | SVM data types |
| `fractal_voronoi.h` | Fractal Voronoi noise |
| `voronoi.h` | Voronoi noise |
| `noise.h` | Noise functions |
| `wave.h` | Wave texture |
| `magic.h` | Magic texture |
| `checker.h` | Checker texture |
| `brick.h` | Brick texture |
| `white_noise.h` | White noise |
| `gradient.h` | Gradient texture |
| `image.h` | Image texture sampling |

#### kernel/light/ — Lighting

| File | Purpose |
|------|---------|
| `light.h` | Light evaluation/sampling |
| `background.h` | Background/Environment light |
| `light_sample.h` | Light sample utilities |
| `area.h` | Area light sampling |
| `spot.h` | Spot light sampling |
| `point.h` | Point light sampling |
| `distant.h` | Distant/Sun light sampling |
| `triangle.h` | Triangle (mesh) light |

#### kernel/util/ — Kernel Utilities

| File | Purpose |
|------|---------|
| `color.h` | Color conversion |
| `colorspace.h` | Color space management |
| `constants.h` | Physical constants |
| `differential.h` | Differentials |
| `inverse_sqrt.h` | Fast inverse sqrt |
| `lookup_table.h` | Lookup tables |
| `profiling.h` | Kernel profiling |

#### kernel/osl/ — Open Shading Language

| File | Purpose |
|------|---------|
| `closures.cpp` | OSL closure registration |
| `closures_setup.h` | Closure setup for rendering |
| `closures_template.h` | Closure templates |
| `globals.cpp/h` | OSL globals |
| `services.cpp/h` | OSL render services |
| `services_shared.h` | Shared service code |
| `services_gpu.h` | GPU OSL services |
| `compat.h` | OSL compatibility |
| `types.h` | OSL type definitions |
| `shaders/*.osl` | 104 OSL shader files |

**Key Relations within kernel/:
- `integrator/*.h` → `bvh/`, `camera/`, `closure/`, `film/`, `geom/`, `light/`, `sample/`, `svm/`, `util/`
- `svm/closure.h` → all `closure/bsdf_*.h`, `closure/bssrdf.h`, `closure/emissive.h`, `closure/volume.h`
- `svm/svm.h` → `svm/node_types.h` → all `svm/*_util.h`, `svm/*.h` (texture nodes)
- `kernel/device/*/globals.h` → `kernel/types.h` (shared KernelData struct)

---

### 3. src/scene/ — Scene Management

| File | Purpose |
|------|---------|
| `scene.h/cpp` | Main `Scene` class - owns all scene data |
| `camera.h/cpp` | Camera parameters, matrices |
| `film.h/cpp` | Film/display parameters, passes |
| `geometry.h/cpp` | Base `Geometry` class |
| `mesh.h/cpp` | Mesh geometry (triangles) |
| `hair.h/cpp` | Hair curve geometry |
| `pointcloud.h/cpp` | Point cloud geometry |
| `volume.h/cpp` | Volume geometry (OpenVDB) |
| `light.h/cpp` | Light objects |
| `object.h/cpp` | Object instances (geometry + transform) |
| `shader.h/cpp` | Shader (material) definitions |
| `shader_graph.h/cpp` | Shader node graph |
| `shader_nodes.h/cpp` | All shader node types |
| `shader_tables` | Precomputed BSDF albedo tables |
| `integrator.h/cpp` | Integrator settings |
| `background.h/cpp` | Background/Environment |
| `particles.h/cpp` | Particle data |
| `procedural.h/cpp` | Procedural geometry |
| `attribute.h/cpp` | Geometry attributes (UV, color, etc.) |
| `alembic.h/cpp` | Alembic cache reader |
| `image.h/cpp` | Image manager |
| `image_impl.h/cpp` | Image loader implementations |
| `image_metadata.h/cpp` | Image metadata reader |
| `image_oiio.h/cpp` | OIIO image loader |
| `image_vdb.h/cpp` | OpenVDB volume loader |
| `osl.h/cpp` | OSL script manager |
| `persistent_data.h` | Persistent data cache |
| `stats.h/cpp` | Scene statistics |
| `tables.h` | Precomputed tables |

**Key Relations**:
- `scene.h` → ALL other scene headers
- `shader_graph.h` → `graph/node.h`, `graph/node_type.h`
- `shader_nodes.h` → `shader_graph.h`, defines every shader node
- `mesh.h/cpp` → `bvh/build.h` (for BVH building)
- `volume.h/cpp` → `subd/` (subdivision), OpenVDB

---

### 4. src/integrator/ — Render Integrator

| File | Purpose |
|------|---------|
| `path_trace.h/cpp` | Main `PathTrace` class - orchestrates rendering |
| `path_trace_work.h/cpp` | Base `PathTraceWork` |
| `path_trace_work_gpu.h/cpp` | GPU work implementation |
| `path_trace_work_cpu.h/cpp` | CPU work implementation |
| `path_trace_display.h/cpp` | Display buffer management |
| `path_trace_tile.h/cpp` | Tile-based rendering |
| `render_scheduler.h/cpp` | Render scheduling (progressive, adaptive) |
| `denoiser.h/cpp` | Denoiser base |
| `denoiser_gpu.h/cpp` | GPU denoiser |
| `denoiser_oidn.h/cpp` | OIDN denoiser |
| `denoiser_oidn_gpu.h/cpp` | OIDN GPU denoiser |
| `adaptive_sampling.h/cpp` | Adaptive sampling logic |
| `work_balancer.h/cpp` | Work balance across GPUs |
| `work_tile_scheduler.h/cpp` | Tile scheduling |

**Key Relations**:
- `path_trace.h` → `session/session.h`, `device/device.h`
- `path_trace_work_gpu.h` → `device/device_queue.h`, `kernel/types.h`
- `render_scheduler.h` → `scene/integrator.h` (settings)

---

### 5. src/session/ — Render Session

| File | Purpose |
|------|---------|
| `session.h/cpp` | Main `Session` class - top-level render control |
| `buffers.h/cpp` | Render buffer management |
| `display_driver.h` | Display output interface |
| `output_driver.h` | File output interface |
| `tile.h/cpp` | Tile manager |
| `cache_eviction.h/cpp` | Texture cache eviction |
| `denoising.h/cpp` | Denoising framebuffers |

**Key Relations**:
- `session.h` → `device/device.h`, `scene/scene.h`, `integrator/render_scheduler.h`
- `buffers.h` → `device/device_memory.h`, `scene/film.h`

---

### 6. src/bvh/ — BVH Construction

| File | Purpose |
|------|---------|
| `bvh.h/cpp` | Base `BVH` class |
| `build.h/cpp` | BVH build algorithm |
| `node.h` | BVH node structures |
| `binning.h` | SAH binning |
| `params.h` | BVH parameters |
| `unaligned.h` | Unaligned node handling |
| `curve.h` | Curve BVH |
| `hair.h/cpp` | Hair BVH |
| `mesh.h/cpp` | Mesh BVH |
| `pointcloud.h/cpp` | Point cloud BVH |
| `multi.h/cpp` | Multi-BVH (instances) |
| `bvh2.h/cpp` | BVH2 implementation |
| `optiX/` | OptiX BVH (delegates to device) |
| `embree/` | Embree BVH (delegates) |
| `hiprt/` | HIP RT BVH |

**Key Relations**:
- `build.h` → `binning.h`, `node.h`
- `bvh.h` → `scene/geometry.h`, `scene/mesh.h`, `scene/hair.h`
- Device-specific BVH (optiX, embree, hiprt) → `device/{backend}/*`

---

### 7. src/subd/ — Subdivision Surfaces

| File | Purpose |
|------|---------|
| `subd_dice.h/cpp` | Dice (tessellate) subdivision |
| `subd_patch.h/cpp` | Subdivision patches |
| `subd_split.h/cpp` | Split patches |
| `subd_stats.h` | Subdivision statistics |
| `patch.h` | Generic patch evaluation |
| `split.h` | Patch splitting |
| `dice.h` | Patch dicing |
| `interpolation.h` | Interpolation utilities |
| `mesh.h` | Subd mesh adapter |

**Key Relations**:
- `subd_dice.h` → `subd_patch.h`, `subd_split.h`
- `subd_split.h` → `subd_patch.h`
- All → `scene/mesh.h`, `util/boundbox.h`

---

### 8. src/graph/ — Node Graph System

| File | Purpose |
|------|---------|
| `node.h/cpp` | Base `Node` class |
| `node_type.h/cpp` | `NodeType` definition |
| `node_enum.h` | Enum definitions |
| `node_set.h` | Set type |
| `socket.h/cpp` | Socket (input/output) |
| `socket_builder.h` | Socket builder pattern |
| `constants.h` | Graph constants |

**Key Relations**:
- `scene/shader_nodes.h` → `graph/node.h` (all shader nodes extend Node)
- `scene/shader_graph.h` → `graph/node.h`, `graph/socket.h`

---

### 9. src/util/ — Utility Library

| File | Purpose |
|------|---------|
| `algorithm.h` | General algorithms (min, max, clamp) |
| `aligned_malloc.h/cpp` | Aligned memory allocation |
| `args.h/cpp` | Command-line argument parsing |
| `array.h` | Dynamic array template |
| `atomic.h` | Atomic operations |
| `boundbox.h` | Bounding box |
| `color.h/cpp` | Color utilities |
| `colorspace.h/cpp` | Color space management |
| `debug.h/cpp` | Debug utilities |
| `defines.h` | Platform defines |
| `deque.h` | Deque (double-ended queue) |
| `dispatch.h` | Task dispatch |
| `filesystem.h/cpp` | File system operations |
| `function.h` | Function wrappers |
| `guarded_allocator.h/cpp` | Memory tracking allocator |
| `half.h/cpp` | Half-precision float |
| `hash.h` | Hash functions |
| `ies.h/cpp` | IES light profile |
| `image.h/cpp` | Image utilities |
| `image_impl.h/cpp` | Image implementation |
| `image_maketx.h/cpp` | Image texture maker |
| `image_metadata.h/cpp` | Image metadata |
| `log.h/cpp` | Logging |
| `map.h` | Ordered map |
| `math.h` | Math functions |
| `math_float2.h` | float2 operations |
| `math_float3.h` | float3 operations |
| `math_float4.h` | float4 operations |
| `math_float8.h` | float8 (AVX) operations |
| `math_int2.h` | int2 operations |
| `math_int3.h` | int3 operations |
| `math_int4.h` | int4 operations |
| `md5.h/cpp` | MD5 hash |
| `murmilge.h/cpp` | Murmilge hash |
| `openimagedenoise.h/cpp` | OIDN interface |
| `openimageio.h/cpp` | OIIO interface |
| `opengl.h/cpp` | OpenGL utilities |
| `parallel.h` | Parallel for |
| `param.h` | Parameter system |
| `path.h/cpp` | File path utilities |
| `progress.h/cpp` | Progress reporting |
| `projection.h` | Projection utilities |
| `queue.h` | Thread-safe queue |
| `rect.h` | Rectangle |
| `render.h` | Render utilities |
| `set.h` | Set container |
| `simd.h` | SIMD abstraction |
| `sse2neon.h` | SSE-to-NEON translation |
| `stack.h` | Stack container |
| `stats.h/cpp` | Statistics |
| `string.h/cpp` | String utilities |
| `system.h/cpp` | System info |
| `task.h/cpp` | Task scheduler |
| `texture.h/cpp` | Texture utilities |
| `thread.h/cpp` | Threading |
| `time.h/cpp` | Time utilities |
| `transform.h/cpp` | 3D transform |
| `types.h` | Type definitions |
| `types_base.h` | Base type traits |
| `types_float2.h` | float2 type |
| `types_float3.h` | float3 type |
| `types_float4.h` | float4 type |
| `types_float8.h` | float8 type |
| `types_int2.h` | int2 type |
| `types_int3.h` | int3 type |
| `types_int4.h` | int4 type |
| `types_uchar4.h` | uchar4 type |
| `types_spectrum.h` | Spectral color type |
| `types_unaligned.h` | Unaligned memory access |
| `vector.h` | Vector container |
| `version.h` | Version info |
| `windows.h/cpp` | Windows-specific |
| `xml.h/cpp` | XML parser |

**Key Relations**: util is the base layer - all other modules depend on it. No reverse dependencies.

---

### 10. src/app/ — Standalone Application

| File | Purpose |
|------|---------|
| `cycles_standalone.cpp` | Main entry point for standalone |
| `cycles_xml.cpp` | XML scene loader |
| `cycles_util.h` | App utilities |
| `opengl/display.h/cpp` | OpenGL display window |
| `opengl/window.h/cpp` | OpenGL window management |

**Key Relations**: `cycles_standalone.cpp` → `session/session.h`, `scene/scene.h`, `device/device.h`

---

### 11. src/test/ — Unit Tests

| File | Purpose |
|------|---------|
| `render_graph_finalize_test.cpp` | Shader graph finalization tests |
| `util_array_test.cpp` | Array tests |
| `util_bvh2_test.cpp` | BVH2 tests |
| `util_math_test.cpp` | Math tests |
| `util_string_test.cpp` | String tests |
| `util_task_test.cpp` | Task scheduler tests |
| `util_time_test.cpp` | Time tests |
| `util_transform_test.cpp` | Transform tests |
| `util_types_base_test.cpp` | Type tests |

---

### 12. third_party/ — Bundled Dependencies

| Directory | Purpose |
|-----------|---------|
| `sky/` | Hosek-Wilkie sky model |
| `cuew/` | CUDA Extension Wrapper |
| `hipew/` | HIP Extension Wrapper |
| `mikktspace/` | Mikkt tangent space |
| `atomic/` | Atomic operations |
| `libc_compat/` | C library compatibility |

---

### 13. tools/ — Development Scripts

| File | Purpose |
|------|---------|
| `sync_blender_commits.py` | Sync commits from Blender repo |
| `sync_git_am.py` | Git am with extra checks |

---

### 14. web/ — Hugo Website

| File | Purpose |
|------|---------|
| `config.toml` | Hugo config |
| `content/_index.md` | Homepage |
| `content/features.md` | Features page |
| `content/development.md` | Development page |
| `themes/custom/` | Custom Hugo theme |

---

### 15. examples/ — XML Scene Examples

| File | Purpose |
|------|---------|
| `scene_*.xml` | Various test scenes (sphere, cube, etc.) |
| `objects/*.xml` | Individual object definitions |
| `osl/*.osl` | Custom OSL shader examples |

---

### 16. lib/ — Pre-compiled Kernel Binaries

| Directory | Purpose |
|-----------|---------|
| `linux_x64/` | Pre-compiled CUDA/OptiX kernels for Linux |
| `windows_x64/` | Windows x64 kernels |
| `windows_arm64/` | Windows ARM64 kernels |
| `macos_x64/` | macOS Intel kernels |
| `macos_arm64/` | macOS Apple Silicon kernels |
| `legacy/` | Older kernel versions |

---

## Build System (CMake)

### Key CMake targets:
- `cycles_kernel` → All kernel source + device kernels
- `cycles_scene` → Scene management
- `cycles_device` → Device abstraction
- `cycles_integrator` → Integrator
- `cycles_session` → Session management
- `cycles_bvh` → BVH construction
- `cycles_subd` → Subdivision
- `cycles_graph` → Node graph
- `cycles_util` → Utility library
- `cycles_bin` → Standalone executable
- `cycles_kernel_osl` → OSL kernel
- `cycles_hydra` → Hydra render delegate

### CMake library dependencies:
```
cycles_bin → cycles_session, cycles_app
cycles_session → cycles_scene, cycles_integrator, cycles_device, cycles_bvh
cycles_integrator → cycles_device, cycles_scene
cycles_scene → cycles_graph, cycles_bvh, cycles_subd, cycles_device
cycles_device → cycles_kernel, cycles_util
cycles_kernel → cycles_kernel_osl (optional)
cycles_bvh → cycles_util
cycles_subd → cycles_util
cycles_graph → cycles_util
```

---

## Data Flow (Rendering Pipeline)

```
1. Session creates Device + Scene
2. Scene loads geometry, shaders, lights, camera
3. Shader Graph → SVM bytecode (shader_nodes.cpp)
4. Scene → BVH build (bvh/build.cpp)
5. Scene data → Device memory (device_memory.h)
6. PathTrace starts rendering (path_trace.cpp)
7. PathTraceWork dispatches kernels (path_trace_work_gpu.cpp)
8. Device queue launches kernels (device_queue.h)
9. Kernel executes on device:
   a. init_from_camera → generate primary rays
   b. intersect_closest → BVH traversal
   c. shade_surface → SVM evaluation → BSDF sampling
   d. shade_light → Light evaluation (NEE)
   e. shade_volume → Volume integration
   f. Repeat b-e for path continuation
   g. film/light_passes → accumulate samples
10. Render scheduler manages passes (render_scheduler.cpp)
11. Optional denoising (denoiser_gpu.cpp)
12. Display/Output driver receives result
```
