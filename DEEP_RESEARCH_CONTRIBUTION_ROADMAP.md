# Deep Research: Cycles Contribution Roadmap and Technical Feasibility Analysis

Research date: 2026-05-11
Sources: Cycles source code, Blender developer docs, SIGGRAPH papers, Mojo HPC paper (arXiv 2509.21039), Inria spectral rendering paper, Weta Manuka paper, NVIDIA wavefront path tracing paper

---

## Executive Summary

After deep analysis of Cycles source code, the GPU rendering landscape, and Mojo actual capabilities, here are the 4 viable contribution paths ranked by impact, feasibility, and likelihood of acceptance by Blender maintainers:

| Rank | Path | Impact | Feasibility | Acceptance Likelihood | Effort |
|------|------|--------|-------------|----------------------|--------|
| 1 | Standalone Cycles Render Server (Python) | High | High | High | 2-3 months |
| 2 | GPU Light Tree Builder (C++/CUDA) | Medium-High | Medium | Medium-High | 3-6 months |
| 3 | Spectral Rendering Foundation (C++) | Very High | Low-Medium | Medium | 12-24 months |
| 4 | Mojo Compute Kernels for specific hot-paths | Medium | Low (today) | Low (today) | Ongoing |

Key honest assessment: Mojo is NOT ready for direct Cycles contributions today. The pragmatic path is C++/CUDA contributions to Cycles itself, with Mojo as a future option once it matures (production-ready expected Q1 2026 per Modular).

---

## 1. Cycles Architecture: What You Are Actually Working With

### 1.1 Wavefront Path Tracing (The Core Engine)

Cycles-X (2022 rewrite) uses wavefront path tracing on GPU, based on the paper "Megakernels Considered Harmful" (Laine et al., NVIDIA 2013).

How it works (from src/kernel/integrator/):

```
GENERATE_CAMERA ---> INTERSECT ---> SHADE_SURFACE
                                         |
                    +--------------------+--------------------+
                    |                    |                    |
              SHADOW (NEE)        VOLUME (hetero)       SSS (subsurf)
                    |                    |                    |
                    +--------------------+--------------------+
                                         |
                                   ACCUMULATE (write render buf)
```

IntegratorState (the per-path data, from state_template.h):

Each path stores:
- path: pixel_index, sample, bounce depths (6 types), rng state, MIS data, throughput (PackedSpectrum), shader_sort_key, guiding state
- ray: origin P, direction D, differentials dP/dD, tmin/tmax, time
- isect: t, u, v, prim, object, type (intersection result)
- volume_stack[4]: object+shader pairs for volume nesting
- subsurface: albedo, radius, anisotropy, N
- guiding: path_segment pointer, sampling probabilities
- shadow_link: dedicated_light_weight, last_isect_prim/object

Critical detail: PackedSpectrum is RGB (3 x float32 = 12 bytes). For spectral rendering, this would need to become N x float32.

### 1.2 GPU Memory Layout

On GPU, all path state uses Structure-of-Arrays (SoA):
- IntegratorState is just an int (index into SoA arrays)
- Each field (path.throughput, path.flag, ray.P, etc.) stored in its own global memory array
- Access via macros: INTEGRATOR_STATE(state, path, throughput) dereferences kernel_data.integrator.state.path.throughput[state]
- Shadow paths stored in separate SoA arrays
- Paths sorted by shader for coherent execution in shade_surface

This SoA design is what makes spectral rendering hard -- adding spectral bins multiplies EVERY array.

### 1.3 The People You Would Work With

Cycles module owners (from Blender dev fund grants and contributor list):
- Brecht Van Lommel -- Original Cycles author, full-time grant, Cycles+core
- Sergey Sharybin -- Head of Development, long-time contributor
- William Leeson -- Cycles grant (since 2021), AMD HIP backend
- Lukas Stockner -- Part-time Cycles grant (SSS, denoiser contributions)
- Weizhen Huang -- Recent contributor
- Xavier Hallade -- Intel oneAPI/SYCL backend

How to contribute (from developer.blender.org/docs/handbook/contributing/):
1. PR on projects.blender.org/blender/blender (GitLab)
2. Must include: problem description, proposed solution, alternatives, limitations, UI mockup
3. Contact via #module-render-cycles on chat.blender.org
4. PRs need a module member to review and merge
5. Talk to developers BEFORE spending time coding bigger features

---

## 2. Contribution Path 1: Standalone Cycles Render Server

### Why This Is 1

This is the most immediately achievable and useful contribution. Cycles already has:
- A complete Hydra Render Delegate (src/hydra/, 17 source files)
- A standalone entry point (src/app/cycles_standalone.cpp)
- A clean Session API (src/session/session.h)
- USD file loading via HdCyclesFileReader

### Architecture

```
+-------------------------------------------+
|       FastAPI Server (Python)             |
|  +------+  +--------------------------+  |
|  | REST |  | WebSocket (live preview) |  |
|  | API  |  |                          |  |
|  +--+---+  +------------+-------------+  |
|     |                   |                |
|  +--v------------------v--------------+  |
|  |  Cycles Python Binding (pybind11)  |  |
|  |  wraps: Session, Scene,            |  |
|  |  SessionParams, Sync               |  |
|  +----------------+-------------------+  |
+--------------------+----------------------+
                     |
+--------------------v----------------------+
|  Cycles Session (C++)                    |
|  Session -> Device -> Scene              |
|  PathTrace -> RenderBuffers              |
|  DisplayDriver -> pixels to WebSocket    |
+------------------------------------------+
```

### Key Source Files

| File | What It Does | What You Would Use |
|------|-------------|-------------------|
| src/session/session.h | Session API, SessionParams, render control | Session(params), session->wait(), set_display_driver() |
| src/session/session.cpp | Thread management, render loop | Render state machine (WAIT/RENDER/END) |
| src/hydra/session.h/cpp | HdCyclesSession wrapping Session | Reference for how Hydra wraps Session |
| src/hydra/render_delegate.h | HdCyclesDelegate | Reference for render settings |
| src/app/cycles_standalone.cpp | CLI standalone app | Template for your entry point |
| src/session/display_driver.h | Display output interface | Custom driver to stream pixels via WebSocket |
| src/session/output_driver.h | File output interface | For final frame output |

### Implementation Steps

1. Build Cycles standalone first:
   cd cycles_mojo && make daemon
   (or cmake -DWITH_CYCLES_STANDALONE=ON)

2. Create Python binding (pybind11):
   - Bind SessionParams (device, samples, tile_size, etc.)
   - Bind SceneParams
   - Bind Session (create, start, wait, get_progress, set_display)
   - Bind BufferParams (width, height, passes)

3. Create FastAPI server:
   - POST /render -- submit render job with USD/file path + settings
   - GET /status -- progress, tile info
   - WS /preview -- WebSocket for live pixel streaming
   - GET /result -- download final EXR/PNG

4. Custom DisplayDriver:
   - Subclass DisplayDriver
   - In draw(), copy pixels to shared memory or pipe to WebSocket
   - This is how Blender viewport gets live updates -- same mechanism

### Why NOT Mojo

- Mojo has no async support (mojo-websockets author: "without proper async support, not usable")
- lightbug_http is basic HTTP only, no WebSocket
- Python FastAPI + uvicorn is production-proven, async-native, WebSocket-native
- Mojo value is in GPU kernels, not HTTP servers

### Likelihood of Blender Acceptance

HIGH. Blender is actively investing in USD and headless rendering workflows. The Hydra delegate already exists. A standalone render server with REST API would be genuinely useful for:
- Pipeline integration (farm rendering without Blender)
- Web-based render management
- Testing and benchmarking automation
- Education and research (easier to experiment with Cycles without Blender)

---

## 3. Contribution Path 2: GPU Light Tree Builder

### Current State

Cycles light tree (src/scene/light_tree.h, src/scene/light_tree.cpp):
- Built entirely on CPU using multi-threaded TaskPool
- Uses SAH (Surface Area Heuristic) splitting with 12 buckets
- MIN_EMITTERS_PER_THREAD = 4096 for parallelization threshold
- Structure: LightTreeNode is a variant of Leaf, Inner, Instance nodes
- Each node stores OrientationBounds (axis, theta_o, theta_e) for directional lights
- LightTreeEmitter stores position, measure, prim_id, light_set_membership

### Why GPU

- For scenes with thousands of lights (archviz, production scenes), the CPU build can be slow
- Scene edits (light added/moved) require full rebuild
- GPU BVH construction is well-studied: LBVH (Lauterbach et al. 2009), Fast BVH Construction on GPUs (Laine 2009), NVIDIA Blackwell Mega Geometry (2024)
- Recent research: "BVH Trees of Many Dynamic Lights for Real-Time Ray Tracing" (ICCS 2025)

### What You Would Build

A CUDA/HIP kernel that builds the light tree on GPU:

1. Leaf node generation: One GPU thread per light emitter, compute centroid + bounding box
2. Morton code computation: Transform centroids to 30-bit Morton codes
3. Radix sort: GPU radix sort on Morton codes (thrust/rocPRIM)
4. Hierarchical build: Bottom-up construction using sorted Morton codes
5. SAH refinement: Optional top-down SAH split for quality

### Files to Modify or Create

| File | Change |
|------|--------|
| src/scene/light_tree.h | Add GPU build method |
| src/scene/light_tree.cpp | Add build_gpu() alongside existing build() |
| src/device/gpu/light_tree_build.cpp | NEW: GPU kernel for light tree construction |
| src/device/cuda/kernel.cu | Register new kernel |
| src/scene/scene.cpp | Choose GPU vs CPU build based on device |

### Acceptance Criteria

Must match CPU build quality (same SAH cost) while being faster for more than 1000 emitters.

---

## 4. Contribution Path 3: Spectral Rendering Foundation

### The Grand Prize -- And Why It Is Hard

This is the single most requested Cycles feature in production. No major GPU path tracer does it well yet. But it is also the hardest contribution.

### What Cycles Currently Does

Everything is RGB. Specifically:
- PackedSpectrum = float3 (RGB, 12 bytes)
- Throughput is RGB multiplication
- BSDF evaluation returns RGB
- Texture sampling returns RGB
- The "Wavelength" node (svm_node_wavelength) does fake conversion: wavelength to XYZ to RGB using a lookup, then scales by 1/2.52f and clamps to 0. This is NOT spectral rendering.

### How Production Spectral Rendering Works

Weta Manuka (750K+ LOC, used on Avatar and Planet of the Apes):
- Hero Wavelength Spectral Sampling (HWSS): Sample ONE random wavelength per path, then add 3 more at fixed offsets (SSE-optimized)
- Batch-shading architecture: collect all shading points, sort by material, evaluate spectrally in batches
- Each path carries its sampled wavelength
- At film plane, convert spectral radiance to CIE XYZ to display color space

Inria "Efficient Spectral Rendering on the GPU" (Ray Tracing Gems II):
- Wavelength multiplexing: 32 wavelengths per ray on RTX 3070
- Performance: multiplexed 32 wavelengths x 16 rays = 0.283s vs single wavelength x 64 rays = 0.317s (12 percent faster)
- Upload spectral assets at bin boundaries, interpolate on GPU
- Decimation at refractive surfaces (random wavelength selection)

### What Would Need To Change In Cycles

This is the full scope. Every single item here touches GPU kernels:

#### Phase 1: Data Model (3-6 months)

1. PackedSpectrum to SpectralBlob:
   - Current: using PackedSpectrum = float3 (in kernel/types.h)
   - New: struct SpectralBlob { float wavelengths[N]; float values[N]; } where N is compile-time parameter
   - N=4 for hero wavelength, N=32 for multiplexed, N=3 for backward-compatible RGB

2. IntegratorState changes (state_template.h):
   - Add: KERNEL_STRUCT_MEMBER(path, float, wavelength, KERNEL_FEATURE_SPECTRAL)
   - Change: all PackedSpectrum throughput/weight fields to SpectralBlob
   - Memory impact: SoA arrays for throughput go from 12 bytes/path to 4*N bytes/path
   - For N=4: 16 bytes (33 percent increase). For N=32: 128 bytes (10x increase)

3. KernelData changes:
   - Add spectral CIE matching functions to kernel/data
   - Add spectral texture representation

#### Phase 2: BSDF Core (6-12 months)

4. BSDF evaluation (src/kernel/closure/):
   - Every BSDF (diffuse, glossy, glass, principled, etc.) must evaluate at wavelength lambda instead of RGB
   - About 20 BSDF types, each with eval(), sample(), pdf() methods
   - This is the bulk of the work

5. Texture sampling (src/kernel/svm/, src/kernel/osl/):
   - Currently returns RGB from texture
   - Must return spectral albedo (interpolated at wavelength bins)
   - Requires spectral texture preprocessing or on-the-fly upsampling

6. Emission (src/kernel/light/):
   - Light spectra must be stored spectrally, not RGB
   - Blackbody emission is naturally spectral (Planck law)
   - Need spectral importance sampling for colored lights

#### Phase 3: Integration (3-6 months)

7. Volume rendering (src/kernel/integrator/volume*):
   - Colored extinction becomes wavelength-dependent absorption
   - Currently RGB attenuation; spectral attenuation is straightforward but memory-heavy

8. Camera and Film:
   - Spectral sensor response to XYZ to display color space
   - Spectral white balance

### Why Mojo Could Help HERE Specifically

This is the ONE area where Mojo comptime would shine:

A Mojo spectral BSDF kernel could use comptime to specialize for N=3 (RGB fast path), N=4 (hero wavelength with SSE), or N=32 (full spectral), all from a single codebase.

But this is 2+ years away from being practical. C++ templates with ifdef can achieve similar results today.

### Blender Acceptance Strategy

This is too large for a single PR. The strategy must be:
1. Talk to Brecht/Sergey first on #module-render-cycles
2. Propose a compile-time spectral flag: WITH_CYCLES_SPECTRAL (CMake option)
3. Start with PackedSpectrum abstraction layer (no behavior change, just code refactor)
4. Add spectral BSDF evaluation as opt-in (flag-guarded)
5. Each phase is a separate PR

---

## 5. Contribution Path 4: Mojo Compute Kernels

### Reality Check

From the Mojo HPC paper (arXiv:2509.21039, "Mojo: MLIR-Based Performance-Portable HPC Science Kernels"):
- Mojo GPU kernels achieve competitive performance with CUDA/HIP for:
  - 7-point stencil (memory-bound)
  - BabelStream (memory bandwidth)
  - Hartree-Fock (compute-bound chemistry)
  - miniBUDE (molecular docking)
- Supports NVIDIA (A100, H100) and AMD MI300X (since June 2025)
- Production-ready expected Q1 2026 (per Modular)

But for Cycles:
- No ray tracing API (no BVH, no OptiX integration)
- No GPU rendering code exists in Mojo anywhere
- Cannot call OptiX/Embree from Mojo kernels
- Mojo value is compute kernels, not full rendering pipelines

### What Could Actually Be Done

1. Spectral BSDF micro-benchmark: Write a Mojo kernel that evaluates Principled BSDF at N wavelengths. Compare to Cycles CUDA kernel. Standalone benchmark, not integrated.

2. Tile-based denoiser: Mojo TileTensor + autotune could optimize tile sizes for denoising. But Cycles already uses OptiX built-in denoiser (trained on 20K+ images on DGX-1). Building a competitive denoiser from scratch is unrealistic.

3. Neural adaptive sampling inference: If a neural network is trained (via PyTorch/JAX) to predict per-pixel convergence, Mojo could run inference alongside path tracing on the same GPU. But this requires the network to exist first.

### Honest Assessment

Mojo contributions to Cycles are not feasible today for anything meaningful. Wait for:
- Mojo production-ready (Q1 2026)
- Mojo-to-C FFI for calling into Cycles device API
- Or: contribute to Cycles in C++, contribute to Mojo ecosystem separately

---

## 6. Neural Adaptive Sampling: The Dark Horse

### Current State

Cycles adaptive sampling (src/integrator/adaptive_sampling.cpp) is purely rule-based: wait min_samples, then check every adaptive_step samples. No learning involved.

The entire logic is about 50 lines of code:
- align_samples(): compute how many samples until next filter point
- need_filter(): check if sample index is on a filter boundary

### State of the Art

- "Forget Superresolution, Sample Adaptively" (2025): First end-to-end sub-1-spp pipeline using stochastic sample placement. Trains a small network to predict where to place samples.
- NVIDIA "Neural Temporal Adaptive Sampling and Denoising" (2020): Two co-trained CNNs -- one for sample allocation, one for denoising.
- AMD Neural Supersampling: Similar approach for real-time.

### How This Could Work In Cycles

1. Add a convergence estimation kernel (GPU)
2. After every N samples, run a lightweight CNN to predict per-pixel variance
3. Use prediction to allocate more samples to high-variance pixels
4. This replaces the current adaptive_step heuristic with a learned model

### Challenge

This requires training data (rendered images with ground truth). Possible sources:
- Blender Open Movie renders (Spring, Elephant Dream, etc.)
- Synthetic training on procedural scenes
- Pre-trained model from research paper

---

## 7. Recommended Action Plan

### Immediate (Week 1-4): Standalone Server
1. Build Cycles standalone (make daemon or cmake)
2. Study src/hydra/ and src/session/ APIs
3. Create minimal Python binding with pybind11
4. Build FastAPI server with render endpoint
5. File PR or discussion on projects.blender.org

### Short-term (Month 2-3): GPU Light Tree
1. Profile current CPU light tree build time on 1000+ light scenes
2. Implement GPU Morton code sort + hierarchical build
3. Benchmark against CPU version
4. File PR with performance numbers

### Medium-term (Month 4-6): Talk to Brecht About Spectral
1. Write a design document (not code) for spectral rendering
2. Post on #module-render-cycles or devtalk.blender.org
3. Propose PackedSpectrum abstraction as Phase 1
4. Get buy-in before writing any code

### Long-term (6-12 months): Spectral Implementation
1. Implement Phase 1 (data model) behind WITH_CYCLES_SPECTRAL
2. Implement Phase 2 (spectral BSDF) for diffuse + glossy only
3. Test on spectral validation scenes
4. Submit incremental PRs

### Mojo: Wait and Watch
- Track Mojo production release (Q1 2026)
- Build Mojo GPU proficiency with the mojo-gpu-puzzles
- Write standalone Mojo spectral BSDF benchmark
- When Mojo has C FFI and async: revisit integration

---

## 8. Key Contacts and Resources

| Resource | URL/Purpose |
|----------|-------------|
| Cycles dev docs | developer.blender.org/docs/features/cycles/ |
| Kernel scheduling | developer.blender.org/docs/features/cycles/kernel_scheduling/ |
| Contributing guide | developer.blender.org/docs/handbook/contributing/ |
| Cycles chat | #module-render-cycles on chat.blender.org |
| Bug tracker | projects.blender.org/blender/blender (Cycles project) |
| Module owners | Brecht Van Lommel, Sergey Sharybin, William Leeson |
| PR template | Problem, Solution, Alternatives, Limitations, Mockup |

### Papers Referenced

1. Laine et al. "Megakernels Considered Harmful: Wavefront Path Tracing on GPUs" (HPG 2013) -- basis for Cycles-X wavefront architecture
2. Wilkie et al. "Hero Wavelength Spectral Sampling" (EGSR 2014) -- HWSS technique used by Manuka
3. Fascione et al. "Manuka: A Batch-Shading Architecture for Spectral Path Tracing" (SIGGRAPH 2018) -- production spectral renderer
4. Nakamura et al. "Efficient Spectral Rendering on the GPU" (Ray Tracing Gems II) -- GPU wavelength multiplexing
5. Zheng et al. "GPU Coroutines for Flexible Splitting and Scheduling" (2024) -- advanced wavefront scheduling
6. "Forget Superresolution, Sample Adaptively" (2025) -- neural adaptive sampling
7. arXiv:2509.21039 "Mojo: MLIR-Based Performance-Portable HPC Science Kernels" -- Mojo GPU benchmarking
8. "BVH Trees of Many Dynamic Lights for Real-Time Ray Tracing" (ICCS 2025) -- GPU light tree construction
