# Cycles GPU Model vs Mojo GPU Model — Deep Comparison & Analysis

## 1. Architecture Comparison: How Each Talks to the GPU

### Cycles' GPU Model (Current — 6 backends)

Cycles uses a **vendor-specific, heterogeneous GPU abstraction layer**:

```
Host (C++)                    Device (Vendor-specific kernels)
─────────────                 ─────────────────────────────────
scene/ → SVM bytecode
         ↓
device/device.h               kernel/device/cuda/*.h    (NVCC → PTX/CUBIN)
  ↓ compile_kernel()          kernel/device/optix/*.h   (OptiX pipeline + SBT)
device_queue.h                kernel/device/hip/*.h     (HIPCC → fatbin)
  ↓ enqueue()                 kernel/device/metal/*.mm  (Metal Shading Language)
device_memory.h               kernel/device/oneapi/*.h  (SYCL/DPC++)
                              kernel/device/cpu/*.h     (Native C++/SSE/AVX)
```

**Key characteristics:**
- Each GPU backend has its **own kernel source tree** (`kernel/device/{cuda,optix,hip,metal,oneapi}/`)
- Kernel code is written in **C++ with vendor-specific intrinsics** (`ccl_device`, `__kernel_gpu__` macros)
- Compilation: NVCC → PTX/CUBIN, HIPCC → fatbin, Metal compiler → .metallib, DPC++ → SPIR-V
- OptiX uses **dedicated ray-tracing pipelines** with Shader Binding Tables (SBT)
- A unified `KernelData` struct is copied to device memory — all scene data lives in one flat blob
- BVH is built per-backend: OptiX `optixBuild`, Embree `rtcCommit`, HIP RT, Metal `MPSCustomAcceleration`
- SVM (Shader Virtual Machine) bytecode is interpreted on the GPU for material evaluation

### Mojo's GPU Model

```
Host (Mojo)                   Device (Mojo → MLIR → vendor)
─────────────                 ─────────────────────────────
DeviceContext()
  ↓ compile_function[kernel]  kernel = ordinary Mojo function
  ↓ enqueue_function(...)     ↓ compiled for target via MLIR
DeviceBuffer / TileTensor     ↓ NVIDIA → PTX
  ↓ map_to_host()             ↓ AMD → ROCm/HSACO
  ↓ synchronize()             ↓ Apple → Metal
```

**Key characteristics:**
- **Single language** for host AND device code (Mojo)
- Kernels are plain functions compiled via `compile_function` → MLIR → vendor backend
- `TileTensor` / `LayoutTensor` with automatic layout management and shared-memory tiling
- `AddressSpace.SHARED` for shared memory, `barrier()` for sync
- `comptime` specialization generates architecture-specific variants at compile time
- `autotune` selects optimal parameters (tile sizes, unroll factors) empirically
- Warp primitives (`shuffle_up/down/xor`, `warp.sum`) map to vendor intrinsics
- Currently supports NVIDIA, AMD, Apple Silicon

---

## 2. Direct Comparison Table

| Aspect | Cycles (Current) | Mojo (Hypothetical) |
|--------|-------------------|---------------------|
| **Language** | C++ (host) + CUDA/HIP/Metal/SYCL (device) | Single: Mojo for both |
| **Kernel compilation** | NVCC/HIPCC/Metal compiler offline → binary | `compile_function` → MLIR → vendor |
| **Kernel launch** | `queue->enqueue(kernel, work_size, args)` | `ctx.enqueue_function(fn, grid_dim, block_dim, args)` |
| **Thread indexing** | Custom macros: `ccl_gpu_thread_index`, per-backend | `thread_idx.x`, `block_idx.x`, `global_idx.x` |
| **Shared memory** | `ccl_gpu_shared` / `__shared__` | `stack_allocation(..., AddressSpace.SHARED)` |
| **Synchronization** | `ccl_gpu_barriersync` → `__syncthreads()` | `barrier()` |
| **Memory model** | Flat `KernelData` blob + `device_vector<T>` | `DeviceBuffer`, `TileTensor`, `LayoutTensor` |
| **BVH** | Per-backend (OptiX RT pipeline, Embree, HIP RT, BVH2) | Must be built from scratch in Mojo |
| **Shader evaluation** | SVM bytecode interpreted on GPU | Would need reimplementation |
| **Ray tracing** | OptiX hardware RT cores (native) | No built-in RT — must implement |
| **Autotuning** | None (manual parameter selection) | `autotune(1,2,4,8,...)` built-in |
| **Compile-time specialization** | `#ifdef` / CMake flags | `comptime if`, `@parameter`, traits |
| **Backend count** | 6 (CUDA, OptiX, HIP, Metal, oneAPI, CPU) | 3 (NVIDIA, AMD, Apple) |
| **Maturity** | 17 years, production-proven | ~2 years, experimental |

---

## 3. Can Mojo Outperform Cycles' Current GPU Code?

### Short answer: No — not in the near term, and not for the full pipeline.

### Detailed analysis:

#### Where Cycles wins decisively:

1. **OptiX hardware ray tracing**: Cycles uses NVIDIA's RT cores natively via OptiX's `optixLaunch` pipeline with dedicated ray-generation, hit, and miss programs. This gets hardware-accelerated BVH traversal on RTX GPUs. Mojo has **no ray tracing API** — you'd have to implement BVH traversal in software (Mojo kernels), which would be orders of magnitude slower than RT cores.

2. **Embree for CPU**: Cycles uses Intel Embree (production-grade ray tracing library with AVX-512 optimization). Mojo would need to either FFI into Embree or reimplement it.

3. **Maturity of the full pipeline**: Cycles' path tracing kernel (`kernel/integrator/`) is a deeply optimized 17-year codebase handling:
   - Unidirectional path tracing with NEE (next event estimation)
   - Subsurface scattering (random walk + disk)
   - Volume rendering (heterogeneous, with equi-angular sampling)
   - Hair/curve intersection
   - Motion blur (deformation + object)
   - Light tree for many-light scenes
   - Manifold next event estimation (MNEE) for caustics
   - Path guiding (OpenPGL integration)

   Each of these took years to optimize. Rewriting this in Mojo would be a multi-year effort.

4. **Multi-backend support**: Cycles runs on CUDA, OptiX, HIP, HIP RT, Metal, oneAPI, and CPU. Mojo currently has 3 GPU backends with limited profiling/debugging support on AMD and Apple.

5. **Shader system**: Cycles has both SVM (bytecode VM on GPU) and OSL (Open Shading Language) for production-quality material authoring. The SVM alone has ~30 closure types and ~60 texture/shader node types. There's no equivalent in Mojo.

#### Where Mojo could theoretically match or help:

1. **Compute-heavy kernels** (not ray tracing): If you isolate pure compute tasks — like SVM bytecode evaluation, BSDF sampling, volume integration math, or denoising — Mojo's MLIR pipeline with `autotune` could potentially match or beat hand-tuned CUDA for specific kernels. The `TileTensor` + `autotune` combination lets you empirically find optimal tiling strategies per architecture without manual tuning.

2. **Compile-time specialization**: Mojo's `comptime if` + `autotune` can generate architecture-specialized kernels more cleanly than Cycles' `#ifdef __KERNEL_OPTIX__` / `#ifdef __KERNEL_CUDA__` macros. For a new feature, Mojo could auto-specialize across GPU architectures.

3. **Unified codebase**: Writing kernel code once in Mojo instead of maintaining 6 separate backend-specific kernel trees (CUDA, OptiX, HIP, Metal, oneAPI, CPU) would be a massive maintenance win — but only if Mojo's MLIR backend generates code competitive with each vendor's native compiler.

---

## 4. What NEW Rendering Avenues Could Mojo Open?

This is the more interesting question. Here's what Mojo could enable that Cycles currently cannot:

### 4.1 Bidirectional Path Tracing (BDPT)

**Why Cycles doesn't have it**: BDPT requires connecting sub-paths from the camera AND from the light source. This means:
- Two separate path random walks per sample
- A connection step that evaluates visibility between all pairs of vertices
- This is extremely memory-intensive on GPU (storing full path vertices for thousands of threads)
- Cycles' current kernel architecture uses a state machine (`IntegratorStateGPU`) optimized for unidirectional tracing — adding BDPT would nearly double the state size

**How Mojo could help**:
- Mojo's `comptime` metaprogramming could generate specialized BDPT kernels with exactly the state layout needed
- `autotune` could find optimal configurations for path lengths, connection strategies
- `TileTensor` shared memory could be used for efficient vertex sharing between thread groups
- BUT: the fundamental algorithmic challenge (memory pressure, warp divergence) remains — Mojo doesn't solve the GPU BDPT problem that has blocked every production path tracer except V-Ray and Mitsuba

### 4.2 Full Spectral Rendering

**Why Cycles uses RGB**: Cycles works in RGB color space (3 components). Full spectral rendering uses 10-80+ wavelength bins. This:
- Multiplies memory bandwidth by 3-25x
- Makes BSDF evaluation much more expensive
- Breaks all the RGB-optimized texture sampling paths

**How Mojo could help**:
- `comptime` parameterized spectral resolution: `fn spectral_bsdf[n_wavelengths: Int](...)` — compile once, get optimized code for any spectral resolution
- SIMD vectorization: Mojo has first-class `SIMD[type, N]` types that could map wavelength bins directly to SIMD lanes
- MLIR could auto-vectorize spectral loops where Cycles would need manual intrinsics
- This is actually one of the **most promising** areas for Mojo — the compile-time specialization could make spectral rendering practical on GPU without hand-writing intrinsics for each wavelength count

### 4.3 Faster Interactive/Real-time Preview (Eevee-like)

**Why Cycles isn't instant**: Cycles is a path tracer — every pixel needs hundreds of samples. Even with:
- Adaptive sampling (stops sampling converged pixels early)
- GPU acceleration
- OptiX denoising
- Light tree for efficient many-light sampling

...a single sample still requires full ray tracing + shading. Cycles can't match Eevee because Eevee is a rasterizer with screen-space effects, not a path tracer.

**How Mojo could help**:
- Mojo could potentially accelerate **denoising** kernels (custom wavelet/neural denoisers) using `TileTensor` shared-memory tiling + `autotune`
- Could enable **hybrid rendering**: a Mojo-based Eevee-like rasterizer for interactive preview that shares scene data with a Mojo-based path tracer for final renders
- MLIR's ability to fuse operations could reduce kernel launch overhead in the tile-based rendering pipeline
- BUT: the fundamental gap between rasterization (Eevee) and path tracing (Cycles) is algorithmic, not language-level

### 4.4 Neural Rendering / AI-Accelerated Sampling

**Most promising new avenue**: Mojo's MLIR pipeline is literally designed for ML workloads.

- **Neural radiance caching**: Train a small neural network on-the-fly to predict indirect lighting, reducing samples needed
- **AI-upscaled path tracing**: Render at lower resolution with fewer samples, use a neural network to reconstruct high-quality output
- **Learned importance sampling**: Replace Cycles' MIS (multiple importance sampling) with learned distributions

Mojo's strengths here:
- Same language for GPU kernels AND ML inference
- `autotune` can optimize neural network tile sizes
- MLIR naturally lowers to vendor tensor cores (NVIDIA Tensor Core, AMD Matrix Core, Apple Neural Engine)
- Mojo's Python interop could bring in PyTorch/JAX training pipelines while keeping inference in Mojo

### 4.5 Custom Hardware Acceleration

Mojo's `comptime` architecture checks:
```mojo
comptime if is_nvidia_gpu():
    # Use Tensor Cores for matrix operations in BSDF evaluation
elif is_amd_gpu():
    # Use Matrix Cores
elif is_apple_gpu():
    # Use Apple GPU tile memory
```

This is cleaner than Cycles' approach of having entirely separate kernel source trees per backend. For a new feature, you write once and specialize.

---

## 5. Realistic Assessment: What Should Actually Be Done

### Option A: Rewrite Cycles in Mojo (DON'T DO THIS)
- Estimated effort: 5-10 years
- Would lose OptiX RT cores, Embree, OSL, production maturity
- No clear performance win for the core path tracing

### Option B: Mojo-accelerated Cycles kernels (PRAGMATIC)
Write specific hot-path kernels in Mojo that Cycles calls via FFI:
1. **Spectral BSDF evaluation kernel**: Mojo's SIMD + comptime specialization
2. **Custom denoiser**: Mojo TileTensor + autotune for tile-optimized denoising
3. **Volume integration**: Mojo's warp primitives for efficient volume sampling
4. **Neural rendering components**: Mojo for on-device ML inference
5. **Texture sampling**: Mojo's LayoutTensor for cache-optimized texture access

### Option C: Mojo-based companion renderer (FORWARD-LOOKING)
Build a new renderer alongside Cycles that shares scene format but uses Mojo throughout:
- Spectral rendering as a first-class feature
- Neural-boosted sampling
- BDPT + Metropolis light transport
- Target: not replacing Cycles, but complementing it for specific use cases (archviz, product viz where spectral accuracy matters)

---

## 6. Bottom Line

| Question | Answer |
|----------|--------|
| Can Mojo outperform Cycles on GPU? | **No**, not for the full pipeline. OptiX RT cores + 17 years of optimization is unbeatable today. |
| Can Mojo match Cycles kernel performance? | **Potentially** for compute-bound kernels (not ray tracing) with autotune + MLIR. |
| Can Mojo enable BDPT? | The language helps (comptime state layouts), but the fundamental GPU memory pressure problem remains. |
| Can Mojo enable spectral rendering? | **Yes — this is the most promising avenue.** SIMD + comptime wavelength parameterization + autotune makes spectral rendering on GPU practical. |
| Can Mojo make Cycles as fast as Eevee? | **No.** That's an algorithmic difference (rasterization vs path tracing), not a language issue. |
| Can Mojo enable neural/AI rendering? | **Yes — second most promising.** MLIR is built for ML, and Mojo could run neural inference alongside path tracing on the same GPU. |
| Is a full rewrite worth it? | **Absolutely not.** Incremental Mojo kernels for specific features (spectral, neural) is the pragmatic path. |

The real value of Mojo for Cycles isn't "faster of the same thing" — it's **new capabilities that Cycles' C++ architecture makes impractical**, specifically spectral rendering and neural-boosted sampling.
