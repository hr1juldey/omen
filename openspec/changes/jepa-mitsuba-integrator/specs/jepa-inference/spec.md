## ADDED Requirements

### Requirement: Load JEPA model from compiled Mojo library

Omen SHALL load JEPA inference model from compiled Mojo shared library (.so/.dll/.dylib) via Python ctypes. Loading SHALL use `ctypes.CDLL` and resolve function symbols: `omen_denoise`, `omen_predict_confidence`, `omen_merge_multires`, `omen_train_step`.

#### Scenario: Load library successfully

- **WHEN** `libomen.so` exists at resolved path
- **THEN** load via `ctypes.CDLL(lib_path)`
- **AND** set argument types for each function:
  - `lib.omen_denoise.argtypes = [SceneGraphC, RenderObservationC, ctypes.POINTER(ctypes.c_float), ctypes.c_int]`
  - `lib.omen_denoise.restype = ctypes.c_int32`
- **AND** resolve all symbols: `omen_denoise`, `omen_predict_confidence`, `omen_merge_multires`, `omen_train_step`
- **AND** verify with: `nm -D libomen.so | grep omen_` showing all 4 symbols
- **AND** set `self.available = True`

#### Scenario: Handle missing library (graceful degradation)

- **WHEN** `libomen.so` not found at any search path
- **THEN** catch `OSError` from `ctypes.CDLL`
- **AND** log: "Omen JEPA library not found at {paths_checked}. Falling back to standard path tracing."
- **AND** set `self.available = False`
- **AND** all subsequent bridge calls return unmodified input (passthrough mode)
- **AND** no exception raised — existing Mitsuba path tracing works normally

### Requirement: Pass scene graph to JEPA model

Omen SHALL serialize extracted SceneGraph data to C-compatible struct and pass pointer to Mojo JEPA model. Data layout MUST match `SceneGraph` struct in `omen_bridge.h`.

#### Scenario: Pass scene graph via ctypes struct

- **WHEN** scene has N meshes, M materials, L lights, 1 camera
- **THEN** define ctypes struct matching C header:
  ```python
  class SceneGraphC(ctypes.Structure):
      _fields_ = [
          ('vertices', ctypes.POINTER(ctypes.c_float)),
          ('num_vertices', ctypes.c_int32),
          ('faces', ctypes.POINTER(ctypes.c_uint32)),
          ('num_faces', ctypes.c_int32),
          ('material_ids', ctypes.POINTER(ctypes.c_uint32)),
          ('material_params', ctypes.POINTER(ctypes.c_float)),
          ('num_materials', ctypes.c_int32),
          ('light_params', ctypes.POINTER(ctypes.c_float)),
          ('num_lights', ctypes.c_int32),
          ('camera_transform', ctypes.c_float * 16),
          ('camera_fov', ctypes.c_float),
          ('camera_near', ctypes.c_float),
          ('camera_far', ctypes.c_float),
          ('camera_aspect', ctypes.c_float),
      ]
  ```
- **AND** populate all fields from extracted scene graph
- **AND** pass struct by value to Mojo function

### Requirement: Pass render observation to JEPA model

Omen SHALL extract render buffers from Mitsuba and pass as `RenderObservation` C struct with RGBA pointer and optional auxiliary passes.

#### Scenario: Pass RGBA render only (denoiser mode)

- **WHEN** mode=1 (denoiser) and render result is TensorXf `(H, W, 3)`
- **THEN** add alpha channel: concatenate ones array → `rgba (H, W, 4)`
- **AND** convert to contiguous numpy: `np.ascontiguousarray(rgba, dtype=np.float32)`
- **AND** extract pointer: `rgba.ctypes.data_as(ctypes.POINTER(ctypes.c_float))`
- **AND** set `depth_ptr = normal_ptr = albedo_ptr = None`
- **AND** create RenderObservationC struct with `rgba_ptr`, `width`, `height`, `size=H*W*4`

#### Scenario: Pass render with auxiliary passes (adaptive mode)

- **WHEN** mode=2 (adaptive) and depth/normal/albedo AOVs are available
- **THEN** extract auxiliary passes from Mitsuba Film's multi-channel bitmap
- **AND** populate all RenderObservationC pointer fields
- **AND** auxiliary passes used as additional conditioning for confidence prediction

### Requirement: Invoke JEPA denoise inference

Omen SHALL call `omen_denoise` function from Mojo library with scene graph and render observation. Function SHALL write denoised RGBA output to pre-allocated buffer.

#### Scenario: Successful denoise inference

- **WHEN** `omen_denoise` is called with valid scene and observation
- **THEN** allocate output buffer: `(ctypes.c_float * (width * height * 4))()`
- **AND** call `result = self.lib.omen_denoise(scene_c, obs_c, output, gpu_device_id)`
- **AND** check return code: `result == 0` → success
- **AND** convert output to numpy: `np.ctypeslib.as_array(output, shape=(height*width*4,)).reshape(height, width, 4)`
- **AND** return as numpy array

#### Scenario: Handle inference failure

- **WHEN** `omen_denoise` returns non-zero error code
- **THEN** map error codes: -1 → GPU unavailable, -2 → invalid params, -3 → OOM
- **AND** log: "JEPA denoise failed (code {result}): {error_message}"
- **AND** return original noisy render unchanged

### Requirement: Invoke confidence prediction

Omen SHALL call `omen_predict_confidence` function to generate per-pixel confidence map and denoised image simultaneously.

#### Scenario: Predict confidence on preview render

- **WHEN** mode=2 (adaptive) and preview render (4 spp) is available
- **THEN** allocate two output buffers: `output_rgba` and `output_confidence` (both `c_float * size`)
- **AND** call `result = self.lib.omen_predict_confidence(scene_c, obs_c, output_rgba, output_confidence, gpu_id)`
- **AND** return `(clean_preview: np.ndarray[H,W,4], confidence: np.ndarray[H,W,1])`

### Requirement: Invoke multi-resolution merge

Omen SHALL call `omen_merge_multires` function to merge low-res high-quality render with high-res noisy render.

#### Scenario: Merge multi-resolution inputs

- **WHEN** mode=3 (multires) and both renders are available
- **THEN** create two RenderObservationC structs: `low_res_obs` and `high_res_obs`
- **AND** call `result = self.lib.omen_merge_multires(scene_c, low_res_obs, high_res_obs, output, scale=4, gpu_id)`
- **AND** output: merged RGBA `(H, W, 4)` — clean color from low-res, sharp detail from high-res

### Requirement: Zero-copy GPU buffer passing

Omen SHALL pass GPU buffers from Mitsuba directly to Mojo without host-device copy when both are on the same GPU.

#### Scenario: Zero-copy path (CUDA variant)

- **WHEN** `mi.variant()` is `cuda_ad_rgb` and render tensor is on GPU
- **THEN** get raw device pointer from Dr.Jit tensor
- **AND** pass as `ctypes.c_void_p(ptr_value)` in RenderObservationC
- **AND** Mojo wraps with `DeviceBuffer[DType.float32](ctx, raw_ptr, count, owning=False)`
- **AND** NO `memcpy` occurs — Mojo reads directly from Mitsuba's GPU memory
- **AND** log "Zero-copy mode: Mitsuba CUDA → Mojo CUDA"

#### Scenario: CPU-GPU memcpy fallback

- **WHEN** Mitsuba is on CPU (`scalar_rgb` or `llvm_ad_rgb`) and JEPA is on GPU
- **THEN** numpy array is already in host memory
- **AND** pass host pointer to Mojo C ABI
- **AND** Mojo allocates device buffer and copies host→device via `ctx.enqueue_copy`
- **AND** log "Memcpy mode: CPU → GPU (10-50ms overhead for 256×256)"
