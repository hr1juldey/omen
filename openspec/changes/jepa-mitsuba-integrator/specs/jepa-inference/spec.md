## ADDED Requirements

### Requirement: Load JEPA model from compiled Mojo library

Omen SHALL load JEPA inference model from compiled Mojo shared library (.so/.dll/.dylib) via Python ctypes. Loading SHALL use `ctypes.CDLL` and resolve function symbols: `omen_denoise`, `omen_predict_confidence`, `omen_merge_multires`.

#### Scenario: Load library successfully

- **WHEN** `libomen.so` exists in `jepa_kernels/` directory
- **THEN** bridge loads library via ctypes.CDLL
- **AND** all function symbols are resolved
- **AND** bridge is ready for inference

#### Scenario: Handle missing library

- **WHEN** `libomen.so` does not exist
- **THEN** bridge logs error message
- **AND** falls back to standard Mitsuba path tracer
- **AND** raises no exception (graceful degradation)

### Requirement: Pass scene graph to JEPA model

Omen SHALL serialize extracted SceneGraph data to C-compatible struct and pass pointer to Mojo JEPA model. Data SHALL include geometry pointers, material arrays, light arrays, and camera parameters matching C ABI definition in `omen_bridge.h`.

#### Scenario: Pass simple scene graph

- **WHEN** scene has 1 mesh, 1 material, 1 light
- **THEN** bridge populates SceneGraph C struct
- **AND** sets geometry array with 1 element containing vertices pointer
- **AND** sets materials array with BSDF parameters
- **AND** sets lights array with emitter data
- **AND** passes pointer via ctypes to `omen_denoise`

#### Scenario: Handle large scene graph

- **WHEN** scene has 1000+ meshes
- **THEN** bridge allocates contiguous arrays for all data
- **AND** fills C struct with array pointers and lengths
- **AND** passes to Mojo without copying (zero-copy where possible)

### Requirement: Pass render observation to JEPA model

Omen SHALL extract noisy RGBA render and optional auxiliary passes (depth, normal, albedo) from Mitsuba Film/ImageBlock and pass to JEPA model as RenderObservation struct.

#### Scenario: Pass noisy render only

- **WHEN** mode=1 (denoiser) and no auxiliary passes enabled
- **THEN** bridge extracts RGBA from `mi.TensorXf`
- **AND** creates RenderObservation with noisy_rgba pointer
- **AND** sets depth, normal, albedo to NULL
- **AND** passes to `omen_denoise`

#### Scenario: Pass render with auxiliary passes

- **WHEN** mode=2 (adaptive) and depth/normal/albedo available
- **THEN** bridge extracts auxiliary passes from separate Film channels
- **AND** creates RenderObservation with all passes populated
- **AND** passes to `omen_predict_confidence`

### Requirement: Invoke JEPA denoise inference

Omen SHALL call `omen_denoise` function from Mojo library with scene graph and render observation. Function SHALL write denoised RGBA output to pre-allocated buffer.

#### Scenario: Successful denoise inference

- **WHEN** `omen_denoise` is called with valid scene and observation
- **THEN** Mojo kernel runs inference on GPU
- **AND** writes denoised pixels to output_rgba buffer
- **AND** returns 0 (success code)
- **AND** output buffer contains denoised image [H, W, 4]

#### Scenario: Handle inference failure

- **WHEN** JEPA model encounters error during inference
- **THEN** `omen_denoise` returns non-zero error code
- **AND** bridge logs error message
- **AND** falls back to returning original noisy render

### Requirement: Invoke confidence prediction

Omen SHALL call `omen_predict_confidence` function from Mojo library to generate per-pixel confidence map and denoised image. Function SHALL write both output_rgba and output_confidence buffers.

#### Scenario: Predict confidence on preview render

- **WHEN** mode=2 (adaptive) and preview render (4 spp) is available
- **THEN** bridge calls `omen_predict_confidence` with scene graph
- **AND** receives output_rgba [H, W, 4] (JEPA-predicted pixels)
- **AND** receives output_confidence [H, W, 1] (0=uncertain, 1=confident)
- **AND** returns success code

#### Scenario: Use confidence for sample allocation

- **WHEN** confidence map is received from JEPA
- **THEN** high-confidence pixels (>0.8) use JEPA prediction
- **AND** low-confidence pixels (<0.5) trigger full path tracing
- **AND** medium-confidence pixels use intermediate sampling

### Requirement: Invoke multi-resolution merge

Omen SHALL call `omen_merge_multires` function from Mojo library to merge low-resolution high-quality render with high-resolution noisy render using scene graph guidance.

#### Scenario: Merge multi-resolution inputs

- **WHEN** mode=3 (multires) and both renders are available
- **THEN** bridge calls `omen_merge_multires` with scene graph
- **AND** passes low_res_high_qual [H/4, W/4, 4]
- **AND** passes high_res_noisy [H, W, 4]
- **AND** passes scale_factor=4
- **AND** receives output_merged [H, W, 4] (scene-guided upscaled)

#### Scenario: Handle resolution mismatch

- **WHEN** low_res render is not exactly 1/scale of high_res
- **THEN** Mojo kernel detects size mismatch
- **AND** resamples low_res to match expected dimensions
- **AND** logs warning about resampling

### Requirement: Zero-copy GPU buffer passing

Omen SHALL wrap existing GPU buffers from Mitsuba (CUDA/HIP pointers) as Mojo DeviceBuffer without memory copying using `UnsafePointer` and owning=False parameter.

#### Scenario: Wrap Mitsuba render buffer

- **WHEN** Mitsuba render result is in GPU memory
- **THEN** bridge extracts raw pointer via `.data.ptr`
- **AND** wraps as `DeviceBuffer[Float32](ptr, size, owning=False)`
- **AND** passes wrapped buffer to Mojo without memcpy
- **AND** Mojo reads directly from Mitsuba's GPU memory
