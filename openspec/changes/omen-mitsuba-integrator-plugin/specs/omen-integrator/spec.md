## ADDED Requirements

### Requirement: Omen integrator renders scenes using path tracing

Omen SHALL implement a Mitsuba 3 integrator that performs Monte Carlo path tracing to render scenes. The integrator MUST be registered as "omen" in Mitsuba's plugin system and MUST be invocable from both Python API and Mitsuba-Blender addon.

#### Scenario: Register Omen as Mitsuba plugin

- **WHEN** Omen module is imported
- **THEN** integrator is registered via `mi.register_integrator("omen", lambda props: OmenIntegrator(props))`

#### Scenario: Render scene with Omen from Python

- **WHEN** user calls `mi.render(scene)` with integrator type="omen"
- **THEN** scene is rendered using path tracing algorithm
- **AND** returned TensorXf contains pixel values

#### Scenario: Render scene from Blender

- **WHEN** user selects "Omen" in Mitsuba-Blender integrator dropdown
- **THEN** Blender renders scene using Omen integrator
- **AND** result appears in Image Editor

### Requirement: Configurable path depth controls

Omen SHALL support `max_depth` parameter to control maximum path length and `rr_depth` parameter to control Russian roulette start depth. Default values SHALL be -1 (infinite) for max_depth and 5 for rr_depth.

#### Scenario: Set maximum path depth

- **WHEN** user sets max_depth=8
- **THEN** paths terminate after 8 bounces maximum

#### Scenario: Infinite path depth

- **WHEN** user sets max_depth=-1
- **THEN** paths continue until Russian roulette termination

#### Scenario: Configure Russian roulette

- **WHEN** user sets rr_depth=3
- **THEN** Russian roulette begins at depth 3

### Requirement: BSDF and emitter sampling

Omen SHALL perform multiple importance sampling combining BSDF sampling and emitter sampling at each path vertex. The integrator SHALL handle both area and directional emitters.

#### Scenario: Sample BSDF at surface

- **WHEN** path intersects surface with BSDF
- **THEN** sample outgoing direction using BSDF::sample()
- **AND** accumulate throughput weight

#### Scenario: Connect to emitter

- **WHEN** path vertex can see emitter
- **THEN** sample direct illumination via next event estimation
- **AND** combine with BSDF sampling using MIS weights

### Requirement: JEPA model parameter placeholder

Omen SHALL accept `jepa_model` string parameter specifying path to JEPA model file. This parameter SHALL be stored but NOT used in initial implementation (reserved for future JEPA acceleration).

#### Scenario: Configure JEPA model path

- **WHEN** user sets jepa_model="/models/jepa.pt"
- **THEN** integrator stores the path
- **AND** parameter is accessible via traverse mechanism

### Requirement: Rendering mode selection

Omen SHALL accept `mode` integer parameter (default: 0) to select rendering strategy:
- 0: Standard path tracing (uses Mitsuba's C++ path tracer)
- 1: JEPA denoiser (post-process)
- 2: Confidence-guided adaptive (multi-pass)
- 3: Multi-resolution merge (spatial upsampling)

#### Scenario: Select standard mode

- **WHEN** user sets mode=0
- **THEN** Omen delegates to Mitsuba's path integrator
- **AND** renders using standard path tracing

#### Scenario: Select denoiser mode

- **WHEN** user sets mode=1
- **THEN** Omen renders at low spp (4-16)
- **AND** applies JEPA denoising
- **AND** returns denoised output

#### Scenario: Select adaptive mode

- **WHEN** user sets mode=2
- **THEN** Omen renders preview pass (4 spp)
- **AND** predicts confidence map via JEPA
- **AND** renders targeted pass (128 spp)
- **AND** merges based on confidence

#### Scenario: Select multi-resolution mode

- **WHEN** user sets mode=3
- **THEN** Omen renders low-res high-quality (25%, 256 spp)
- **AND** renders high-res noisy (100%, 4 spp)
- **AND** merges via JEPA scene-guided upsampling

### Requirement: Scene graph extraction

Omen SHALL extract structured scene data from Mitsuba Scene object for JEPA conditioning. Extraction SHALL include geometry (vertices, faces, material indices), materials (BSDF parameters), lights (emitter properties), and camera (transform and properties).

#### Scenario: Extract geometry from scene

- **WHEN** scene contains mesh shapes
- **THEN** extractor reads vertex_positions and faces
- **AND** returns structured Geometry objects

#### Scenario: Extract materials from scene

- **WHEN** shapes have BSDF assignments
- **THEN** extractor reads BSDF parameters (diffuse, roughness, metallic)
- **AND** returns structured Material objects

#### Scenario: Extract lights from scene

- **WHEN** scene contains emitters
- **THEN** extractor reads emitter properties (position, intensity, type)
- **AND** returns structured Light objects

### Requirement: JEPA bridge interface

Omen SHALL provide ctypes bridge to load and call Mojo JEPA kernels via C ABI. Bridge SHALL support denoise, predict_confidence, and merge_multires functions.

#### Scenario: Load JEPA library

- **WHEN** Omen initializes with mode > 0
- **THEN** bridge attempts to load omen.so/omen.dll/omen.dylib
- **AND** logs error if load fails
- **AND** falls back to standard path tracing

#### Scenario: Call denoise function

- **WHEN** mode=1 and JEPA library loaded
- **THEN** bridge calls omen_denoise() with scene graph and noisy render
- **AND** returns denoised RGBA buffer

#### Scenario: Call confidence function

- **WHEN** mode=2 and JEPA library loaded
- **THEN** bridge calls omen_predict_confidence() with scene graph and preview
- **AND** returns confidence map [H, W, 1]

#### Scenario: Call multires merge function

- **WHEN** mode=3 and JEPA library loaded
- **THEN** bridge calls omen_merge_multires() with scene graph and both renders
- **AND** returns merged high-resolution buffer

### Requirement: GPU enable parameter

Omen SHALL accept `use_gpu` boolean parameter (default: true) to enable/disable GPU acceleration. Initial implementation SHALL store this parameter for future Mojo kernel integration.

#### Scenario: Enable GPU acceleration

- **WHEN** user sets use_gpu=true
- **THEN** parameter is stored for Mojo kernel dispatcher

#### Scenario: Disable GPU acceleration

- **WHEN** user sets use_gpu=false
- **THEN** parameter is stored and CPU-only rendering is indicated

### Requirement: CLAUDE_POLICY.md compliance

Omen implementation SHALL follow all project policies: absolute imports only, files under 100 lines of executable code, SOLID principles, and Ruff linting compliance.

#### Scenario: Absolute imports

- **WHEN** code imports from omen_integrator package
- **THEN** imports use absolute format: `from omen_integrator.core import render_path_tracer`
- **AND** no relative imports (`from .`) exist

#### Scenario: File size limits

- **WHEN** source files are created
- **THEN** each file contains maximum 100 lines of executable Python
- **AND** files exceeding limit are split into smaller modules

#### Scenario: Ruff compliance

- **WHEN** code is linted
- **THEN** `ruff check --fix` passes without errors
- **AND** `ruff check format` passes without changes
