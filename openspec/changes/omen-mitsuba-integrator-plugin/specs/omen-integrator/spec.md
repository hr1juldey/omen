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
