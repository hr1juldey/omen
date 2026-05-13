## ADDED Requirements

### Requirement: MitsubaBackend implements Backend ABC
The system SHALL provide a `MitsubaBackend` class that implements the `Backend` abstract interface with `load_scene()`, `render()`, and `get_aov_buffers()` methods.

#### Scenario: Backend instantiation
- **WHEN** MitsubaBackend is instantiated
- **THEN** it holds a reference to a mitsuba scene and sensor configuration

### Requirement: Load scene from numpy arrays
MitsubaBackend SHALL construct a mitsuba Scene from numpy arrays (vertices, faces, camera params, light params) without any file I/O or serialization.

#### Scenario: Load simple mesh scene
- **WHEN** load_scene() is called with vertex positions, face indices, camera matrix, and light data
- **THEN** a mi.Scene object is created that can be rendered

### Requirement: Render produces AOV buffers
MitsubaBackend.render() SHALL produce noisy AOV buffers (color, albedo, normal, depth) as numpy arrays suitable for Mojo denoising.

#### Scenario: Render with 4 SPP
- **WHEN** render() is called with spp=4
- **THEN** noisy color buffer and auxiliary AOV buffers are returned as numpy float32 arrays

### Requirement: Use existing omen_integrator
MitsubaBackend SHALL use the registered "omen" integrator from `src/omen_integrator/` for path tracing with NEE and BSDF sampling.

#### Scenario: Integrator registration
- **WHEN** MitsubaBackend initializes
- **THEN** omen_integrator.register() is called to register the custom integrator with mitsuba
