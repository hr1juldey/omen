## ADDED Requirements

### Requirement: Session delegates to render_denoiser pipeline
OmenSession.render_scene() SHALL call `src/omen/modes/denoiser.render_denoiser()` which runs the full AI pipeline: Mitsuba render with AOV → scene graph extraction → Mojo GPU kernels → JEPA encode/decode → clean RGBA. The session SHALL NOT implement its own render or denoising logic.

#### Scenario: Full denoise render
- **WHEN** OmenSession.render_scene() is called with a depsgraph and spp=4, tier="medium"
- **THEN** render_denoiser() is called with the constructed mi.Scene and JEPABridge, returning clean RGBA pixels

#### Scenario: JEPABridge not available
- **WHEN** JEPABridge fails to initialize (no nabla/modular)
- **THEN** render_denoiser falls back to returning raw noisy render pixels without JEPA denoising (graceful degradation)

#### Scenario: Mitsuba render fails
- **WHEN** mi.render() raises an exception
- **THEN** session logs the error and returns a black frame instead of crashing

### Requirement: Session creates JEPABridge lazily
OmenSession SHALL create a JEPABridge instance on first render call, not at module import time. Subsequent renders reuse the same bridge.

#### Scenario: First render creates bridge
- **WHEN** render_scene() is called for the first time
- **THEN** JEPABridge() is instantiated and cached

#### Scenario: Subsequent renders reuse bridge
- **WHEN** render_scene() is called again
- **THEN** the cached JEPABridge is reused without re-initialization

### Requirement: mojo_runtime.py deleted
`src/omen_engine/mojo_runtime.py` SHALL be deleted. Mojo kernel loading is handled by `src/omen/kernels/*.py` bridges which are called by the existing pipeline.

#### Scenario: No mojo_runtime import
- **WHEN** omen_engine modules are imported
- **THEN** mojo_runtime is not imported or referenced anywhere
