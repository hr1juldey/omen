## ADDED Requirements

### Requirement: Session orchestrates render pipeline
The system SHALL provide an `OmenSession` class that orchestrates the full render pipeline: sync scene → backend render → Mojo denoise → pixel output. The session SHALL coordinate between depsgraph sync, path tracer backend, and Mojo kernel execution.

#### Scenario: Full render pipeline
- **WHEN** OmenSession.render() is called with a depsgraph
- **THEN** the session syncs scene data, calls backend.render(), passes AOV buffers to Mojo kernels, and returns clean pixels

### Requirement: Session uses pluggable backend
The session SHALL accept a `Backend` instance and delegate path tracing to it. The session SHALL NOT depend on any specific path tracer implementation.

#### Scenario: Using Mitsuba backend
- **WHEN** session is configured with MitsubaBackend
- **THEN** path tracing is performed by Mitsuba and AOV buffers are returned as numpy arrays

#### Scenario: Backend failure handling
- **WHEN** backend.render() raises an exception
- **THEN** the session logs the error and reports a render failure to Blender without crashing
