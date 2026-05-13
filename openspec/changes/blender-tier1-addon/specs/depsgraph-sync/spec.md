## ADDED Requirements

### Requirement: Sync mesh vertices from depsgraph
The system SHALL extract mesh vertex positions and face indices from Blender's evaluated depsgraph and convert them to numpy arrays suitable for path tracer consumption.

#### Scenario: Mesh with triangles
- **WHEN** OmenSync processes a depsgraph containing a mesh object with triangle faces
- **THEN** vertex positions are returned as a numpy float32 array of shape (N, 3) and face indices as int32 array

#### Scenario: Mesh with quads
- **WHEN** OmenSync processes a depsgraph containing a mesh with quad faces
- **THEN** quads are triangulated and returned as triangle indices

#### Scenario: Geometry nodes applied
- **WHEN** OmenSync processes a depsgraph where geometry nodes have modified a mesh
- **THEN** the evaluated mesh (post-geometry-nodes) is extracted, not the original mesh

### Requirement: Sync camera from depsgraph
The system SHALL extract camera parameters (projection matrix, view matrix, field of view, resolution) from the depsgraph as numpy arrays.

#### Scenario: Perspective camera
- **WHEN** OmenSync processes a depsgraph with a perspective camera
- **THEN** camera-to-world matrix, fov, and resolution are extracted

### Requirement: Sync lights from depsgraph
The system SHALL extract light parameters (type, color, intensity, position, direction) from the depsgraph.

#### Scenario: Point light
- **WHEN** OmenSync processes a depsgraph with a point light
- **THEN** light position, color, and intensity are extracted as numpy arrays

#### Scenario: No lights
- **WHEN** OmenSync processes a depsgraph with no lights
- **THEN** a default ambient light is used and no crash occurs
