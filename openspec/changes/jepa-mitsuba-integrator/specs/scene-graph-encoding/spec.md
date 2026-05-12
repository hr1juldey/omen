## ADDED Requirements

### Requirement: Extract geometry from Mitsuba scene

Omen SHALL extract vertex positions, face indices, and material assignments from all mesh shapes in a Mitsuba scene. Extraction SHALL use `mi.Scene.shapes()` to iterate shapes and access `vertex_positions` and `faces` attributes.

#### Scenario: Extract Cornell box geometry

- **WHEN** scene contains Cornell box with 2 boxes (2 meshes total)
- **THEN** extractor returns list of Geometry objects
- **AND** each Geometry contains vertices as Float32 array [N×3]
- **AND** each Geometry contains faces as UInt32 array [M×3]
- **AND** material indices are extracted per face

#### Scenario: Handle empty scene

- **WHEN** scene contains no shapes
- **THEN** extractor returns empty list
- **AND** no error is raised

### Requirement: Extract material parameters from BSDFs

Omen SHALL extract BSDF parameters from materials attached to shapes. Extraction SHALL support `PrincipledBSDF` and `RoughBSDF` types with parameters: diffuse reflectance, roughness, metallic, IOR, transmission weight.

#### Scenario: Extract PrincipledBSDF parameters

- **WHEN** shape has PrincipledBSDF material
- **THEN** extractor reads base_color (RGB), roughness (float), metallic (float)
- **AND** returns Material object with these parameters

#### Scenario: Extract rough plastic material

- **WHEN** shape has RoughBSDF material
- **THEN** extractor reads reflectance (RGB), roughness (float)
- **AND** returns Material object with these parameters

#### Scenario: Handle unknown BSDF type

- **WHEN** shape has BSDF type not supported
- **THEN** extractor logs warning
- **AND** returns Material with default values

### Requirement: Extract light emitters from scene

Omen SHALL extract position, intensity, color, and type from all emitters in the scene. Extraction SHALL support point lights, area lights, and directional lights via `mi.Scene.emitters()`.

#### Scenario: Extract point light from Cornell box

- **WHEN** scene contains point light emitter
- **THEN** extractor reads position (Float32[3]), intensity (float)
- **AND** returns Light object with type=POINT

#### Scenario: Extract area light emitter

- **WHEN** scene contains area light emitter
- **THEN** extractor reads position, normal (Float32[3]), surface area
- **AND** returns Light object with type=AREA

#### Scenario: Handle scene with no emitters

- **WHEN** scene contains no emitters
- **THEN** extractor returns empty list
- **AND** warning is logged (scene may be black)

### Requirement: Extract camera sensor parameters

Omen SHALL extract camera transform, FOV, clip planes, and resolution from the scene's sensor. Extraction SHALL use `mi.Scene.sensors()[0]` to access the active sensor.

#### Scenario: Extract perspective camera

- **WHEN** scene has PerspectiveCamera
- **THEN** extractor reads to_world transform matrix
- **AND** extracts fov_x (float in radians)
- **AND** extracts near_clip and far_clip distances

#### Scenario: Extract sensor resolution

- **WHEN** sensor film has size set
- **THEN** extractor reads width and height in pixels
- **AND** returns Camera object with resolution [W, H]

### Requirement: Encode scene graph as structured tensors

Omen SHALL convert extracted scene data into structured tensor format suitable for Mojo JEPA model. Encoding SHALL flatten geometry to vertex array, material parameters to fixed-size feature vectors, and lights to structured array.

#### Scenario: Encode simple scene

- **WHEN** scene has 1 mesh (100 vertices), 1 material, 1 light
- **THEN** encoder produces geometry tensor [100×3] (vertices)
- **AND** material tensor with 8 features (diffuse RGB, roughness, metallic, IOR, etc.)
- **AND** light tensor with position[3], intensity, type

#### Scenario: Handle variable geometry count

- **WHEN** scene has multiple meshes with different vertex counts
- **THEN** encoder concatenates all vertices into single tensor
- **AND** stores mesh boundaries (start_idx, count) per object
