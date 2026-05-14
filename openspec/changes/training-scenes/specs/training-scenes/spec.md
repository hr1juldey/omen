## ADDED Requirements

### Requirement: Cornell Box scene builder
The system SHALL provide `build_cornell_box()` that returns a `(mi.Scene, scene_graph)` tuple. The scene SHALL contain: a box room (6 walls), red left wall, green right wall, white floor/ceiling/back, a small area light on the ceiling, and two diffuse boxes on the floor. All materials SHALL be diffuse BSDF. The scene_graph SHALL contain geometry vertices (8+ faces), 3 material types (red diffuse, green diffuse, white diffuse), and 1 area light.

#### Scenario: Cornell Box renders without error
- **WHEN** `build_cornell_box()` is called
- **THEN** it SHALL return a valid Mitsuba scene and a scene_graph dict
- **AND** rendering at 64spp SHALL produce an image with visible color bleeding from red/green walls

#### Scenario: Cornell Box scene_graph has correct structure
- **WHEN** `build_cornell_box()` returns `(_, scene_graph)`
- **THEN** `scene_graph["geometry"]["vertices"]` SHALL be a numpy array with shape (N, 3) where N > 0
- **AND** `scene_graph["materials"]["params"]` SHALL have at least 3 rows (red, green, white)
- **AND** `scene_graph["lights"]["params"]` SHALL have exactly 1 row (area light)

### Requirement: Veach Ajar Door scene builder
The system SHALL provide `build_veach_ajar()` that returns a `(mi.Scene, scene_graph)` tuple. The scene SHALL contain: a dark room with a slightly open door letting light in, a glass sphere (dielectric), a metal sphere (conductor), a matte sphere (diffuse), and 3 light sources (point, spot, area from door). The scene_graph SHALL list dielectric, conductor, and diffuse materials plus 3 light types.

#### Scenario: Veach scene renders with multiple light types
- **WHEN** `build_veach_ajar()` is called and rendered
- **THEN** the render SHALL show caustics from the glass sphere
- **AND** metallic reflections from the conductor sphere
- **AND** contributions from all 3 light source types

#### Scenario: Veach scene_graph lists all BSDF types
- **WHEN** `build_veach_ajar()` returns `(_, scene_graph)`
- **THEN** `scene_graph["materials"]["types"]` SHALL contain "dielectric", "conductor", and "diffuse"
- **AND** `scene_graph["lights"]["types"]` SHALL contain "point", "spot", and "area"

### Requirement: Shaderball scene builder
The system SHALL provide `build_shaderball()` that returns a `(mi.Scene, scene_graph)` tuple. The scene SHALL contain: a central sphere displayed on a checkerboard plane, with material variants (conductor, roughconductor, plastic, roughplastic, dielectric) rendered as separate spheres or as a configurable single sphere. The scene_graph SHALL list all 5+ material types.

#### Scenario: Shaderball renders all material types
- **WHEN** `build_shaderball()` is called and rendered at 64spp
- **THEN** the render SHALL show distinct material appearances for each BSDF type
- **AND** no NaN or inf values in the output

#### Scenario: Shaderball scene_graph has comprehensive material coverage
- **WHEN** `build_shaderball()` returns `(_, scene_graph)`
- **THEN** `scene_graph["materials"]["types"]` SHALL contain at least 5 distinct BSDF types
- **AND** the types SHALL include "conductor", "roughconductor", "plastic", "roughplastic", and "dielectric"

### Requirement: Studio Product scene builder
The system SHALL provide `build_studio_product()` that returns a `(mi.Scene, scene_graph)` tuple. The scene SHALL contain: 2-3 product objects (spheres/cylinders) with conductor and roughplastic materials, a ground plane, and 3-point studio lighting (key, fill, rim area lights). No external HDRI files required — use constant environment + area lights.

#### Scenario: Studio scene renders with 3-point lighting
- **WHEN** `build_studio_product()` is called and rendered
- **THEN** the render SHALL show well-lit product objects with key/fill/rim light contributions
- **AND** metallic and plastic materials SHALL be visually distinguishable

#### Scenario: Studio scene_graph lists conductor and roughplastic
- **WHEN** `build_studio_product()` returns `(_, scene_graph)`
- **THEN** `scene_graph["materials"]["types"]` SHALL contain "roughconductor" and "roughplastic"
- **AND** `scene_graph["lights"]["types"]` SHALL contain "area" at least 3 times

### Requirement: Foggy Corridor scene builder
The system SHALL provide `build_foggy_corridor()` that returns a `(mi.Scene, scene_graph)` tuple. The scene SHALL contain: an L-shaped corridor with diffuse walls, a null-BSDF volume boundary containing a homogeneous medium (fog), and 2 lights (point, spot). The scene_graph SHALL include the null BSDF type and volume parameters.

#### Scenario: Foggy corridor renders with volumetric scattering
- **WHEN** `build_foggy_corridor()` is called with `integrator="volpath"` and rendered
- **THEN** the render SHALL show visible fog/light scattering in the corridor
- **AND** no NaN or black pixels from failed volume integration

#### Scenario: Foggy corridor scene_graph includes volume metadata
- **WHEN** `build_foggy_corridor()` returns `(_, scene_graph)`
- **THEN** `scene_graph["materials"]["types"]` SHALL contain "null" (volume boundary)
- **AND** the scene_graph SHALL include a "volume" key with sigma_t and albedo parameters

### Requirement: Training data generator
The system SHALL provide `TrainingDataGenerator` class that generates (noisy, clean) image pairs from any scene builder function. It SHALL accept configurable SPP levels, resolution, and seed. It SHALL render both images with different seeds and return them as numpy arrays in NHWC format.

#### Scenario: Generate a single training pair
- **WHEN** `TrainingDataGenerator.generate_pair(build_cornell_box, noisy_spp=4, clean_spp=256)` is called
- **THEN** it SHALL return `(noisy, clean, scene_graph)` where noisy and clean are numpy arrays of shape (H, W, 3)
- **AND** the clean image SHALL have lower variance (higher quality) than the noisy image

#### Scenario: Generate batch of pairs
- **WHEN** `TrainingDataGenerator.generate_batch(build_cornell_box, count=10, noisy_spp=4, clean_spp=256)` is called
- **THEN** it SHALL return a list of 10 `(noisy, clean, scene_graph)` tuples
- **AND** each pair SHALL use a different random seed

#### Scenario: Reduced resolution for training
- **WHEN** `TrainingDataGenerator` is initialized with `max_resolution=(480, 270)`
- **THEN** all generated images SHALL be at most 480x270 regardless of scene defaults
- **AND** the aspect ratio SHALL be preserved

### Requirement: CLI entry point for scene rendering
The system SHALL provide a CLI via `python -m omen.scenes` that can render any of the 5 benchmark scenes with configurable SPP, resolution, and output path.

#### Scenario: Render Cornell Box via CLI
- **WHEN** `python -m omen.scenes --scene cornell --spp 64 --output cornell.exr` is executed
- **THEN** the system SHALL render the Cornell Box at 64spp and save to `cornell.exr`
- **AND** print the render time and output path

#### Scenario: Generate training pairs via CLI
- **WHEN** `python -m omen.scenes --scene cornell --spp-pair 4,256 --count 5 --output-dir ./data/` is executed
- **THEN** the system SHALL generate 5 training pairs and save them as numpy files
- **AND** print the number of pairs generated and total disk size

#### Scenario: List available scenes
- **WHEN** `python -m omen.scenes --list` is executed
- **THEN** the system SHALL print all 5 scene names with brief descriptions
