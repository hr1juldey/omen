## ADDED Requirements

### Requirement: MitsubaBackend builds mi.Scene from numpy only
MitsubaBackend SHALL provide a `build_scene()` method that takes numpy arrays (vertices, faces, camera_matrix, camera_fov, width, height, lights, materials) and returns an `mi.Scene` object. It SHALL NOT render, extract AOVs, or run inference.

#### Scenario: Build scene from depsgraph data
- **WHEN** build_scene() is called with vertices (Nx3), faces (Fx3), camera params, and light list
- **THEN** an mi.Scene object is returned that can be passed to render_denoiser()

#### Scenario: Empty mesh
- **WHEN** build_scene() is called with zero vertices and zero faces
- **THEN** a fallback scene with a single quad is created instead of crashing

### Requirement: MitsubaBackend converts Blender materials to Mitsuba BSDFs
build_scene() SHALL extract material parameters from the materials dict (roughness, metallic, base_color) and assign Mitsuba BSDFs to mesh shapes.

#### Scenario: Principled BSDF material
- **WHEN** materials contain roughness=0.5, metallic=0.0, base_color=[0.8, 0.2, 0.1]
- **THEN** the mesh is assigned a Mitsuba roughdiffuse or principled BSDF with matching parameters

#### Scenario: No materials provided
- **WHEN** materials list is empty
- **THEN** a default grey diffuse BSDF is applied to all shapes
