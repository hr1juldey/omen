## ADDED Requirements

### Requirement: Sync extracts material parameters from depsgraph
OmenSync SHALL extract material BSDF parameters (base_color RGB, roughness, metallic, emission) from each mesh object's active material in the evaluated depsgraph. The output SHALL be a list of material dicts compatible with SceneGraphEncoder.

#### Scenario: Object with principled BSDF
- **WHEN** a mesh object has a Principled BSDF material with roughness=0.3, metallic=1.0, base_color=[0.9, 0.8, 0.1]
- **THEN** sync returns materials=[{"base_color": [0.9, 0.8, 0.1], "roughness": 0.3, "metallic": 1.0, "emission": [0,0,0]}]

#### Scenario: Object with no material
- **WHEN** a mesh object has no material assigned
- **THEN** a default material {"base_color": [0.8, 0.8, 0.8], "roughness": 0.5, "metallic": 0.0, "emission": [0,0,0]} is used

#### Scenario: Emission material
- **WHEN** a mesh object has an Emission shader with strength=10 and color=[1, 0.9, 0.8]
- **THEN** sync returns emission=[10.0, 9.0, 8.0] in the material dict
