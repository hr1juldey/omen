## ADDED Requirements

### Background: Python Dict Features (NOT C Structs)

Since both Mitsuba and Nabla are Python libraries, scene data is passed as Python dicts of numpy/nabla tensors — no C struct packing needed. Variable-length data handled naturally with Python lists.

### Requirement: Extract structured scene features from Mitsuba

Omen SHALL extract geometry, materials, and light data from Mitsuba scenes as Python dicts of numpy arrays.

#### Scenario: Extract Cornell box features

- **WHEN** `extract_scene_features(mi.cornell_box())` is called
- **THEN** the scene has: 5 shapes (floor, ceiling, back wall, red wall, green wall), 2 boxes (tall, short), 1 area light
- **AND** for each shape:
  - `params = mi.traverse(shape)`
  - vertices: `np.array(params['vertex_positions']).reshape(-1, 3)`
  - face indices: `np.array(shape.face_indices(0)).reshape(-1, 3)` if mesh
  - normals: `np.array(params['vertex_normals']).reshape(-1, 3)` if `shape.has_vertex_normals()`
  - material: `shape.bsdf()` → BSDF type + parameters
- **AND** for each emitter:
  - if `emitter.is_environment()`: type=envmap, no position
  - else: `params = mi.traverse(emitter)` → position, intensity, radiance
- **AND** for sensor: `sensor.fov()`, `params['to_world']` (4x4 transform)
- **AND** return dict:
  ```python
  {
      'geometry': np.array,      # (N_total_verts, 6) — pos_xyz + normal_xyz
      'face_material_ids': np.array,  # (N_total_faces,) — material type per face (consumed by MoE tile router)
      'materials': np.array,     # (N_materials, 9) — type_id + 8 params
      'lights': np.array,        # (N_lights, 7) — type + pos + intensity + rgb
      'camera': np.array,        # (16,) — 4x4 transform flattened
      'n_objects': int,
      'n_lights': int,
  }
  ```

#### Scenario: Handle variable mesh sizes

- **WHEN** scene has meshes of different sizes
- **THEN** concatenate all mesh data: `np.concatenate([mesh_a_verts, mesh_b_verts, ...])`
- **AND** track offsets for face_material_ids mapping
- **AND** pad to fixed maximum if needed for batch processing

#### Scenario: Extract BSDF parameters

- **WHEN** shape has a BSDF material
- **THEN** identify type:
  - `RoughdielectricBSDF` → type_id=0, params: [alpha_u, alpha_v, 0, 0, 0, 0, 0, 0]
  - `DiffuseBSDF` → type_id=1, params: [reflectance_r, g, b, 0, 0, 0, 0, 0]
  - `ConductorBSDF` → type_id=2, params: [eta_r, eta_g, eta_b, k_r, k_g, k_b, 0, 0]
  - `DielectricBSDF` → type_id=3, params: [int_ior, ext_ior, 0, 0, 0, 0, 0, 0]
  - Unknown → type_id=-1, params: zeros(8)
- **AND** material feature: `[type_id, *params]` = 9 floats per material

### Requirement: Scene feature dict to Nabla tensors

Omen SHALL convert scene feature dicts to Nabla tensors for model input.

#### Scenario: Prepare scene features for SceneGraphEncoder

- **WHEN** scene features dict is available
- **THEN** convert to Nabla tensors:
  ```python
  features = {
      'geometry': nb.Tensor.from_dlpack(scene_dict['geometry'].astype(np.float32)).cuda(),
      'materials': nb.Tensor.from_dlpack(scene_dict['materials'].astype(np.float32)).cuda(),
      'lights': nb.Tensor.from_dlpack(scene_dict['lights'].astype(np.float32)).cuda(),
      'camera': nb.Tensor.from_dlpack(scene_dict['camera'].astype(np.float32)).cuda(),
  }
  ```
- **AND** pass to SceneGraphEncoder which handles variable sizes internally

### Requirement: Compute scene delta between frames

Omen SHALL compute per-frame scene changes as structured delta tensors.

#### Scenario: Compute camera movement delta

- **WHEN** camera transforms between frames are available
- **THEN** compute: `T_delta = T_frame_N @ np.linalg.inv(T_frame_N-1)`
- **AND** extract translation delta (3 floats) + rotation quaternion delta (4 floats) = 7 floats
- **AND** include in delta tensor

#### Scenario: Detect birth events (new scene elements)

- **WHEN** new emitter or shape appears (not in previous frame)
- **THEN** detect via scene features diff: compare object counts and hashes
- **AND** encode: `[type_id, position_xyz, size_xyz, density]` = 8 floats
- **AND** flag as high-surprise (confidence=0, trigger full path trace)

#### Scenario: Flatten delta for SceneDeltaEncoder

- **WHEN** all per-frame deltas are computed
- **THEN** flatten into delta tensor:
  ```
  [camera(7), objects(N×8), lights(M×7), births(K×8), materials(P×5)]
  ```
- **AND** pad to fixed maximum length with zeros
- **AND** shape: `(1, max_delta_dim)` for SceneDeltaEncoder input
