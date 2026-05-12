## ADDED Requirements

### Background: Production Training Data

Cornell box is for bootstrap validation only. Omen's base model MUST be trained on diverse production-level scenes to handle real-world complexity: glass caustics, subsurface scattering, volumetrics, hair, complex geometry, and multi-light setups. A native converter transforms `.blend` files into Mitsuba scene dicts for the training pipeline.

### Requirement: Convert Blender scenes to Mitsuba format

Omen SHALL provide a native converter that reads `.blend` files and produces Mitsuba-compatible scene dictionaries suitable for training pair generation.

#### Scenario: Convert .blend file to Mitsuba scene dict

- **WHEN** `convert_blend_to_mitsuba(blend_path)` is called
- **THEN** load Blender file via `bpy` (Blender Python API) in headless mode: `blender --background --python converter.py`
- **AND** iterate all mesh objects: extract vertices, faces, normals, UV coordinates
- **AND** convert Blender materials to Mitsuba BSDFs:
  - `Principled BSDF` -> `mi.PrincipledBSDF` with diffuse, roughness, metallic, specular, transmission, clearcoat, sheen
  - `Glass BSDF` -> `mi.DielectricBSDF` with IOR
  - `Emission` -> `mi.AreaLight` with radiance
  - `Volume` -> `mi.HomogeneousVolume` with absorption/scattering coefficients
- **AND** convert lights:
  - `Point Light` -> `mi.PointLight` with position, intensity, color
  - `Area Light` -> `mi.AreaLight` on emissive mesh
  - `Sun Light` -> `mi.DirectionLight` or `mi.EnvironmentMap`
  - `Spot Light` -> `mi.SpotLight` with cone angle
- **AND** convert camera: `bpy Camera` -> `mi.PerspectiveCamera` with FOV, clip planes, transform
- **AND** return Mitsuba scene dict: `mi.load_dict(scene_dict)`
- **AND** log: "Converted {blend_path}: {N} meshes, {M} materials, {L} lights"

#### Scenario: Handle texture maps

- **WHEN** Blender material references image textures
- **THEN** extract texture file paths from Blender material node tree
- **AND** copy textures to Mitsuba-compatible location
- **AND** create `mi.BitmapTexture` references in scene dict
- **AND** support UV-mapped textures, environment maps, normal maps
- **AND** handle packed textures: export to temporary file

#### Scenario: Handle complex geometry

- **WHEN** Blender scene has subdivision surfaces, modifiers, particles (hair)
- **THEN** apply modifiers before export (subdivision, mirror, boolean)
- **AND** export hair as curve primitives or `mi.Cylinder` segments
- **AND** handle instanced objects: convert to Mitsuba shape groups
- **AND** log geometry stats: "Total: {verts} vertices, {faces} faces, {curves} curves"

#### Scenario: Handle volumetrics

- **WHEN** Blender scene has volume objects (smoke, fire, fog)
- **THEN** extract voxel data from Blender's volume grids
- **AND** convert to `mi.HomogeneousVolume` or `mi.GridVolume` depending on density
- **AND** preserve emission for fire effects
- **AND** flag as high-surprise event type for JEPA training

### Requirement: Batch training pair generation from scene library

Omen SHALL generate training pairs from a library of production scenes, rendering at varied parameters to maximize model generalization.

#### Scenario: Generate training pairs from scene library

- **WHEN** pre-training the base model
- **THEN** iterate over scene library directory containing converted Mitsuba scenes
- **AND** for each scene, generate N training pairs with:
  - Random camera positions (spherical sampling around scene bounds)
  - Random light intensity variations (0.5x to 2.0x)
  - Random material parameter perturbations (roughness +-0.1, color shift +-5%)
  - Random spp pairs: (4, 256), (8, 256), (16, 256), (1, 256)
- **AND** render each pair: `noisy = mi.render(scene, spp=low, seed=s)` then `gt = mi.render(scene, spp=256, seed=s)`
- **AND** extract scene graph for each camera angle
- **AND** store as training sample: `(noisy_rgba, gt_rgba, scene_graph)`

#### Scenario: Scene library categories

- **WHEN** building the training scene library
- **THEN** include diverse scene categories:
  - **Interiors**: rooms, furniture, lighting fixtures (diffuse + specular + glass)
  - **Architecture**: buildings, facades, interiors with sunlight
  - **Products**: glass bottles, metal objects, jewelry (caustics, SSS, metals)
  - **Vehicles**: cars, bikes with paint, chrome, glass
  - **Characters**: skin SSS, hair, cloth, eyes
  - **Nature**: foliage, water, sky, terrain
  - **Volumes**: smoke, fog, fire, clouds
- **AND** target: 50+ scenes across categories for base model training
- **AND** each scene rendered from 20+ camera angles = 1000+ training samples minimum

#### Scenario: Validate training diversity

- **WHEN** training pairs are generated
- **THEN** compute statistics across the dataset:
  - Material type distribution (diffuse, glossy, glass, metal, SSS, volume)
  - Light count distribution (1-20 lights per scene)
  - Geometric complexity (100 to 1M faces)
  - Noise characteristics (caustic regions, shadow edges, flat surfaces)
- **AND** ensure no single category dominates >40% of training data
- **AND** log: "Training set: {N} pairs, {categories} categories, material dist: {dist}"

### Requirement: Train base model on production data

Omen SHALL train the base JEPA model on the full production scene library, not just Cornell box. Cornell box is used only for per-phase validation during training.

#### Scenario: Full pre-training schedule

- **WHEN** training base model for distribution
- **THEN** Phase 1 (denoiser head): 5000 iterations on diverse 4spp+256spp pairs
- **AND** Phase 2 (confidence head): 2000 iterations on variance maps from diverse scenes
- **AND** Phase 3 (multires merge): 2000 iterations on multi-scale pairs from diverse scenes
- **AND** Phase 4 (temporal prediction): 5000 iterations on animation sequences from diverse scenes
- **AND** validate each phase on held-out scenes (NOT in training set)
- **AND** target metrics:
  - Denoiser: SSIM > 0.92 across all scene categories
  - Confidence: correlation > 0.7 with actual variance across all categories
  - Merge: PSNR > 30dB across all categories
  - Temporal: SSIM > 0.85 across all animation types

#### Scenario: Cornell box as validation benchmark

- **WHEN** running phase validation during training
- **THEN** use Cornell box as a standardized validation scene
- **AND** log per-phase metrics on Cornell box for comparison across training runs
- **AND** Cornell box metrics are NOT the only quality gate — production scene metrics matter more

#### Scenario: Export and distribute base model

- **WHEN** base model training completes
- **THEN** save via Nabla `state_dict()` to `base_v0.omen`
- **AND** compute SHA256 checksum
- **AND** create metadata JSON: architecture hash, training config, per-category metrics
- **AND** bundle with Omen distribution or host for download
- **AND** size target: ~30MB (8M params in BF16)
