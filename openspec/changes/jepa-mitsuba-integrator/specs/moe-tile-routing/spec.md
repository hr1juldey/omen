## ADDED Requirements

### Background: Why tile-based, not per-pixel

A single pixel has no meaning. One pixel with material_id=2 could be the center of a chrome ball, the edge of a chrome-glass boundary, or a sub-pixel hair highlight. Without spatial context, MoE routing is guesswork.

Production renderers already compute cryptomatte/object ID passes as REGION data. Omen reuses the same 8×8 Swin window structure for MoE routing — zero extra tiling overhead.

**Per-pixel routing problems:**
- No spatial context → expert sees a single scalar, not a pattern
- Adjacent pixels may route to different experts → visible seam artifacts at material boundaries
- Wastes compute: 64 separate routing decisions per window when 1 tile-level decision suffices

**Tile-based routing advantages:**
- Expert sees full 8×8 spatial structure: edges, gradients, material transitions
- Mixed-material tiles get multiple experts → smooth boundary blending
- One routing decision per 64 tokens → efficient
- Aligns with cryptomatte: material histograms are natural tile-level features

### Requirement: Tile fingerprint computation

Omen SHALL compute a compact fingerprint for each 8×8 tile from auxiliary buffer data. The fingerprint captures the material/light/geometry composition of the tile, NOT individual pixel values.

#### Scenario: Compute fingerprint from 8×8 auxiliary window

- **WHEN** the U-Net bottleneck reshapes feature maps into 8×8 Swin windows
- **THEN** for each window, extract auxiliary channels: albedo(3) + normal(3) + depth(1) + material_id(1) + motion(2) = 10 channels per pixel
- **AND** compute material histogram: count pixels per material_id (0-7) → normalize by 64 → 8-dim vector
- **AND** compute normal variance: `var(normals, axis=spatial)` across 64 pixels → 3-dim (indicates edge/curvature density)
- **AND** compute depth variance: `var(depth, axis=spatial)` → 1-dim (indicates transparency/overlap)
- **AND** compute edge density: fraction of pixels where `||normal_gradient|| > threshold` → 1-dim
- **AND** compute dominant material: `argmax(material_histogram)` → 1-dim
- **AND** compute mean albedo: `mean(albedo, axis=spatial)` → 3-dim
- **AND** compute velocity mean: `mean(motion_vectors, axis=spatial)` across 64 pixels → 2-dim
- **AND** compute velocity variance: `var(motion_vectors, axis=spatial)` across 64 pixels → 2-dim
- **AND** compute velocity max: `max(||motion_vector||)` across 64 pixels → 1-dim
- **AND** compute occlusion fraction: pixels where velocity discontinuity > threshold → 1-dim
- **AND** concatenate: fingerprint = [mat_hist(8) + normal_var(3) + depth_var(1) + edge_density(1) + dominant_mat(1) + mean_albedo(3) + vel_mean(2) + vel_var(2) + vel_max(1) + occ_frac(1)] = 23-dim
- **AND** when motion vectors unavailable: zero-fill velocity/occlusion channels, fingerprint is still 23-dim
- **AND** fingerprint is a FIXED-SIZE vector regardless of tile content

#### Scenario: Fingerprint captures tile semantics

- **WHEN** fingerprint is computed for different tile types
- **THEN** flat diffuse wall tile: material_hist=[0.95,0.05,0,...], normal_var≈0, edge_density≈0, depth_var≈0
- **AND** glass-metal edge tile: material_hist=[0,0.45,0.55,...], normal_var=high, edge_density=0.3
- **AND** hair detail tile: material_hist=[0,...,0.9,...], normal_var=very_high, edge_density=0.8
- **AND** volume/smoke tile: material_hist=[0,...,0.8,...], depth_var=very_high
- **AND** these fingerprints are clearly distinguishable → routing is reliable

### Requirement: Tile-level MoE routing

Omen SHALL route entire 8×8 tiles (all 64 tokens) to expert FFNs based on the tile fingerprint. All tokens in a tile share the same expert selection.

#### Scenario: Route tile to material experts

- **WHEN** tile fingerprint (17-dim) is computed
- **THEN** project fingerprint via learned Linear(23, n_material) → material routing scores
- **AND** add auxiliary-loss-free bias: `scores = linear(fingerprint) + bias`
- **AND** select top-K material experts (K=2 for medium tier, K=3 for high tier)
- **AND** compute softmax weights for selected experts
- **AND** ALL 64 tokens in the tile are processed by the same selected experts
- **AND** combine: `output = shared_expert(tokens) + Σ(weight_i × expert_i(tokens))`

#### Scenario: Route tile to light experts

- **WHEN** tile fingerprint is computed
- **THEN** project fingerprint via learned Linear(23, n_light) → light routing scores
- **AND** select top-1 light expert (light type is usually uniform within 8×8 tile)
- **AND** combine with material expert output

#### Scenario: Route tile to geometry experts

- **WHEN** tile fingerprint is computed
- **THEN** project fingerprint via learned Linear(23, n_geo) → geometry routing scores
- **AND** select top-1 geometry expert
- **AND** combine with material + light expert output

### Requirement: Expert taxonomy

Omen SHALL define fixed expert categories for material, light, and geometry types. These correspond to BSDF/emitter/geometry types available in production renderers.

#### Scenario: Material expert categories

- **WHEN** material experts are initialized
- **THEN** 8 material experts, each specialized for:
  - Expert 0: Diffuse/Lambertian — flat albedo, low noise, easy denoise
  - Expert 1: Glossy/Glass — reflections, refraction, caustics, high noise
  - Expert 2: Metal/Chrome — conductor BSDF, complex specular, colored reflections
  - Expert 3: SSS/Skin — subsurface scattering, translucency, diffusion profiles
  - Expert 4: Volume/Smoke — participating media, heterogeneous, high variance
  - Expert 5: Emissive — area lights, neon, self-illuminating surfaces
  - Expert 6: Hair/Fur — curve primitives, anisotropic highlights, deep shadows
  - Expert 7: Cloth/Fabric — microfiber structure, woven patterns, interreflection

#### Scenario: Light expert categories

- **WHEN** light experts are initialized
- **THEN** 5 light experts, each specialized for:
  - Expert 0: Point/Spot light — local, hard falloff, sharp shadows
  - Expert 1: Area light — soft shadows, rectangular/disk, smooth penumbra
  - Expert 2: Sun/Directional — hard parallel shadows, sky illumination
  - Expert 3: Environment/HDRI — ambient, indirect dome, long-range transport
  - Expert 4: Emissive geometry — mesh lights, emissive surfaces, complex shapes

#### Scenario: Geometry expert categories

- **WHEN** geometry experts are initialized
- **THEN** 5 geometry experts, each specialized for:
  - Expert 0: Flat surfaces — low normal variance, trivial denoise
  - Expert 1: Curved/organic — smooth normal changes, gradient-aware
  - Expert 2: Edges/silhouettes — high normal discontinuity, aliasing, fireflies
  - Expert 3: Fine detail/hair — sub-pixel geometry, anisotropic noise
  - Expert 4: Transparent/overlapping — depth discontinuity, refraction artifacts

MOTION EXPERTS (routed by tile velocity statistics — see motion-blur-handling spec):
  - Expert 0: Static — low velocity variance, high temporal reuse
  - Expert 1: Linear motion — uniform velocity, warp + accumulate
  - Expert 2: Fast motion/blur — high velocity, shutter smear, deblur
  - Expert 3: Occlusion boundary — velocity discontinuity, inpainting-style

### Requirement: Shared expert (always active)

Omen SHALL include 1 shared expert per MoE layer that is always active regardless of routing decision. This provides base denoising capability for all tiles.

#### Scenario: Shared expert processes every tile

- **WHEN** any tile passes through MoE FFN
- **THEN** shared expert processes the tile's 64 tokens unconditionally
- **AND** shared expert output is always included in the final combination
- **AND** shared expert learns: Gaussian noise removal, spatial filtering, universal patterns
- **AND** routed experts add specialized processing ON TOP of shared expert output

### Requirement: Mixed-tile handling at material boundaries

Omen SHALL handle tiles that span material boundaries by activating multiple experts and blending their outputs spatially.

#### Scenario: Tile at glass-metal boundary

- **WHEN** tile fingerprint shows material_hist = {glass: 0.45, metal: 0.55} and high edge_density
- **THEN** activate BOTH glass expert (1) and metal expert (2) with weights [0.45, 0.55]
- **AND** also activate edge geometry expert (2) due to high normal variance
- **AND** each expert processes the FULL 8×8 tile (expert sees transition zone)
- **AND** output = shared_expert(x) + 0.45×glass_expert(x) + 0.55×metal_expert(x) + geo_edge_expert(x)
- **AND** result: smooth blend across boundary, no seam artifacts

#### Scenario: Uniform tile (single material)

- **WHEN** tile fingerprint shows material_hist = {diffuse: 0.98} and low variance
- **THEN** activate diffuse expert (0) with weight ~1.0
- **AND** activate flat geometry expert (0)
- **AND** lightweight routing — mostly shared expert + one specialist
- **AND** fast path for easy regions

### Requirement: Auxiliary-loss-free load balancing (DeepSeek-V3)

Omen SHALL balance expert utilization using per-expert bias vectors that do NOT participate in gradient computation.

#### Scenario: Dynamic bias adjustment during training

- **WHEN** a training step completes
- **THEN** compute per-expert load: `load[i] = count(tiles routed to expert i) / total_tiles`
- **AND** for each expert: if `load[i] > target_load + tolerance`: bias[i] -= 0.001
- **AND** for each expert: if `load[i] < target_load - tolerance`: bias[i] += 0.001
- **AND** bias is added to routing scores BEFORE top-K selection: `scores = W@fingerprint + bias`
- **AND** bias is a plain tensor, NOT a model parameter, NO gradient flows through it
- **AND** zero interference with denoising loss — balancing is orthogonal

### Requirement: Tier-specific MoE configuration

Omen SHALL configure MoE complexity based on model tier.

#### Scenario: Fast tier (4M params) — no MoE

- **WHEN** using Fast tier model
- **THEN** no MoE — pure Swin Transformer bottleneck
- **AND** reason: model too small to benefit from expert specialization
- **AND** all tiles processed by shared FFN (equivalent to 1 shared expert only)

#### Scenario: Medium tier (16M params) — MoE in bottleneck

- **WHEN** using Medium tier model
- **THEN** MoE in U-Net bottleneck only (not in encoder/decoder)
- **AND** 8 material + 5 light + 5 geometry + 4 motion + 1 shared = 23 total experts
- **AND** top-K=2 material experts, top-1 light, top-1 geo, top-1 motion per tile
- **AND** active experts per tile: 2 + 1 + 1 + 1 + 1 shared = 6 active

#### Scenario: High tier (64M params) — MoE in bottleneck and encoder

- **WHEN** using High tier model
- **THEN** MoE in U-Net bottleneck AND encoder path
- **AND** 8 material + 5 light + 5 geometry + 4 motion + 1 shared = 23 total experts
- **AND** top-K=3 material experts, top-1 light, top-1 geo, top-1 motion per tile
- **AND** active experts per tile: 3 + 1 + 1 + 1 + 1 shared = 7 active
- **AND** encoder MoE uses smaller expert FFNs (fewer params per expert)

### Requirement: Blender-compatible auxiliary buffers

Omen SHALL use auxiliary buffers that are already available from Blender/Cycles renders via standard AOV (Arbitrary Output Variables). No custom render passes required.

#### Scenario: Read material IDs from Cryptomatte AOV

- **WHEN** denoising a Blender render with Omen
- **THEN** read Cryptomatte pass from Blender's AOV system — provides per-pixel object/material ID
- **AND** use as material_id channel in tile fingerprint computation
- **AND** no custom plugin needed — Cryptomatte is a standard Blender feature

#### Scenario: Read normals and depth from standard passes

- **WHEN** auxiliary buffers are needed for tile fingerprint
- **THEN** read normal pass from Blender's built-in Normal render pass (3 channels)
- **AND** read depth/Z pass from Blender's built-in Depth render pass (1 channel)
- **AND** read albedo from Blender's built-in Diffuse Color pass (3 channels)
- **AND** all standard AOVs — no custom render engine modification required
