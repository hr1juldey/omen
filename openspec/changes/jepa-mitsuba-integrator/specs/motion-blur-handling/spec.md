## ADDED Requirements

### Background: Motion Blur Breaks Per-Instant Auxiliary Buffers

Motion blur is the average of samples taken at different shutter times. A single pixel may contain contributions from multiple materials, lights, and geometry at different time instants. Standard auxiliary buffers (albedo, normal, depth, material_id) capture ONE instant — making them ambiguous or misleading under motion blur.

Omen addresses this with:
1. Motion vector AOV (screen-space velocity per pixel)
2. Previous-frame temporal reprojection (warp clean frame N-1 to frame N coordinates)
3. Motion-aware MoE expert routing (4th routing dimension)
4. Shutter-aware auxiliary buffer handling
5. Graceful degradation when motion vectors are unavailable

### Requirement: Motion vector AOV extraction

Omen SHALL read 2D screen-space motion vectors from the render engine's AOV system.

#### Scenario: Read motion vectors from Blender

- **WHEN** denoising a Blender render with motion blur enabled
- **THEN** enable motion vector pass: `scene.render.use_pass_vector = True`
- **AND** read motion vector pass: `(H, W, 2)` — screen-space velocity (dx, dy) per pixel
- **AND** velocity units: pixels per frame (positive = motion in screen space)
- **AND** include in auxiliary buffer stack: albedo(3) + normal(3) + depth(1) + material_id(1) + motion(2) = 10 channels

#### Scenario: Read motion vectors from Mitsuba

- **WHEN** denoising a Mitsuba render with motion blur enabled
- **THEN** configure AOV integrator: `mi.load_dict({'type': 'aov', 'aovs': 'motion:vector,...'})` wrapping the path integrator
- **AND** extract motion vectors from the AOV output

#### Scenario: Handle missing motion vectors

- **WHEN** motion vector pass is not available (render engine doesn't support it, or motion blur disabled)
- **THEN** fill motion vector channels with zeros: `motion_vectors = zeros(H, W, 2)`
- **AND** set `motion_available = False`
- **AND** motion MoE experts are never activated (static-only mode)
- **AND** temporal reprojection is disabled (single-frame denoise only)
- **AND** log: "Motion vectors unavailable — static denoise mode (no temporal reprojection)"

### Requirement: Temporal reprojection of previous clean frame

Omen SHALL warp the previous frame's clean denoised output to the current frame's coordinates using motion vectors, enabling temporal sample accumulation.

#### Scenario: Reproject previous frame

- **WHEN** mode=4 (animation) AND previous clean frame is available AND motion vectors are available
- **THEN** warp previous clean frame: `reprojected = bilinear_warp(prev_clean, motion_vectors)`
- **AND** compute motion coherence per pixel: `coherence = 1.0 - clamp(length(motion_vector) / max_velocity, 0, 1)`
- **AND** compute occlusion mask: `occluded = velocity_discontinuity > threshold` at pixel neighbors
- **AND** compute reprojection weight: `alpha = prev_confidence × coherence × (1 - occluded)`
- **AND** merge: `output = alpha × reprojected + (1 - alpha) × current_noisy`

#### Scenario: Handle occlusion boundaries

- **WHEN** motion vectors show velocity discontinuity at a pixel (foreground moves differently from background)
- **THEN** mark pixel as occluded: `occluded = true`
- **AND** set reprojection weight to 0 for occluded pixels (don't reuse previous frame data)
- **AND** these pixels are routed to occlusion boundary motion expert (Expert 3)

#### Scenario: Handle first frame (no previous frame)

- **WHEN** frame 0 of sequence (no previous clean frame)
- **THEN** skip temporal reprojection entirely
- **AND** use single-frame denoise (same as Mode 1)
- **AND** store denoised frame 0 for use as `prev_clean` in frame 1

#### Scenario: Handle jump cut

- **WHEN** jump cut is detected (translation > 1 unit or rotation > 45 deg between frames)
- **THEN** clear previous frame buffer
- **AND** treat as first frame of new sequence
- **AND** disable temporal reprojection for this frame

### Requirement: Motion-aware MoE expert routing

Omen SHALL add 4 motion experts as a 4th routing dimension to the tile-based MoE system, routed by tile velocity statistics.

#### Scenario: Compute motion statistics per tile

- **WHEN** 8×8 tiles are being fingerprinted for MoE routing
- **THEN** add to tile fingerprint computation (extends 17-dim → 23-dim):
  - `velocity_mean`: mean(motion_vectors) across 64 pixels in tile → 2-dim
  - `velocity_var`: variance(motion_vectors) across 64 pixels → 2-dim
  - `velocity_max`: max(||motion_vector||) across 64 pixels → 1-dim
  - `occlusion_fraction`: pixels where velocity discontinuity > threshold → 1-dim
- **AND** total tile fingerprint = 23-dim

#### Scenario: Route tile to motion experts

- **WHEN** tile fingerprint with motion statistics is computed
- **THEN** project via Linear(23, 4) → motion routing scores
- **AND** add auxiliary-loss-free bias
- **AND** select top-1 motion expert per tile
- **AND** 4 motion experts:
  - Expert 0: Static — low velocity variance, high temporal reuse weight
  - Expert 1: Linear motion — uniform velocity, warp + accumulate strategy
  - Expert 2: Fast motion/blur — high velocity, shutter smear, deblur processing
  - Expert 3: Occlusion boundary — velocity discontinuity, inpainting-style, low temporal reuse

#### Scenario: Static tile (no motion)

- **WHEN** tile fingerprint shows velocity_mean ≈ 0, velocity_var ≈ 0
- **THEN** route to Motion Expert 0 (Static)
- **AND** high temporal reuse weight (alpha ≈ 0.7-0.9) from previous frame
- **AND** standard denoise + accumulated temporal data

#### Scenario: Fast motion tile

- **WHEN** tile fingerprint shows velocity_max > threshold, high velocity_var
- **THEN** route to Motion Expert 2 (Fast motion/blur)
- **AND** low temporal reuse weight (alpha ≈ 0.1-0.3)
- **AND** deblur processing — expert learns to sharpen motion-blurred regions
- **AND** rely more on current frame data than previous frame

#### Scenario: Occlusion boundary tile

- **WHEN** tile fingerprint shows occlusion_fraction > 0.2 (velocity discontinuity)
- **THEN** route to Motion Expert 3 (Occlusion boundary)
- **AND** temporal reuse disabled for occluded pixels within tile
- **AND** inpainting-style processing — fill disoccluded regions from current frame only

### Requirement: Shutter-aware auxiliary buffers

Omen SHALL handle the ambiguity of auxiliary buffers under motion blur by computing temporal variance across shutter samples.

#### Scenario: Compute auxiliary buffer temporal variance

- **WHEN** motion blur is enabled AND multi-time auxiliary buffers are available
- **THEN** capture albedo, normal, depth at multiple shutter times (t=0, t=0.5, t=1)
- **AND** compute temporal variance per pixel: `temporal_var = var([albedo_t0, albedo_t1, albedo_t2], axis=time)`
- **AND** high temporal variance indicates motion blur zone (material mixing across time)
- **AND** add temporal variance to tile fingerprint: `aux_temporal_var` = mean variance across tile → 1-dim

#### Scenario: Fall back to single-time auxiliary buffers

- **WHEN** multi-time auxiliary buffers are not available (most renderers)
- **THEN** use single-time auxiliary buffers as-is
- **AND** rely on motion vector statistics (velocity_mean, velocity_var) to detect blur zones instead
- **AND** this is the common case — most renderers don't output multi-time AOVs

### Requirement: Updated tile fingerprint dimension

Omen SHALL extend the tile fingerprint from 17-dim to 23-dim to include motion statistics.

#### Scenario: Full tile fingerprint with motion

- **WHEN** all auxiliary buffers (including motion vectors) are available
- **THEN** tile fingerprint = [
    material_histogram(8) + normal_var(3) + depth_var(1) + edge_density(1) + dominant_mat(1) + mean_albedo(3) +
    velocity_mean(2) + velocity_var(2) + velocity_max(1) + occlusion_frac(1)
  ] = 23-dim
- **AND** all MoE routing projections updated to Linear(23, n_experts)

#### Scenario: Degraded fingerprint without motion

- **WHEN** motion vectors are NOT available
- **THEN** set velocity channels to zeros: `velocity_mean=0, velocity_var=0, velocity_max=0, occlusion_frac=0`
- **AND** fingerprint is still 23-dim (zero-padded)
- **AND** motion experts are never selected (scores ≈ 0 + bias discourages selection)
- **AND** static expert (Expert 0) handles all tiles

### Requirement: Blender-compatible motion vector pass

Omen SHALL use Blender's built-in motion vector pass without requiring custom render engine modifications.

#### Scenario: Enable and read motion vectors from Blender

- **WHEN** Omen is used with Blender and motion blur is enabled
- **THEN** enable programmatically: `scene.render.use_motion_blur = True` AND `scene.render.use_pass_vector = True`
- **AND** read from render result: `render.layers['Vector'].data` → (H, W, 4) containing (speed_x, speed_y, speed_z, speed_w)
- **AND** extract 2D screen-space component: `motion_vectors = vector_pass[:, :, :2]`
- **AND** convert to pixel-per-frame units if needed

#### Scenario: Blender motion blur disabled by user

- **WHEN** user has motion blur turned off in Blender render settings
- **THEN** motion vector pass will be all zeros
- **AND** Omen operates in static denoise mode automatically
- **AND** no warning needed — static mode is normal for non-blurred renders
