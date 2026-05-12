## ADDED Requirements

### Requirement: Mode 3 multi-resolution render pipeline

Omen SHALL implement a two-pass multi-resolution pipeline that renders low-res high-quality + high-res noisy, then merges using JEPA with scene graph guidance. Target: 8-16× speedup vs uniform 256spp.

#### Scenario: PASS 1 — Render low-res high-quality

- **WHEN** `render_multires(scene, scale=4)` is called
- **THEN** modify sensor film size to `[H//scale, W//scale]` = `[H//4, W//4]`:
  ```python
  params = mi.traverse(sensor)
  params['film.size'] = [H // scale, W // scale]
  params.update()
  ```
- **AND** render at 256spp: `low_res = mi.render(scene, spp=256)` → `(H//4, W//4, 3)`
- **AND** memory usage: ~6.25% of full resolution (pixels = (1/4)²)
- **AND** render time: ~16× faster than full res at 256spp (fewer pixels + same spp)
- **AND** store as `low_res_rgba` with alpha channel → `(H//4, W//4, 4)`

#### Scenario: PASS 2 — Render high-res noisy

- **WHEN** PASS 1 completes
- **THEN** restore sensor film size to `[H, W]`:
  ```python
  params['film.size'] = [H, W]
  params.update()
  ```
- **AND** render at 4spp: `high_res = mi.render(scene, spp=4)` → `(H, W, 3)`
- **AND** store as `high_res_rgba` → `(H, W, 4)`
- **AND** this pass captures full geometric detail (edges, thin geometry) but with Monte Carlo noise

#### Scenario: JEPA merge with scene graph guidance

- **WHEN** both `low_res_rgba (H//4, W//4, 4)` and `high_res_rgba (H, W, 4)` are available
- **THEN** extract scene graph: `scene_graph = extract_scene_graph(scene)`
- **AND** call `merged = bridge.merge_multires(scene_graph, low_res_rgba, high_res_rgba, scale=4)`
- **AND** internally Mojo kernel:
  - Upsamples low-res to full res using bilinear interpolation (base color)
  - Extracts high-frequency detail from high-res noisy pass (edges, geometry)
  - Uses scene graph geometry edges to guide where to preserve detail vs smooth
  - Uses material boundaries to prevent texture bleeding across surfaces
- **AND** output: `merged (H, W, 4)` — clean color from low-res, sharp edges from high-res

### Requirement: Geometry-aware upsampling in Mojo kernel

Omen SHALL implement scene-graph-guided merge in Mojo GPU kernels using TileTensor. The merge kernel SHALL use exact geometry edges and material boundaries from scene graph to prevent DLSS-style hallucination.

#### Scenario: Merge kernel launch

- **WHEN** `omen_merge_multires` is called
- **THEN** create `DeviceContext()`
- **AND** zero-copy wrap low-res and high-res buffers as `DeviceBuffer(owning=False)`
- **AND** create TileTensors:
  ```mojo
  comptime high_layout = row_major[H, W, 4]()
  comptime low_layout = row_major[H//4, W//4, 4]()
  ```
- **AND** bind kernel: `comptime kernel = merge_kernel[type_of(high_layout), type_of(low_layout)]`
- **AND** launch with 2D grid: `grid_dim=(ceildiv(W, 16), ceildiv(H, 16)), block_dim=(16, 16)`
- **AND** `ctx.synchronize()` then copy output to host

#### Scenario: Edge-aware merge logic per pixel

- **WHEN** merge kernel processes pixel (row, col) in the high-res output
- **THEN** compute corresponding low-res position: `lr_row = row / scale, lr_col = col / scale`
- **AND** bilinear sample low-res: `low_color = bilinear(low_res, lr_row, lr_col)`
- **AND** read high-res noisy: `high_color = high_res[row, col]`
- **AND** check scene graph: is this pixel near a geometry edge or material boundary?
  - If near edge: weight toward `high_color` (preserve detail)
  - If flat region: weight toward `low_color` (smooth, converged)
- **AND** blend: `output[row, col] = edge_weight * high_color + (1 - edge_weight) * low_color`

#### Scenario: No hallucination artifacts

- **WHEN** merge completes
- **THEN** verify edges are sharp (no bilinear blur across geometry boundaries)
- **AND** verify no invented textures (DLSS-style hallucination)
- **AND** verify material boundaries preserved (flat wall doesn't bleed into adjacent glass)
- **AND** verify: PSNR > 30dB vs 256spp ground truth at full resolution

### Requirement: Validate speedup from multi-resolution

Omen SHALL measure effective speedup and quality of multi-resolution rendering.

#### Scenario: Calculate effective speedup

- **WHEN** mode=3 render completes
- **THEN** measure time for PASS 1 (low-res 256spp): `t_low`
- **AND** measure time for PASS 2 (high-res 4spp): `t_high`
- **AND** measure time for JEPA merge: `t_merge`
- **AND** measure time for uniform 256spp at full res: `t_uniform`
- **AND** compute speedup: `t_uniform / (t_low + t_high + t_merge)`
- **AND** target: 8-16× speedup (PASS 1 is ~16× faster due to 6.25% pixels, PASS 2 is ~64× faster due to 4spp)

#### Scenario: Quality vs ground truth

- **WHEN** 256spp full-res ground truth is available
- **THEN** compute SSIM between merged output and GT
- **AND** compute PSNR between merged output and GT
- **AND** target: SSIM > 0.92, PSNR > 30dB
- **AND** log: "Multires: {speedup:.1f}× speedup, SSIM={ssim:.3f}, PSNR={psnr:.1f}dB"

### Requirement: Self-training for multi-res merge

Omen SHALL train merge model using triplets of (low-res clean, high-res noisy, full-res ground truth).

#### Scenario: Generate multi-res training triplets

- **WHEN** training mode for multires is enabled
- **THEN** render at 25% res 256spp → `low_res_clean` (input A)
- **AND** render at 100% res 4spp → `high_res_noisy` (input B)
- **AND** render at 100% res 256spp → `ground_truth` (target)
- **AND** store triplet `(low_res_clean, high_res_noisy, ground_truth)`

#### Scenario: Train merge model

- **WHEN** training triplets are available
- **THEN** pass to Mojo via `omen_train_step` C ABI
- **AND** loss: `L1(merge(low_res, high_res, scene_graph), ground_truth)`
- **AND** train for 100 iterations on Cornell box
- **AND** validate: PSNR > 30dB on held-out views
