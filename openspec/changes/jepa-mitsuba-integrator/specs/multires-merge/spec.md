## ADDED Requirements

### Requirement: Render low-resolution high-quality pass

Omen SHALL render scene at reduced resolution (25% scale) with high samples per pixel (256 spp) using Mitsuba's path tracer. Low-res pass SHALL provide converged color but lack high-frequency detail.

#### Scenario: Render low-res Cornell box

- **WHEN** mode=3 (multires) and first pass is requested
- **THEN** set sensor film size to [H/4, W/4]
- **AND** call `mi.render(scene, spp=256)`
- **AND** store result as low_res_tensor [H/4, W/4, 4]
- **AND** verify result is converged (low noise)

#### Scenario: Handle low-res memory efficiency

- **WHEN** rendering at 25% resolution
- **THEN** memory usage is ~6.25% of full resolution
- **AND** render time is ~16× faster than full res at 256 spp
- **AND** logs timing and memory metrics

### Requirement: Render high-resolution noisy pass

Omen SHALL render scene at full resolution (100% scale) with low samples per pixel (4 spp) using Mitsuba's path tracer. High-res pass SHALL provide full geometric detail but with Monte Carlo noise.

#### Scenario: Render high-res noisy Cornell box

- **WHEN** mode=3 (multires) and second pass is requested
- **THEN** set sensor film size to [H, W]
- **AND** call `mi.render(scene, spp=4)`
- **AND** store result as high_res_tensor [H, W, 4]
- **AND** verify result has geometric detail (edges, textures)

#### Scenario: Handle high-res noise pattern

- **WHEN** high-res render at 4 spp completes
- **THEN** result has visible Monte Carlo noise
- **AND** noise is especially noticeable in flat regions
- **AND** detail is preserved in edges and textures

### Requirement: Merge multi-resolution passes with JEPA

Omen SHALL invoke JEPA model to merge low-res clean and high-res noisy renders using scene graph knowledge. Merge SHALL use exact geometry edges (from scene graph) to guide upsampling, avoiding DLSS-style hallucination artifacts.

#### Scenario: Merge multi-resolution inputs

- **WHEN** low_res_high_qual [H/4, W/4, 4] and high_res_noisy [H, W, 4] are available
- **THEN** call `omen_merge_multires` with scene graph
- **AND** pass scale_factor=4 (upsampling ratio)
- **AND** receive output_merged [H, W, 4]
- **AND** verify output has clean color (from low-res) and sharp detail (from high-res)

#### Scenario: Validate edge quality

- **WHEN** multi-resolution merge completes
- **THEN** edges are sharp (no bilinear upsampling blur)
- **AND** no DLSS-style hallucination artifacts (invented textures)
- **AND** material boundaries are preserved (from scene graph knowledge)

### Requirement: Validate speedup from multi-resolution

Omen SHALL measure effective speedup of multi-resolution rendering compared to uniform sampling at target quality. Speedup calculation SHALL compare (25% res 256 spp + 100% res 4 spp) vs (100% res 256 spp).

#### Scenario: Calculate effective speedup

- **WHEN** mode=3 render completes
- **THEN** measure time for PASS 1 (low-res 256 spp)
- **AND** measure time for PASS 2 (high-res 4 spp)
- **AND** measure time for JEPA merge
- **AND** calculate: time_uniform_full / (time_multi_total)
- **AND** verify speedup > 4× (target: 8-16×)

### Requirement: Self-training on multi-resolution pairs

Omen SHALL train JEPA multi-resolution merge head using self-supervised pairs. Training SHALL render same scene at (low_res, high_spp) and (high_res, low_spp) and (high_res, high_spp) as ground truth.

#### Scenario: Generate multi-res training data

- **WHEN** training mode is enabled for multi-resolution
- **THEN** render at 25% res 256 spp (input A)
- **AND** render at 100% res 4 spp (input B)
- **AND** render at 100% res 256 spp (ground truth)
- **AND** store triplet (A, B, GT) as training sample

#### Scenario: Train multi-res merge model

- **WHEN** training triplets (low_res_clean, high_res_noisy, ground_truth) are available
- **THEN** call JEPA training function with batch of samples
- **AND** update merge model weights to minimize L1 loss vs ground truth
- **AND** log training metrics (PSNR, SSIM vs ground truth)
