## ADDED Requirements

### Requirement: Decode visualization output
The system SHALL optionally produce side-by-side visualization of GT (clean), noisy (low spp), and denoised images when the --show CLI flag is provided.

#### Scenario: Visualization with --show flag
- **WHEN** training completes with --show flag enabled
- **THEN** a matplotlib figure is saved showing three images side-by-side: GT (spp=64), Noisy (spp=2), and Denoised (model output)

#### Scenario: No visualization without flag
- **WHEN** training runs without --show flag
- **THEN** no matplotlib figures are generated and no display windows are opened

### Requirement: Denoised image reconstruction
The system SHALL reconstruct a full denoised image by stitching decoded tile outputs back into the original image resolution, blending overlap regions.

#### Scenario: 512x512 full image reconstruction
- **WHEN** 4 tiles of 256x256 are processed and stitched
- **THEN** the output is a 512x512 RGB image

#### Scenario: Overlap blending
- **WHEN** adjacent tiles have 16px overlap regions
- **THEN** the overlap SHALL be blended using linear interpolation to avoid seam artifacts

### Requirement: Save visualization to file
The system SHALL save visualization figures to a timestamped file in the logs/ directory.

#### Scenario: Auto-named output file
- **WHEN** --show is enabled and training completes
- **THEN** a file is saved as `logs/tiled_denoise_VISUAL_{timestamp}.png`
