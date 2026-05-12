## ADDED Requirements

### Requirement: Render preview pass for confidence prediction

Omen SHALL render quick preview pass at low samples per pixel (4 spp) using Mitsuba's path tracer. Preview render SHALL use standard `mi.render()` with integrator="path" and spp parameter.

#### Scenario: Render Cornell box preview

- **WHEN** mode=2 (adaptive) and first pass is requested
- **THEN** system calls `mi.render(scene, spp=4, seed=fixed)`
- **AND** stores result as preview_tensor [H, W, 4]
- **AND** extract scene graph for JEPA conditioning

#### Scenario: Handle preview render failure

- **WHEN** mi.render() fails during preview pass
- **THEN** system logs error with Mitsuba exception message
- **AND** falls back to standard path tracing without JEPA

### Requirement: Predict per-pixel confidence using JEPA

Omen SHALL invoke JEPA model to generate per-pixel confidence map and denoised preview. Confidence SHALL indicate pixel difficulty: 0=uncertain (needs full sampling), 1=confident (JEPA prediction sufficient).

#### Scenario: Generate confidence map from preview

- **WHEN** preview render (4 spp) and scene graph are available
- **THEN** call `omen_predict_confidence` with both inputs
- **AND** receive confidence_map [H, W, 1]
- **AND** receive denoised_preview [H, W, 4]
- **AND** verify confidence values are in [0, 1] range

#### Scenario: Classify pixels by confidence

- **WHEN** confidence map is generated
- **THEN** high-confidence pixels: confidence > 0.8 (flat surfaces, direct lighting)
- **AND** medium-confidence pixels: 0.5 < confidence ≤ 0.8 (moderate complexity)
- **AND** low-confidence pixels: confidence ≤ 0.5 (caustics, SSS, sharp highlights)

### Requirement: Render targeted high-spp pass

Omen SHALL render second pass at high samples per pixel (128 spp) using Mitsuba's path tracer. High-spp pass SHALL be full-frame render (not tile-based due to Mitsuba Python API limitations).

#### Scenario: Render high-spp pass

- **WHEN** confidence map is available and second pass is requested
- **THEN** system calls `mi.render(scene, spp=128, seed=fixed)`
- **AND** stores result as high_spp_tensor [H, W, 4]
- **AND** uses same seed as preview for consistency

#### Scenario: Handle high-spp render failure

- **WHEN** mi.render() fails during high-spp pass
- **THEN** system logs error message
- **AND** returns denoised preview from PASS 1 (best effort)

### Requirement: Merge passes based on confidence

Omen SHALL combine preview pass (JEPA-predicted) and high-spp pass (path-traced) using confidence map as weights. High-confidence pixels use JEPA output, low-confidence pixels use path-traced output.

#### Scenario: Merge adaptive passes

- **WHEN** both preview and high_spp renders are available
- **THEN** create output tensor [H, W, 4]
- **AND** for each pixel:
  - If confidence > 0.8: use JEPA-predicted pixel
  - If confidence ≤ 0.5: use high-spp path-traced pixel
  - Else: blend 50% JEPA + 50% path-traced
- **AND** return merged result

#### Scenario: Validate sample reduction

- **WHEN** mode=2 adaptive render completes
- **THEN** system calculates effective samples saved
- **AND** verifies: (high_conf_pixels × 4 + low_conf_pixels × 128) < (total_pixels × 64)
- **AND** logs sample reduction ratio (target: 4-8x)

### Requirement: Self-training on confidence prediction

Omen SHALL train JEPA confidence head using variance across multiple low-spp renders as ground truth for uncertainty. Training SHALL render same frame 8 times at 4 spp and compute pixel-wise variance.

#### Scenario: Generate training data for confidence

- **WHEN** training mode is enabled for confidence head
- **THEN** render scene 8 times at 4 spp with different seeds
- **AND** compute variance across renders per pixel [H, W, 1]
- **AND** normalize variance to [0, 1] range (low variance = high confidence)
- **AND** store as confidence_labels for training

#### Scenario: Train confidence head

- **WHEN** training data (noisy renders + confidence labels) is available
- **THEN** call JEPA training function with batch of samples
- **AND** update confidence head weights
- **AND** log training loss (MSE between predicted and target confidence)
