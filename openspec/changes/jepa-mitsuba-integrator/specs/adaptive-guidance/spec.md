## ADDED Requirements

### Requirement: Mode 2 adaptive sampling pipeline

Omen SHALL implement a two-pass adaptive rendering pipeline that uses JEPA confidence prediction to allocate samples efficiently. High-confidence pixels use JEPA prediction; low-confidence pixels get path-traced samples.

#### Scenario: Two-pass adaptive render

- **WHEN** `render_adaptive(scene, spp_target=128)` is called
- **THEN** **PASS 1** (preview + confidence):
  - Render at 4spp: `preview = mi.render(scene, sensor=0, spp=4)` → TensorXf `(H, W, 3)`
  - Extract scene graph: `scene_graph = extract_scene_graph(scene)`
  - Add alpha: `rgba = np.concatenate([preview, np.ones((H,W,1))], axis=2)` → `(H, W, 4)`
  - Call `clean_preview, confidence = bridge.predict_confidence(scene_graph, rgba, H, W)`
- **AND** **PASS 2** (high-spp path trace):
  - Render at 128spp: `high_spp = mi.render(scene, sensor=0, spp=128)` → TensorXf `(H, W, 3)`
  - Add alpha: `high_rgba = np.concatenate([high_spp, np.ones((H,W,1))], axis=2)`
- **AND** **MERGE** (confidence-weighted blend):
  - `output = confidence * clean_preview + (1 - confidence) * high_rgba`
  - Where confidence shape `(H,W,1)` broadcasts over 4-channel RGBA
- **AND** return merged output as numpy `(H, W, 4)`

#### Scenario: Handle bridge unavailable

- **WHEN** `bridge.available == False`
- **THEN** skip PASS 1 confidence prediction
- **AND** render at full `spp_target` directly: `mi.render(scene, spp=128)`
- **AND** log "JEPA unavailable, rendering at uniform {spp_target}spp"

#### Scenario: Handle preview render failure

- **WHEN** `mi.render()` raises exception during PASS 1
- **THEN** catch Mitsuba exception
- **AND** log error with exception message
- **AND** fall back to uniform `mi.render(scene, spp=spp_target)` without JEPA

### Requirement: Confidence map properties and pixel classification

Omen SHALL produce a per-pixel confidence map from JEPA inference that indicates prediction reliability. The ConfidenceHead MLP outputs sigmoid activation → [0, 1].

#### Scenario: Confidence map dimensions and range

- **WHEN** `bridge.predict_confidence()` returns a confidence map
- **THEN** shape: `(H, W, 1)` — one float per pixel
- **AND** values in [0.0, 1.0] (sigmoid output from ConfidenceHead)
- **AND** ConfidenceHead architecture: `Linear(192, 96) → SiLU → Linear(96, 48) → SiLU → Linear(48, 1) → Sigmoid`

#### Scenario: Pixel classification by confidence

- **WHEN** confidence map is generated
- **THEN** classify pixels:
  - **High-confidence** (confidence > 0.8): flat surfaces, uniform materials, direct lighting, diffuse regions — JEPA prediction is reliable
  - **Medium-confidence** (0.5 < confidence ≤ 0.8): moderate complexity — blend JEPA + path-traced
  - **Low-confidence** (confidence ≤ 0.5): caustics, subsurface scattering, sharp specular highlights, geometric edges — need full path tracing
- **AND** confidence SHALL correlate with actual prediction error: `pearson(confidence, 1 - abs(error)) > 0.7`

#### Scenario: Expected confidence distribution on Cornell box

- **WHEN** rendering Cornell box at 256×256
- **THEN** ~70% of pixels high-confidence (walls, floor, box surfaces)
- **AND** ~20% medium-confidence (soft shadow edges, indirect lighting regions)
- **AND** ~10% low-confidence (caustic region under light, sharp shadow boundaries)
- **AND** effective sample reduction: ~(0.7 × 4 + 0.3 × 128) / 128 ≈ 5× reduction

### Requirement: Compute effective sample reduction

Omen SHALL measure and report effective sample reduction achieved by adaptive mode compared to uniform sampling.

#### Scenario: Calculate sample reduction ratio

- **WHEN** adaptive render is complete
- **THEN** count pixels by confidence: `n_high = (confidence > 0.8).sum()`, `n_low = (confidence < 0.2).sum()`
- **AND** compute effective total samples: `effective = (total_pixels * 4) + (n_low * (128 - 4))`
- **AND** compute uniform samples: `uniform = total_pixels * 128`
- **AND** reduction = `uniform / effective`
- **AND** target: 4-8× sample reduction
- **AND** log: "Adaptive: {pct_high:.0f}% high-conf, {pct_low:.0f}% low-conf, {reduction:.1f}× sample reduction"

#### Scenario: Time ratio measurement

- **WHEN** benchmarking adaptive vs uniform
- **THEN** measure PASS 1 time: `t1 = preview + confidence_prediction`
- **AND** measure PASS 2 time: `t2 = high_spp_render`
- **AND** measure merge time: `t3 = confidence_weighted_merge`
- **AND** compute speedup: `speedup = uniform_128spp_time / (t1 + t2 + t3)`
- **AND** target: adaptive total < 50% of uniform 128spp time

### Requirement: Quality validation for adaptive mode

Omen SHALL validate adaptive output quality against ground truth.

#### Scenario: SSIM quality check

- **WHEN** 256spp ground truth is available
- **THEN** compute SSIM between adaptive output and 256spp GT
- **AND** assert SSIM > 0.95 (target quality match)
- **AND** compute PSNR: assert > 30dB
- **AND** log: "Adaptive quality: SSIM={ssim:.3f}, PSNR={psnr:.1f}dB"

#### Scenario: Adaptive mode benchmark on Cornell box

- **WHEN** benchmarking Mode 2 on Cornell box at 256×256
- **THEN** render uniform 128spp baseline → record time T_uniform and quality Q_uniform
- **AND** render adaptive mode → record time T_adaptive and quality Q_adaptive
- **AND** verify: `Q_adaptive SSIM > 0.95 * Q_uniform SSIM`
- **AND** verify: `T_adaptive < 0.5 * T_uniform`
- **AND** log comparison:
  ```
  Method         Time    SSIM    Effective SPP
  uniform_128    X.Xs    0.98    128
  adaptive       Y.Ys    0.96    ~20-30
  speedup        X/Y             4-8×
  ```

### Requirement: Self-training for confidence head

Omen SHALL train the confidence head using pixel-wise variance across multiple low-spp renders as ground truth uncertainty labels.

#### Scenario: Generate variance-based training data

- **WHEN** training mode for confidence is enabled
- **THEN** render same scene 8 times at 4spp with different seeds: `mi.render(scene, spp=4, seed=i)` for i in range(8)
- **AND** stack renders: `stack[i]` for each render → shape `(8, H, W, 3)`
- **AND** compute per-pixel variance: `variance = np.var(stack, axis=0)` → `(H, W, 3)`
- **AND** average across RGB channels: `uncertainty = np.mean(variance, axis=2)` → `(H, W)`
- **AND** normalize to [0, 1]: `confidence_label = 1 - (uncertainty / uncertainty.max())`
- **AND** store as training pair: `(noisy_render, confidence_label)`

#### Scenario: Train confidence head

- **WHEN** training pairs (noisy renders + confidence labels) are available
- **THEN** run training loop via Nabla Python API: `optimizer = nb.nn.optim.AdamW(model.trainable_params(), lr=1e-3)`
- **AND** loss: `MSE(predicted_confidence, confidence_label)`
- **AND** train for 100 iterations on Cornell box variance data
- **AND** validate: predicted confidence correlates with actual variance (r > 0.7)
