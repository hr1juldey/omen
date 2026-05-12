## ADDED Requirements

### Requirement: Mode 1 denoiser pipeline

Omen SHALL implement a single-pass denoiser that takes a low-spp Mitsuba render and produces a clean output using JEPA scene-aware inference. Pipeline: `mi.render(spp=4)` -> scene extraction -> DLPack transfer -> Nabla JEPA denoise -> clean RGBA.

#### Scenario: Render and denoise in single pass

- **WHEN** `render_denoiser(scene, spp=4)` is called
- **THEN** render preview: `image = mi.render(scene, sensor=0, spp=4)` -> TensorXf `(H, W, 3)`
- **AND** extract scene graph: `scene_graph = extract_scene_graph(scene)` -> Python dict of numpy arrays
- **AND** add alpha channel: concatenate ones -> `rgba = concat([image, ones(H,W,1)], axis=2)` -> `(H, W, 4)`
- **AND** transfer via DLPack: `nb_rgba = nb.Tensor.from_dlpack(rgba)` (zero-copy on GPU)
- **AND** convert scene graph to Nabla tensors: `nb_scene = {k: nb.ndarray(v) for k, v in scene_graph.items()}`
- **AND** call model: `latent = model.encode(nb_scene, nb_rgba)` then `clean_rgba = model.decode(latent)`
- **AND** convert back: `output = clean_rgba.numpy()` -> `(H, W, 4)`
- **AND** return clean RGBA as numpy array

#### Scenario: Handle model unavailable

- **WHEN** `model.available == False` (Nabla not installed or model not loaded)
- **THEN** return raw `mi.render()` output unchanged
- **AND** log: "JEPA unavailable, returning raw render"

#### Scenario: Denoise Cornell box

- **WHEN** denoising `mi.cornell_box()` at 256x256 with 4spp
- **THEN** 4spp render completes in <200ms
- **AND** JEPA denoise completes in <100ms (GPU)
- **AND** total pipeline <300ms
- **AND** output SSIM > 0.90 vs 256spp ground truth
- **AND** output has less noise than input (measured by variance)

### Requirement: Quality validation for denoiser

Omen SHALL validate denoiser output quality against ground truth to detect artifacts or hallucinations.

#### Scenario: Compare denoised vs ground truth

- **WHEN** denoiser output is available AND 256spp ground truth is available
- **THEN** compute SSIM between denoised output and 256spp GT
- **AND** compute PSNR between denoised output and 256spp GT
- **AND** assert SSIM > 0.90 and PSNR > 28dB
- **AND** check for hallucination: compute per-pixel difference, max diff < 0.5 in [0,1] space
- **AND** log metrics: "Denoise: SSIM={ssim:.3f}, PSNR={psnr:.1f}dB, max_diff={maxd:.3f}"

#### Scenario: Detect artifacts

- **WHEN** denoised output has unnatural patterns
- **THEN** compute local variance in 8x8 blocks
- **AND** flag blocks with variance > 2x expected (given material properties from scene graph)
- **AND** if flagged blocks > 10% of image: log warning "Denoiser artifacts detected in {pct}% of pixels"
- **AND** return artifact map as optional debug output
