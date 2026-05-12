## ADDED Requirements

### Background: 3D-Aware Architecture (NOT ViT-Tiny)

LeWM uses ViT-Tiny (5.5M params) designed for 2D image classification (ImageNet). Omen works in 3D — Mitsuba provides exact geometry, materials, and light positions. A 2D patch encoder wastes this information.

New architecture: **Scene-Aware Dual Encoder** (~8M total, down from ~20M)
- Scene Graph Encoder (~1M): structured embeddings for known 3D elements
- Render Feature Encoder (~1.5M): Conv2d on noisy RGBA + aux buffers
- Cross-Attention Fusion (~0.5M): scene features guide render features
- ARPredictor (~4M): simplified from 10.8M (4 layers instead of 6)
- All using Nabla Python API (`nb.nn.Module`, `F.scaled_dot_product_attention`)

### Requirement: Scene Graph Encoder (~1M params)

Omen SHALL encode Mitsuba scene data as structured embeddings — NOT as image patches.

#### Scenario: Encode geometry features

- **WHEN** scene geometry is available from Mitsuba
- **THEN** extract vertex positions and face normals: `np.array(params['vertex_positions']).reshape(-1, 3)`
- **AND** project: `geo_latent = Linear(6, 64)(concat(positions, normals))` — 6 input dims (pos_xyz + normal_xyz)
- **AND** aggregate: `geo_attn = MultiHeadAttention(64, num_heads=4)` — self-attention over geometry
- **AND** output: mean-pooled geometry embedding shape `(1, 64)`

#### Scenario: Encode material features

- **WHEN** shape BSDF parameters are available
- **THEN** map material type to embedding: `mat_type_emb = Embedding(num_types, 64)(type_ids)`
- **AND** project BSDF parameters: `mat_proj = Linear(64 + 8, 64)(concat(type_emb, bsdf_params))`
- **AND** output: mean-pooled material embedding shape `(1, 64)`

#### Scenario: Encode light features

- **WHEN** scene emitters are available
- **THEN** extract per-light: `[type_id, pos_xyz, intensity, color_rgb]` = 7 floats
- **AND** project: `light_latent = Linear(7, 64)(light_params)`
- **AND** output: mean-pooled light embedding shape `(1, 64)`

#### Scenario: Aggregate scene features into latent

- **WHEN** geometry, material, and light embeddings are available
- **THEN** concatenate: `all_features = concat([geo_emb, mat_emb, light_emb])` shape `(3, 64)`
- **AND** cross-attention: `aggregated = MultiHeadAttention(64, num_heads=4)(all_features)`
- **AND** project to latent: `scene_latent = Linear(64, 192)(aggregated.mean(axis=1))` shape `(1, 192)`

### Requirement: Render Feature Encoder (~1.5M params)

Omen SHALL encode noisy renders + auxiliary buffers using Conv2d (NOT ViT patches).

#### Scenario: Encode noisy RGBA with auxiliary buffers

- **WHEN** a noisy render is available from Mitsuba
- **THEN** input: noisy RGBA `(H, W, 4)` + depth `(H, W, 1)` + normal `(H, W, 3)` = 8 channels
- **AND** Conv2d stack:
  - `conv1 = Conv2d(8, 32, 3x3, stride=2)` → `(H/2, W/2, 32)` + ReLU
  - `conv2 = Conv2d(32, 64, 3x3, stride=2)` → `(H/4, W/4, 64)` + ReLU
  - `conv3 = Conv2d(64, 128, 3x3, stride=2)` → `(H/8, W/8, 128)` + ReLU
- **AND** global average pool: `(H/8 * W/8, 128)` → `(128,)`
- **AND** project: `render_latent = Linear(128, 192)(pooled)` shape `(1, 192)`

#### Scenario: Handle missing auxiliary buffers

- **WHEN** depth or normal buffers are not available
- **THEN** fill with zeros: `depth = zeros(H, W, 1)`, `normal = zeros(H, W, 3)`
- **AND** proceed with 8 channels total (RGBA + zero-filled aux)

### Requirement: Cross-Attention Fusion (~0.5M params)

Omen SHALL fuse scene and render features using cross-attention.

#### Scenario: Fuse scene and render latents

- **WHEN** both scene_latent `(1, 192)` and render_latent `(1, 192)` are available
- **THEN** cross-attention: render queries scene
  - query = render_latent (what does the render show?)
  - key = scene_latent (what does the scene contain?)
  - value = scene_latent (scene information to inject)
- **AND** `fused = F.scaled_dot_product_attention(render_latent, scene_latent, scene_latent)`
- **AND** output: `combined_latent` shape `(1, 192)`

### Requirement: ARPredictor with AdaLN-zero (~4M params)

Omen SHALL implement an autoregressive predictor using ConditionalBlock transformer layers. Simplified from LeWM's 10.8M (4 layers instead of 6, heads=8 instead of 16).

#### Scenario: Predict next frame latent

- **WHEN** history buffer has H=3 previous latents and current latent is available
- **THEN** concatenate: `input = concat([history[-3:], current_latent])` shape `(1, 4, 192)`
- **AND** encode scene delta: `delta_emb = SceneDeltaEncoder(delta)` shape `(1, 192)`
- **AND** process through 4 ConditionalBlock layers:
  - Each layer: AdaLN-zero modulation via delta_emb
  - `adaLN = Sequential(SiLU, Linear(192, 1152))` → 6 modulation params
  - `modulate(x, shift, scale) = x * (1 + scale) + shift`
  - Gate starts at 0 (identity at init)
- **AND** output: `predicted_latent = norm(x[:, -1])` shape `(1, 192)`

#### Scenario: SceneDeltaEncoder (155K params, from LeWM)

- **WHEN** per-frame scene changes are available
- **THEN** flatten deltas: `[camera(7), objects(N×8), lights(M×7), births(K×8), materials(P×5)]`
- **AND** Conv1d smoothing: `Conv1d(delta_dim, smoothed_dim, kernel_size=1)`
- **AND** MLP: `Linear(smoothed, 768) → SiLU → Linear(768, 192)`
- **AND** output: `delta_embedding` shape `(1, 192)`

### Requirement: SIGReg loss — Custom Mojo GPU kernel (0 learnable params)

Omen SHALL implement SIGReg from LeWM as a custom Mojo GPU kernel via Nabla's `call_custom_kernel()`. Lambda=0.09 from lewm.yaml.

#### Scenario: Compute SIGReg loss

- **WHEN** model embeddings are available during training
- **THEN** pass embeddings to custom Mojo kernel via `UnaryOperation` subclass
- **AND** kernel computes Epps-Pulley statistic:
  - 17 knots on [0, 3]
  - 1024 random projections (sampled once, cached)
  - Gaussian characteristic function: `phi = exp(-t²/2)`
  - Trapezoidal weights
- **AND** total loss: `L = L_pred + 0.09 * L_sigreg` (lambda from lewm.yaml)
- **AND** SIGReg has ZERO learnable parameters

#### Scenario: SIGReg Python wrapper

- **WHEN** SIGRegOp is called in Nabla training loop
- **THEN** class `SIGRegOp(UnaryOperation)` with:
  - `name = "sigreg_kernel"`
  - `kernel()` calls `call_custom_kernel("sigreg_kernel", kernel_dir, embeddings, ...)`
  - `vjp_rule()` provides gradient for autograd integration
- **AND** composes with `nb.grad`, `nb.vmap`, `@nb.compile`

### Requirement: ConfidenceHead (~30K params)

Omen SHALL produce per-pixel confidence from the latent representation.

#### Scenario: Predict per-pixel confidence

- **WHEN** `bridge.predict_confidence()` is called
- **THEN** architecture: `Linear(192, 96) → SiLU → Linear(96, 48) → SiLU → Linear(48, 1) → Sigmoid`
- **AND** output: confidence map shape `(H, W, 1)` with values in [0.0, 1.0]
- **AND** high-confidence (>0.8): flat surfaces, diffuse regions
- **AND** low-confidence (<=0.5): caustics, specular highlights, geometric edges

### Requirement: Decoder (~1M params)

Omen SHALL decode latent representations back to RGBA images.

#### Scenario: Decode latent to RGBA

- **WHEN** predicted_latent `(1, 192)` is available
- **THEN** project: `Linear(192, 128 * (H/8) * (W/8))` → reshape to `(H/8, W/8, 128)`
- **AND** upsample via Conv2dTranspose:
  - `conv_t1 = Conv2dTranspose(128, 64, 3x3, stride=2)` → `(H/4, W/4, 64)` + ReLU
  - `conv_t2 = Conv2dTranspose(64, 32, 3x3, stride=2)` → `(H/2, W/2, 32)` + ReLU
  - `conv_t3 = Conv2dTranspose(32, 4, 3x3, stride=2)` → `(H, W, 4)` + Sigmoid
- **AND** output: RGBA `(H, W, 4)` with values in [0.0, 1.0]
