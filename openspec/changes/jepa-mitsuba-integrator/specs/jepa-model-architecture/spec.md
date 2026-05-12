## ADDED Requirements

### Background: Omen as Render Engine Turbocharger

Omen is NOT just a denoiser. It is a scene-aware rendering turbocharger that fights on three fronts:
1. **Denoising** (vs OIDN/OptiX) — scene-blind pixel denoisers, 460K-57M params
2. **Upscaling + Frame Generation** (vs DLSS 4.0) — pixel-level interpolation, NVIDIA-locked
3. **Deterministic Generation** (vs Diffusion models) — text-controlled, stochastic, non-reproducible

JEPA is the ONLY architecture that unifies all three. The core insight: Omen is NOT a latent-to-image generator. It is a **guided restoration network** — a U-Net denoiser conditioned by a JEPA scene understanding system.

Architecture paradigm:
```
JEPA = the BRAIN  (scene understanding + temporal prediction, 1024-dim latent)
U-Net = the HANDS (fast pixel-level denoising, proven like OIDN)
```

### Three Model Tiers

| Tier | Params | Use Case | VRAM at 4K |
|------|--------|----------|------------|
| Fast | 4M | Test the waters, beat OIDN | ~1.2GB |
| Medium | 16M | Kill OptiX with scene awareness | ~2.5GB |
| High | 64M | Palace of mirrors, fog, 20K lights at 4K60 | ~4.5GB+ |

### Requirement: Scene Graph Encoder (0.3M-1M params depending on tier)

Omen SHALL encode render engine scene data as structured embeddings into a 1024-dim latent — NOT as image patches.

#### Scenario: Encode geometry features

- **WHEN** scene geometry is available from render engine
- **THEN** extract vertex positions and face normals
- **AND** project via Linear(6, C_base)
- **AND** aggregate via Mamba SSM (MaIR-style NSS scanning) when vertex count > 10K, else MultiHeadAttention(C_base, num_heads=4)
- **AND** reason: production scenes can have 100K+ vertices making O(n²) attention infeasible; Mamba's O(n) handles this
- **AND** output: geometry embedding

#### Scenario: Encode material features

- **WHEN** shape BSDF parameters are available
- **THEN** project material type + BSDF params via Linear
- **AND** output: material embedding

#### Scenario: Encode light features

- **WHEN** scene emitters are available
- **THEN** extract per-light: type, position, intensity, color
- **AND** project via Linear(7, C_base)

#### Scenario: Aggregate into 1024-dim scene latent

- **WHEN** geometry, material, and light embeddings are available
- **THEN** aggregate: Mamba SSM for large scenes (100K+ tokens), MultiHeadAttention for small scenes (<10K tokens)
- **AND** project to scene_latent shape `(batch, 1024)`

### Requirement: U-Net Denoiser with Scene Conditioning (3M-55M params depending on tier)

Omen SHALL denoise noisy renders using a U-Net conditioned by JEPA scene latent via AdaLN modulation. The U-Net takes noisy pixels + previous clean frame as input — NOT generated from latent.

#### Scenario: Denoise noisy 4K render

- **WHEN** noisy render (4-16 spp) and scene latent (1024-dim) are available
- **THEN** input: concatenate `[noisy_rgba(4), prev_clean(4), albedo(3), normal(3)]` = 14 channels
- **AND** encoder path: multi-scale Conv2d with strided downsampling, skip connections
- **AND** skip connections: MLA-compressed (low-rank projection, 16× reduction) — stored as latent, reconstructed at decoder
- **AND** bottleneck: Swin Transformer blocks (windowed 8×8 attention) with AdaLN conditioned by scene_latent, plus MoE FFN
  - Windowed attention at H/16×W/16 = ~510 windows of 64 tokens = trivial cost even at 4K
  - MoE FFN replaces standard MLP: tile-based routing (8×8 windows) using cryptomatte-style material/light/geo masks (not per-pixel, not per-scene)
  - Tile fingerprint = material histogram + normal variance + depth edge density within each 8×8 window
  - All 64 tokens in a tile routed together — expert sees spatial context, not a meaningless single pixel
  - Shared expert (always active) + routed experts for specific material/light/geo types
  - Auxiliary-loss-free load balancing via per-expert bias (DeepSeek-V3 pattern)
  - Fast tier: pure Swin, no MoE (too small)
  - Medium tier: Swin + MoE (top-2) + Restormer transposed-attention fallback
  - High tier: Swin + MoE (top-3) + Restormer + AdaLN for 4K60 production
- **AND** decoder path: Conv2dTranspose + MLA-reconstructed skip concatenation from encoder
- **AND** output: clean RGBA `(H, W, 4)` in linear HDR space
- **AND** model sizes by tier:
  - Fast: C_base=48, 4 levels, bottleneck=384ch, no MoE
  - Medium: C_base=96, 5 levels, bottleneck=768ch, MoE top-2
  - High: C_base=192, 6 levels, bottleneck=1536ch, MoE top-3

#### Scenario: Handle missing previous frame (first frame)

- **WHEN** no previous clean frame is available (first frame of sequence)
- **THEN** fill with zeros: `prev_clean = zeros(B, H, W, 4)`
- **AND** U-Net operates in single-frame mode (graceful degradation)

#### Scenario: Handle missing auxiliary buffers

- **WHEN** albedo or normal buffers are not available
- **THEN** fill with zeros for missing channels
- **AND** proceed with available inputs

### Requirement: Energy Conservation Loss (0 learnable params)

Omen SHALL enforce physics-based energy conservation during training. The denoiser can redistribute light but cannot create photons.

#### Scenario: Prevent energy gain during denoising

- **WHEN** training on (noisy, ground_truth) pairs
- **THEN** compute per-pixel energy: `E_in = sum(noisy, axis=-1)`, `E_out = sum(denoised, axis=-1)`
- **AND** energy violation: `violation = relu(E_out - E_in - 0.01)` where 0.01 is tolerance
- **AND** loss: `L_energy = mean(violation)`
- **AND** total training loss: `L_total = L_denoise + 0.1 * L_energy + 0.09 * L_sigreg`

### Requirement: ARPredictor with AdaLN-zero (0.5M-6M params depending on tier)

Omen SHALL implement an autoregressive predictor for temporal coherence and frame generation.

#### Scenario: Predict next frame for temporal coherence

- **WHEN** history buffer has H=3 previous latents and current latent is available
- **THEN** concatenate history + current
- **AND** encode scene delta via SceneDeltaEncoder
- **AND** process through hybrid SSM+Attention ConditionalBlock layers with AdaLN-zero modulation
  - Hybrid: Mamba SSM for efficient temporal sequence processing + 1 attention layer for precise recall
  - Reason: MambaIRv2 notes Mamba is "weaker at precise recall/copying" — one attention layer provides exact token access
- **AND** output: predicted latent for next frame
- **AND** model sizes by tier:
  - Fast: 2 layers, 4 heads (pure attention — short sequences)
  - Medium: 4 layers (3 Mamba + 1 Attention), 8 heads
  - High: 8 layers (6 Mamba + 2 Attention), 16 heads

#### Scenario: SceneDeltaEncoder

- **WHEN** per-frame scene changes are available
- **THEN** flatten deltas: camera, objects, lights, births, materials
- **AND** Linear smoothing + MLP
- **AND** output: delta_embedding shape `(1, 1024)`

### Requirement: ConfidenceHead (0.1M-1M params depending on tier)

Omen SHALL produce per-pixel confidence from the scene latent.

#### Scenario: Predict per-pixel confidence

- **WHEN** scene_latent is available
- **THEN** MLP: `Linear(1024, 512) → SiLU → Linear(512, 256) → SiLU → Linear(256, 1) → Sigmoid`
- **AND** output: confidence map shape `(H, W, 1)` with values in [0.0, 1.0]
- **AND** high-confidence (>0.8): flat surfaces, diffuse regions
- **AND** low-confidence (<=0.5): caustics, specular highlights, geometric edges

### Requirement: SIGReg loss — Custom Mojo GPU kernel (0 learnable params)

Omen SHALL implement SIGReg as a custom Mojo GPU kernel via Nabla's `call_custom_kernel()`. Lambda=0.09.

#### Scenario: Compute SIGReg loss

- **WHEN** model embeddings are available during training
- **THEN** pass embeddings to custom Mojo kernel
- **AND** kernel computes Epps-Pulley statistic (17 knots, 1024 projections)
- **AND** SIGReg has ZERO learnable parameters

### Requirement: MLA skip connection compression (from DeepSeek-V2)

Omen SHALL compress U-Net skip connections using MLA-style low-rank projections. Reduces 4K VRAM from ~6GB to ~375MB.

#### Scenario: Compress encoder feature for skip storage

- **WHEN** U-Net encoder produces feature map at level L
- **THEN** compress via `W_down`: `latent = Linear(C, C/16)(feature)` — store `latent` instead of full feature
- **AND** at decoder: reconstruct via `W_up`: `feature = Linear(C/16, C)(latent)` — use reconstructed feature for skip concat
- **AND** compression ratio: 16× per level
- **AND** memory at 4K (High tier): Level 0 = 200MB (was 3.17GB), Level 1 = 100MB (was 1.59GB), total ~375MB (was ~6GB)
- **AND** learnable projections (W_down, W_up) trained end-to-end with the U-Net

#### Scenario: Handle edge regions where compression may lose detail

- **WHEN** normal buffer shows high discontinuity (edges, silhouettes)
- **THEN** optionally store full-resolution features for edge-adjacent tiles
- **AND** use compressed features only for smooth regions (flat surfaces, diffuse areas)

### Requirement: Material/Light/Geometry-aware MoE — Tile-based Routing with Cryptomatte Masks

Omen SHALL use tile-based Mixture-of-Experts routing based on material type, light type, and geometry type — NOT per scene type and NOT per individual pixel. A single pixel has no meaning; routing needs spatial context. Each 8×8 Swin window (64 tokens) is routed as a unit using a tile fingerprint computed from cryptomatte-style auxiliary buffer histograms.

#### Scenario: Compute tile fingerprint and route to experts

- **WHEN** U-Net bottleneck processes an 8×8 Swin window (64 tokens)
- **THEN** compute tile fingerprint from the 64 tokens' auxiliary buffers:
  - Material histogram: count of each material_id within the tile (cryptomatte-style) → 8-dim
  - Normal variance across tile: measures edge/curvature density → 3-dim
  - Depth variance across tile: measures transparency/overlap → 1-dim
  - Edge density: fraction of pixels with high normal discontinuity → 1-dim
  - Dominant material and mean albedo → 4-dim
- **AND** route via learned projection on tile fingerprint (NOT per-pixel) to expert scores per category:
  - 8 material experts: diffuse, glossy/glass, metal, SSS/skin, volume/smoke, emissive, hair/fur, cloth
  - 5 light experts: point/spot, area, sun/directional, environment/HDRI, emissive geometry
  - 5 geometry experts: flat, curved/organic, edges/silhouettes, fine detail/hair, transparent
- **AND** route ALL 64 tokens in the tile together to the selected experts
- **AND** activate top-K experts per category plus 1 shared expert (always active, base denoising)
- **AND** combine: `output = shared_expert(x) + Σ(weight_i × expert_i(x))`

#### Scenario: Handle mixed-material tiles at boundaries

- **WHEN** an 8×8 tile contains multiple material types (e.g., metal-glass edge)
- **THEN** tile fingerprint shows mixed histogram → multiple experts activated
- **AND** both metal and glass experts process the full tile → expert sees spatial transition
- **AND** output is weighted blend of all activated experts → smooth boundary, no seam artifacts
- **AND** contrast with per-pixel routing: adjacent pixels could route to different experts causing visible seams

#### Scenario: Auxiliary-loss-free load balancing

- **WHEN** MoE experts are being trained
- **THEN** maintain per-expert bias vector (NOT a loss term — no gradient)
- **AND** after each training step: if expert overloaded → bias[expert] -= 0.001, if underloaded → bias[expert] += 0.001
- **AND** bias added to routing scores: `scores = route_proj(tile_fingerprint) + bias`
- **AND** zero interference with denoising quality — balancing is orthogonal to training loss
