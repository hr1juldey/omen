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
- **AND** bottleneck: Swin Transformer blocks (windowed 8×8 attention) with AdaLN conditioned by scene_latent
  - Windowed attention at H/16×W/16 = ~510 windows of 64 tokens = trivial cost even at 4K
  - Fast tier: pure Swin (2 blocks, 4 heads)
  - Medium tier: Swin + Restormer transposed-attention fallback for O(N) safety
  - High tier: Swin + Restormer + AdaLN for 4K60 production workloads
- **AND** decoder path: Conv2dTranspose + skip concatenation from encoder
- **AND** output: clean RGBA `(H, W, 4)` in linear HDR space
- **AND** model sizes by tier:
  - Fast: C_base=48, 4 levels, bottleneck=384ch
  - Medium: C_base=96, 5 levels, bottleneck=768ch
  - High: C_base=192, 6 levels, bottleneck=1536ch

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
