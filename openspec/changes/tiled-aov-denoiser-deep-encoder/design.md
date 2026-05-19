## Context

Omen's existing GPU training tests process full-resolution images through conv2d_safe, causing OOM at 512x512+ (9.9GB VRAM measured). The scene encoder is a 2-layer MLP (18→32→128, ~3K params) that mean-pools all scene entities — losing non-local light transport information. Research from LeWM (LeCun 2026), D-JEPA (ICLR 2025), I-JEPA (CVPR 2023), and FiLM (AAAI 2018) validates the architectural choices in this design.

Hardware constraint: RTX 3060 12GB VRAM, 32GB system RAM. Target: 256x256 tiles @ 128 channels = ~2GB VRAM per tile.

## Goals / Non-Goals

**Goals:**
- Prove tiled 256x256 AOV denoising works end-to-end with loss convergence to zero
- Validate deep (12-layer) residual scene encoder produces meaningful scene representations
- Validate FiLM conditioning from scene_latent at every conv layer
- Multi-term loss: MSE + SIGReg + energy conservation physics loss
- Decode and visualize GT / noisy / denoised output
- Scale to arbitrary resolution by tiling (512, 1024, 2048+)
- Sustained training mode for long-term stability testing

**Non-Goals:**
- Modifying any src/omen/ code (test file only)
- Full U-Net decoder (encoder pipeline + latent loss only)
- Production inference pipeline
- Real-time denoising performance

## Decisions

### D1: Tile size = 256x256 with 16px overlap

**Choice**: 256x256 tiles with 16px overlap per side (288x288 input, 256x256 output).

**Rationale**: At 256x256 with 128 channels, VRAM is ~2GB (measured). Overlap handles boundary effects — with 2 stride-2 convs, receptive field is ~14px, so 16px overlap is sufficient. Each 1024x1024 image = 16 tiles (4x4 grid).

**Alternative considered**: 128x128 tiles — more tiles, more overhead, less spatial context per tile. 512x512 tiles — 6GB VRAM, too tight for 12GB card with graph overhead.

### D2: Scene encoder = 12-layer residual MLP (18→128→...→128)

**Choice**: Linear(18, 128) → 10× ResBlock(128→128 + silu + skip) → Linear(128, 128). ~250K params.

**Rationale**: ImageNet showed deeper = more semantic abstraction. Same principle: 12 layers let the encoder learn hierarchical scene representations:
- Layers 1-3: raw features (light positions, material colors, bbox)
- Layers 4-6: pairwise relationships (light-material interactions)
- Layers 7-9: global properties (scene complexity, dominant transport)
- Layers 10-12: scene signature (unique noise fingerprint)

Overfitting is DESIRED — per-user pretraining means the encoder should memorize scene noise characteristics. The 128-dim bottleneck prevents memorizing pixel-level noise (only scene-level properties pass through).

**Alternative considered**: Per-entity transformer (encode each light/material separately, then cross-attend) — too complex for nabla's graph engine, would blow compile time.

### D3: FiLM conditioning at every conv layer

**Choice**: After each conv2d_safe, before activation: `output = γ * conv_out + β` where γ,β = Linear(scene_latent).

**Rationale**: LeWM uses AdaLN (a FiLM variant) at every transformer layer for action conditioning. StyleGAN uses AdaIN for style injection. The Distill survey confirms feature-wise transformations compound into meaningful modulations across many domains. For nabla: FiLM is just matmul + elementwise ops — very stable, no graph complexity issues.

**Alternative considered**: Concat injection (append scene_latent as extra channels) — simpler but less parameter-efficient, requires changing conv channel counts, less principled.

### D4: Tile position encoding = 2 sin/cos channels

**Choice**: Append `sin(2π*x/W)` and `sin(2π*y/H)` to AOV input (10→12ch). Normalized to [0,1] based on tile position in the full image grid.

**Rationale**: Center tiles see different indirect lighting than edge tiles. 2 channels is negligible cost. Sin encoding is smooth and differentiable.

### D5: Multi-term loss

**Choice**: `total = L_mse + λ_sigreg * L_sigreg + λ_energy * L_energy`

Where:
- `L_mse = mean(square(fused_latent - target_latent))` — JEPA prediction loss
- `L_sigreg = -mean(log(std(fused_latent) + eps))` — variance regularization from omen's sigreg_fn
- `L_energy = mean(square(sum(abs(render_latent)) - sum(abs(target_latent))))` — energy conservation: render latent energy should match target energy. Prevents the encoder from learning trivially scaled representations.

**Rationale**: LeWM uses only 2 terms (MSE + SIGReg) for robot control. Omen needs the energy term because raytracing follows energy conservation laws — the denoised result should preserve total scene energy.

**Alternative considered**: Full SIGReg with Epps-Pulley statistic — overkill for a test, simple variance regularization suffices.

### D6: Architecture data flow

```
Scene features (18d) → 12-layer MLP → scene_latent (128d)  [RUNS ONCE]
                                                    │
                                        ┌───────────┤ FiLM generators
                                        │           │
AOV tile 256×256×12 (+pos)               │           │
    │                                   │           │
Conv1(12→128, stride=2) ── FiLM(γ1,β1)─┘           │
    │ silu                                            │
Conv2(128→128, stride=2) ── FiLM(γ2,β2)─────────────┘
    │ silu
GlobalAvgPool → (128d)
    │
Linear(128→128) → render_latent
    │
CrossAttn: gate = sigmoid(render @ W_g + b_g)
           fused = render + gate * scene_latent
           LayerNorm(fused)
    │
Loss: MSE(fused, target) + SIGReg + EnergyConservation
```

## Risks / Trade-offs

**[12-layer scene encoder JIT compile time]** → First compile may take 5-10 min with 12 extra linear layers + residual connections in the graph. Mitigated by JIT cache persistence (proven: 676s first → 58s cached).

**[Tile boundary artifacts]** → 16px overlap may not be enough for scenes with large-scale caustics. Mitigated by scene_latent providing global context — the network knows about off-screen lights even at tile boundaries.

**[Energy conservation loss scale]** → λ_energy needs tuning. Too high = latent collapses to match energy, too low = no effect. Start with λ_energy = 0.01, tune via grid search.

**[Overfitting to 5 scenes]** → DESIRED for test, but means the model won't generalize to unseen scenes without retraining. This is the intended design — per-user pretraining.

## Open Questions

- Should the overlap region be blended with a cosine window, or is hard crop sufficient?
- What is the optimal λ_energy? Needs empirical tuning during the test.
- Can we add a 3rd conv layer (Conv3: 128→128, stride=1) for more spatial feature extraction without blowing VRAM? Probably yes — conv2d_safe on 64×64×128 with stride=1 is ~19MB im2col.
