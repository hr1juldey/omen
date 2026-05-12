# Omen — Research Paper Grade Evaluation & Publication Strategy

**Date**: 2026-05-12
**Status**: Pre-implementation evaluation
**Purpose**: Assess Omen's novelty, identify gaps, plan publication strategy

---

## Table of Contents

1. [Novelty Assessment](#1-novelty-assessment)
2. [Prior Art & Related Work](#2-prior-art--related-work)
3. [Strengths](#3-strengths)
4. [Critical Weaknesses](#4-critical-weaknesses)
5. [Baselines Required](#5-baselines-required)
6. [Ablation Studies](#6-ablation-studies)
7. [Missing Components](#7-missing-components)
8. [Risk Assessment](#8-risk-assessment)
9. [Target Venues](#9-target-venues)
10. [Publication Strategy](#10-publication-strategy)
11. [Training Data Plan](#11-training-data-plan)
12. [Blender Shared Node System Architecture](#12-blender-shared-node-system-architecture)
13. [Omen Blender Plugin Design](#13-omen-blender-plugin-design)
14. [Actionable Next Steps](#14-actionable-next-steps)

---

## 1. Novelty Assessment

### 1.1 Genuinely Novel Contributions

| # | Contribution | Novelty Level | Strength | Rationale |
|---|-------------|---------------|----------|-----------|
| 1 | **Tile-based MoE routing with cryptomatte-style material masks** | **High** | Strong | Nobody has used cryptomatte-style material histograms within 8x8 windows to route denoising experts. MoCE-IR (CVPR 2025) uses complexity-based routing for generic image restoration, but scene-aware material/light/geometry routing for MC denoising is unexplored territory. |
| 2 | **Scene-graph conditioning for denoising** (AdaLN-zero from JEPA world model) | **Medium-High** | Strong | Disney's Deep-Z denoiser (2024) uses kernel prediction, but conditioning denoiser predictions on structured scene graph embeddings (materials, geometry, lights) via JEPA is novel. |
| 3 | **MLA skip compression on U-Net** (from DeepSeek-V2/V3, applied to denoising) | **Medium** | Moderate | Novel application domain. MLA was designed for LLM attention KV caching, not U-Net skip connections. The 16x compression on skip connections (6GB → 375MB at 4K) is an engineering contribution. |
| 4 | **Motion-aware MoE experts** (static/linear/fast/occlusion) | **Medium** | Moderate | Temporal denoising exists (OptiX, OIDN), but routing different expert networks based on per-tile motion statistics is new. 4 motion experts as a routing dimension alongside material/light/geo. |
| 5 | **Dual-pipeline Mitsuba+JEPA architecture** (JEPA = scene understanding brain, U-Net = fast denoising hands) | **Medium** | Moderate | Architecturally interesting and well-motivated paradigm, but the separation itself isn't deeply novel — it's the combination that matters. |
| 6 | **Mojo/Nabla implementation of the entire pipeline** | **Low** | Engineering only | Engineering contribution, not scientific novelty. |

### 1.2 Incremental / Already-Existing Work

| Technique | Prior Art | Status |
|-----------|-----------|--------|
| Kernel-predicting neural denoisers | Disney Deep-Z (2024), KPCN (Bako 2017) | Standard |
| Auxiliary buffer conditioning (albedo, normal, depth) | NVIDIA AFGSA, every production denoiser | Standard |
| MoE for image restoration | MoCE-IR (CVPR 2025), Steered MoE | Exists but not for MC denoising |
| U-Net for denoising | Standard architecture, many papers | Standard |
| Temporal reprojection | DLSS, FSR, OptiX temporal denoiser | Standard |
| FP8 inference | DeepSeek-V3, standard quantization | Standard |
| SIGReg loss | LeWM (Maes et al. 2026, LeCun) | Directly borrowed |
| AdaLN-zero conditioning | DiT, LeWM | Borrowed |

### 1.3 The Novelty Is in the Combination

Individually, each technique exists. The **integration** and the **application domain** (production Monte Carlo denoising with scene understanding) is where the novelty lives. This is a valid research contribution — many top papers combine existing techniques in novel ways for new domains.

---

## 2. Prior Art & Related Work

### 2.1 Neural Monte Carlo Denoising (State of the Art 2024-2025)

| Paper | Year | Venue | Key Technique | Relation to Omen |
|-------|------|-------|---------------|-----------------|
| [Disney Neural Deep-Z Denoiser](https://studios.disneyresearch.com/2024/06/18/neural-denoising-for-deep-z-monte-carlo-renderings/) | 2024 | Disney Research | Kernel-predicting neural denoiser for deep-Z images | Omen differs: scene-graph conditioning + MoE routing vs kernel prediction |
| [Neural Kernel Regression for Consistent MC Denoising](https://dl.acm.org/doi/abs/10.1145/3687949) | 2024 | SIGGRAPH 2024 | Neural kernel regression, consistency guarantees | Omen differs: tile-based MoE + scene awareness |
| Target-Aware Image Denoising for Inverse MC Rendering | 2024 | SIGGRAPH 2024 | Target-aware denoising for inverse rendering | Different problem (inverse rendering) |
| [Neural Bilateral Grid](https://www.semanticscholar.org/paper/Real-time-Monte-Carlo-Denoising-with-the-Neural-Meng-Zheng/f139781dfdf139da37da718ee330faba8add6e62) | 2024 | — | Neural bilateral grid, 1-spp real-time | Omen differs: production quality, not real-time focus |
| [AMD Neural Supersampling + Denoising](https://gpuopen.com/learn/neural_supersampling_and_denoising_for_real-time_path_tracing/) | 2024 | AMD GPUOpen | Joint supersampling + denoising | Industry reference, different use case |
| [Neural Denoising for Spectral MC Rendering](https://diglib.eg.org/bitstreams/124580f6-844c-4f88-b3bd-19af87a3ce48/download) | 2022 | EG | 3-step pipeline with spectral + auxiliary features | Omen extends with scene graph conditioning |
| [Adversarial MC Denoising with Auxiliary Feature Modulation](https://www.researchgate.net/publication/337117567_Adversarial_Monte_Carlo_denoising_with_conditioned_auxiliary_feature_modulation) | 2019 | — | GAN + conditioned auxiliary features | Omen uses MoE routing instead of GAN |
| KPCN (Bako et al.) | 2017 | SIGGRAPH | Kernel-predicting convolutional network | Foundation that Omen builds on |

### 2.2 MoE for Image Restoration

| Paper | Year | Venue | Key Technique |
|-------|------|-------|---------------|
| [MoCE-IR (CVPR 2025)](https://github.com/eduardzamfir/MoCE-IR) | 2025 | CVPR | Complexity-based expert routing for all-in-one image restoration |
| MoEDiff-SR | 2025 | arXiv | Diffusion + MoE for MRI super-resolution |
| N-SMoE | 2024-25 | — | Neural-steered MoE for medical image denoising |
| Steered MoE Regression | 2023 | — | Block-based regression with edge-aware SMoE |
| Spatially-Heterogeneous Distortions (ACCV) | 2020 | ACCV | MoE for restoring heterogeneous distortions |

**Key Gap in Literature**: Nobody has applied MoE routing to **Monte Carlo denoising** with scene-aware features. This is Omen's clearest novelty claim.

### 2.3 JEPA / World Models

| Paper | Year | Key Technique |
|-------|------|---------------|
| LeWM (Maes et al., LeCun) | 2026 | JEPA world model for robotics, 2 losses (prediction + SIGReg) |
| I-JEPA (Assran et al.) | 2023 | Joint-Embedding Predictive Architecture for images |
| V-JEPA (Bardes et al.) | 2024 | Video JEPA |
| D-JEPA | 2024 | Direct predecessor for JEPA denoising |

**Omen's Extension**: Replace robot actions with scene deltas (material/geometry/light changes). Use AdaLN-zero to inject scene context into every transformer layer.

### 2.4 DeepSeek Innovations (Directly Borrowed)

| Technique | Source | Omen Usage |
|-----------|--------|-----------|
| Multi-head Latent Attention (MLA) | DeepSeek-V2/V3 | Skip connection compression (16x) |
| Auxiliary-loss-free MoE balancing | DeepSeek-V3 | Tile-based expert load balancing |
| DualPipe (async pipeline) | DeepSeek-V3 | Overlapping Mitsuba render + JEPA denoise |
| MTP (multi-token prediction) | DeepSeek-V3 | Speculative multi-frame prediction |
| FP8 mixed precision | DeepSeek-V3 | E4M3 forward, BF16 attention softmax |

---

## 3. Strengths

### 3.1 Technical Strengths

1. **Strong technical grounding**: The design shows deep understanding of Mitsuba's differentiable rendering (Dr.Jit autodiff), JEPA architecture (LeWM port), and production rendering needs.

2. **Well-designed MoE taxonomy**: 8 material experts + 5 light experts + 5 geometry experts + 4 motion experts + 1 shared expert = 23 total. Each expert has clear specialization rationale.

3. **23-dimensional tile fingerprint**: Material histogram(8) + normal_var(3) + depth_var(1) + edge_density(1) + dominant_mat(1) + mean_albedo(3) + velocity_mean(2) + velocity_var(2) + velocity_max(1) + occlusion_frac(1) = 23 dims. This is a well-crafted feature vector for routing.

4. **Graceful degradation**: Zero-fill missing AOV channels, shared expert fallback when no motion vectors. Never crash.

5. **Smart scene latent cache**: Two-level hashing (topology_hash for structure, dynamic_hash for values) with smart invalidation rules. Full re-encode on births/material type changes/vertex count changes; incremental delta update for position/intensity changes.

6. **Comprehensive specification**: More detailed than most academic paper supplements. Implementation-ready.

### 3.2 Practical Strengths

1. **Production focus**: Designed for actual Blender integration, not just benchmarks.
2. **Multiple operation modes**: Denoiser, adaptive, multires, animation.
3. **Three model tiers**: Same architecture, different size/quality/speed tradeoffs.
4. **Energy conservation loss**: Physically-motivated loss function.
5. **AOV handling**: Render-time enabling of AOV passes for both Mitsuba and Blender.

### 3.3 Architectural Strengths

1. **Tile-based routing is the right call**: Per-pixel routing loses context (1 pixel has no meaning). 8x8 tiles with cryptomatte masks preserve spatial semantics.
2. **Swin at bottleneck, Mamba at full-res**: Correct allocation — O(n²) at 8.3M pixels is impossible, O(n²) at ~510 windows of 64 tokens is trivial.
3. **MLA for skip compression**: Directly addresses the 6GB skip connection problem at 4K.

---

## 4. Critical Weaknesses

### 4.1 For Top-Tier Venues

| # | Weakness | Severity | Impact |
|---|----------|----------|--------|
| 1 | **Zero experimental results** | **Critical** | No metrics, no comparisons, no ablation studies. Reviewers will reject without data. |
| 2 | **Unclear baseline comparisons** | **Critical** | Specs mention OIDN/OptiX but no planned experiments. |
| 3 | **Overly ambitious scope** | **High** | Three model tiers, 23 MoE experts, 4 modes, MLA, motion, FP8. Reviewers will ask: "Do we need ALL of this?" |
| 4 | **Missing motivation for JEPA** | **High** | Why is JEPA better than just conditioning a U-Net with scene embeddings? No clear justification for the predictive architecture. |
| 5 | **No theoretical analysis** | **Medium** | Why does tile-based routing work better than alternatives? No formal grounding. |
| 6 | **Complexity risk** | **Medium** | So many components that ablation studies become combinatorially explosive. |

### 4.2 Claims That Need Proof

| Claim | Bar | How to Prove |
|-------|-----|-------------|
| "Beats DLSS 4.0" | Extremely high | DLSS has NVIDIA tensor cores + years of optimization. Need side-by-side at matched spp. |
| "4K60 at 4.5GB VRAM" | High | MLA compression is theoretically sound but needs runtime validation. |
| "Universal scene awareness" | Medium | Does cryptomatte material ID actually capture enough semantics? |
| "Graceful degradation with missing AOVs" | Medium | Does shared expert alone actually work when all AOVs are missing? |

---

## 5. Baselines Required

### 5.1 Must Compare Against (Critical)

| Baseline | Version | Why |
|----------|---------|-----|
| **Intel OIDN 2.x** | Latest | State-of-the-art CPU denoiser, open source, widely used |
| **NVIDIA OptiX Denoiser** | Latest | State-of-the-art GPU denoiser, production standard |
| **KPCN** | Original + reimplementation | Kernel-predicting convolutional networks, foundational work |
| **Neural Bilateral Grid** | Latest | Real-time 1-spp denoising, efficiency baseline |

### 5.2 Should Compare Against (Important)

| Baseline | Why |
|----------|-----|
| **Disney Deep-Z Neural Denoiser** | Production kernel prediction (if code/data available) |
| **Restormer** | Transformer-based image restoration baseline |
| **Standard U-Net (no MoE, no scene conditioning)** | Ablation baseline |
| **Per-pixel MoE routing** | Directly validates tile-based vs pixel-based claim |
| **Temporal accumulation only** | Validates motion expert contribution |

### 5.3 Metrics Required

| Metric | Type | Notes |
|--------|------|-------|
| **PSNR** | Pixel-level | Standard, expected |
| **SSIM** | Structural | Standard, expected |
| **LPIPS** | Perceptual | Learned perceptual similarity |
| **FLIP** | Perceptual (rendering-specific) | NVIDIA's FLIP metric for rendering |
| **Temporal flicker** | Temporal | Frame-to-frame consistency (e.g., TEPE metric) |
| **Wall-clock time** | Performance | Inference time at various resolutions |
| **VRAM usage** | Performance | Peak memory at 1080p, 1440p, 4K |
| **SPP efficiency** | Practical | Quality vs spp curves |

---

## 6. Ablation Studies

### 6.1 Essential Ablations (Must Have)

| # | Experiment | What It Validates |
|---|-----------|-------------------|
| 1 | **Full model vs. no MoE (single FFN)** | Does MoE routing actually help over a bigger single expert? |
| 2 | **Tile-based routing vs. per-pixel routing** | Seam artifacts, context preservation, routing quality |
| 3 | **Scene graph conditioning vs. raw AOV only** | Does JEPA scene understanding add value over direct AOV input? |
| 4 | **With vs. without motion experts** | Temporal flicker reduction, motion blur handling |
| 5 | **With vs. without MLA skip compression** | Quality vs memory tradeoff at 4K |

### 6.2 Recommended Ablations (Should Have)

| # | Experiment | What It Validates |
|---|-----------|-------------------|
| 6 | **Expert count: 8 vs 16 vs 23 experts** | Optimal expert count for MC denoising |
| 7 | **Tile size: 4x4 vs 8x8 vs 16x16** | Optimal tile granularity |
| 8 | **Fingerprint dimensions: 8 vs 17 vs 23** | Motion features contribution |
| 9 | **Swin vs Restormer attention at bottleneck** | Architecture choice validation |
| 10 | **Mamba encoder vs pure Conv2d encoder** | Mamba contribution at full-res |
| 11 | **SIGReg weight λ: 0.0 vs 0.05 vs 0.09 vs 0.15** | Loss sensitivity |
| 12 | **Each expert category individually** | Material vs light vs geometry vs motion contribution |

### 6.3 Ablation Matrix

```
                    Full    No MoE   No Scene   No Motion   No MLA    No Mamba
                    ----    ------   --------   ---------   ------    --------
PSNR                xxx     xxx      xxx        xxx         xxx       xxx
SSIM                xxx     xxx      xxx        xxx         xxx       xxx
LPIPS               xxx     xxx      xxx        xxx         xxx       xxx
FLIP                xxx     xxx      xxx        xxx         xxx       xxx
Temporal Flicker    xxx     xxx      xxx        xxx         xxx       xxx
Inference Time      xxx     xxx      xxx        xxx         xxx       xxx
VRAM                xxx     xxx      xxx        xxx         xxx       xxx
```

---

## 7. Missing Components for Publication

### 7.1 Critical (Must Have Before Submission)

| Component | Effort | Notes |
|-----------|--------|-------|
| **Results section** | High | PSNR, SSIM, LPIPS, FLIP on diverse scenes |
| **Baseline comparisons** | High | Side-by-side with OIDN, OptiX, KPCN |
| **Ablation table** | Medium | At least 5 ablation experiments |
| **Training dataset description** | Medium | Scene count, diversity, generation method |
| **Qualitative comparisons** | Medium | Side-by-side renders with artifacts highlighted |
| **Failure case analysis** | Low | When does Omen break? (caustics, volumetrics, very low spp) |

### 7.2 Recommended (Strengthens Paper)

| Component | Effort | Notes |
|-----------|--------|-------|
| **Perceptual user study** | Medium | 10-20 people, A/B preference testing |
| **Convergence curves** | Low | Quality vs spp plots for each baseline |
| **Speed/quality Pareto** | Low | Frontier plot showing Omen vs baselines |
| **Training loss curves** | Low | Show SIGReg + prediction loss convergence |
| **Expert activation visualization** | Medium | Show which experts fire on which scene regions |
| **Routing entropy analysis** | Low | Show routing diversity across scene types |

---

## 8. Risk Assessment

### 8.1 High-Risk Claims

| Claim | Risk | Why | Mitigation |
|-------|------|-----|-----------|
| Tile MoE improves over single FFN | **Low** | MoCE-IR proved MoE for IR | Run the ablation |
| Cryptomatte masks help routing | **Medium** | Intuitively correct but simple depth+normal might be enough | Ablation: full fingerprint vs reduced fingerprint |
| Scene graph conditioning helps | **High** | Biggest unknown — raw AOV might be sufficient | Critical ablation: scene graph vs raw AOV |
| MLA skip compression works | **Medium** | Theory sound but 16x might lose quality at 4K | Quality vs compression ablation |
| Motion-aware experts help | **Medium** | Must show temporal flicker reduction, not just single-frame | Temporal metric comparison |
| Self-training on Cornell box works | **High** | Distribution gap between simple boxes and complex scenes | Plan Blender demo file training |
| Beats DLSS 4.0 | **Very High** | NVIDIA has years of optimization + tensor cores | Don't claim this until proven |
| 4K60 at 4.5GB VRAM | **High** | Unproven, MLA compression needs runtime validation | Profile after implementation |

### 8.2 Risk Mitigation Priority

1. **Implement tile-based MoE first** — validate core claim
2. **Run scene graph ablation early** — if raw AOV is sufficient, reframe contribution
3. **Profile MLA compression** — validate memory claims before publishing
4. **Don't oversell DLSS comparison** — position as complementary, not replacement

---

## 9. Target Venues

### 9.1 Primary Targets

| Venue | Deadline | Fit | Why | Required Work |
|-------|----------|-----|-----|---------------|
| **EGSR 2026** | ~Feb 2026 | **Best fit** | Rendering-focused, accepts systems papers, practical bias | Tile MoE + ablations + baselines |
| **HPG 2026** | ~Apr 2026 | **Strong fit** | Mojo/MoE/speed focus, performance-oriented | Performance benchmarks + quality metrics |
| **SIGGRAPH 2026 (Technical Paper)** | ~Jan 2026 | Ambitious | Rendering track, high visibility | Very strong results + perceptual study |
| **SIGGRAPH Asia 2026** | ~Jun 2026 | Good | Slightly lower bar than main SIGGRAPH | Full experimental validation |

### 9.2 Secondary Targets

| Venue | Fit | Why |
|-------|-----|-----|
| **EG (Eurographics) 2027** | Good | Broader graphics audience |
| **CVPR 2026 workshop** | Possible | If reframed as "MoE for image restoration" |
| **NeurIPS 2026 workshop** | Possible | If framed as "scene-aware neural rendering" |
| **arXiv tech report** | Always | Publish design while building experiments |

### 9.3 Venue Recommendation

**Start with EGSR or HPG** — these are the right audience for a rendering system paper. SIGGRAPH main track is achievable as a follow-up if results are strong.

---

## 10. Publication Strategy

### 10.1 Recommended: Split into Multiple Papers

**Don't try to publish everything at once.** The architecture is too complex for a single paper. Split into focused contributions:

#### Paper 1: Tile-Based MoE Routing for MC Denoising
- **Core contribution**: 8x8 tile-based expert routing with cryptomatte-style material/light/geometry masks
- **What to include**: U-Net backbone, MoE bottleneck, 23-dim fingerprint, auxiliary-loss-free balancing
- **What to leave out**: JEPA conditioning, MLA compression, motion experts, FP8, DualPipe
- **Target**: EGSR 2026 or HPG 2026
- **Baselines**: OIDN, OptiX, KPCN, single FFN, per-pixel MoE
- **Estimated pages**: 10-12

#### Paper 2: Scene-Graph Conditioned Denoising via JEPA World Model
- **Core contribution**: JEPA world model conditions denoiser predictions on structured scene graph embeddings
- **What to include**: Scene graph encoder, AdaLN-zero conditioning, SIGReg loss, scene delta encoding
- **What to leave out**: MoE details (cite Paper 1), MLA, motion, performance optimizations
- **Target**: SIGGRAPH 2027 main track
- **Baselines**: Paper 1 (without scene graph) + OIDN + OptiX
- **Estimated pages**: 12-14

#### Paper 3: Motion-Aware MoE Experts for Animated Sequence Denoising
- **Core contribution**: 4 motion experts (static/linear/fast/occlusion) + temporal reprojection
- **What to include**: Motion vector handling, tile fingerprint motion dimensions, expert specialization
- **What to leave out**: Full architecture details (cite Papers 1-2)
- **Target**: EG 2027 or SIGGRAPH Asia 2026
- **Estimated pages**: 8-10

### 10.2 Alternative: Single Systems Paper

If splitting feels wrong for the story:

- **Title**: "Omen: Scene-Aware Neural Monte Carlo Denoising with Tile-Based Mixture of Experts"
- **Target**: SIGGRAPH 2026 or EGSR 2026
- **Risk**: Reviewers may find it too complex, ask for simplification
- **Mitigation**: Focus on MoE routing as the headline, everything else as supporting

### 10.3 Publication Timeline

```
2026 Q1-Q2:  Implement core system (tile MoE + U-Net + basic training)
2026 Q2-Q3:  Generate training data from Blender demo files
2026 Q3:     Run ablation studies + baseline comparisons
2026 Q3:     Write Paper 1 → Submit to EGSR/HPG
2026 Q4:     Add JEPA conditioning → Run ablations
2027 Q1:     Write Paper 2 → Submit to SIGGRAPH
2027 Q2:     Add motion experts → Temporal experiments
2027 Q3:     Write Paper 3 → Submit to EG/SA
```

---

## 11. Training Data Plan

### 11.1 Selected Blender Demo Scenes (Top 15)

| # | Scene | Category | Size | Training Value | License |
|---|-------|----------|------|----------------|---------|
| 1 | Blender 5.1 — Singularity | Splash Art | 670MB | Complex materials, production quality, diverse geometry | CC-BY |
| 2 | Blender 4.5 — DOGWALK | Splash Art | 383MB | Character + fur/hair + environment, animation potential | CC-BY |
| 3 | Blender 4.2 — Gold | Splash Art | 300MB | Metallic materials, reflective surfaces | CC-BY |
| 4 | Agent 327 Barbershop | Cycles | 280MB | Interior scene, diverse materials (glass, metal, wood, skin), branched PT | CC-BY |
| 5 | Cosmos Laundromat | Cycles | 230MB | Outdoor scene, characters, animated | CC-BY |
| 6 | Classroom | Cycles | 72MB | Classic archviz, diverse materials, standard benchmark scene | **CC0** |
| 7 | Barcelona Pavilion | Cycles | 24MB | Architecture, clean materials, geometric complexity | CC-BY |
| 8 | Italian Flat | Cycles | 368MB | Interior, furniture, complex material combinations | CC-BY |
| 9 | Blender 3.4 — Charge (Open Movie) | Splash Art | 1.4GB | Full open movie scene, characters, animation | CC-BY |
| 10 | Monster Under The Bed | Cycles | — | SSS skin, complex shading, character rendering | CC-BY |
| 11 | Hair Styles | Hair/Geo Nodes | — | Fur/hair rendering (challenging for denoisers) | CC-BY-SA |
| 12 | Animal Fur Examples | Hair/Geo Nodes | — | More hair variety, different fur types | CC0 |
| 13 | Ember Forest | EEVEE | — | Volumetrics, fire, atmosphere | — |
| 14 | Wasp Bot | EEVEE | — | Metal + emission, robotic materials | — |
| 15 | Nishita Sky Demo | Cycles | — | Sky/atmosphere rendering, outdoor lighting | — |

### 11.2 Additional Scenes to Consider

| Scene | Why |
|-------|-----|
| Blender 4.1 — Lynxsdesign (276MB) | Complex product rendering |
| Blender 3.6 — Pet Projects (245MB) | Characters + environment |
| Lone Monk | Volumetrics, atmospheric effects |
| Architectural Visualization | Standard archviz benchmark |
| Chocolate Donut | SSS, food materials |

### 11.3 Training Dataset Specifications

**From 15 scenes with animated cameras (30-50 frames each):**

| Metric | Value |
|--------|-------|
| **Total training pairs** | 500-750 |
| **Noisy input** | 1-4 spp (randomly sampled per pair) |
| **Ground truth** | 256-4096 spp (converged reference) |
| **AOV buffers per pair** | albedo(3), normal(3), depth(1), motion_vectors(2), cryptomatte_material(4), cryptomatte_object(4) |
| **Resolution** | 1920x1080 (Full HD), subset at 3840x2160 (4K) |
| **License coverage** | All CC0 or CC-BY (fine for research use) |

### 11.4 Scene Coverage Matrix

| Scene Type | Count | Challenge Coverage |
|------------|-------|--------------------|
| Interior / Archviz | 4 | Indirect lighting, glossy surfaces, caustics |
| Exterior / Outdoor | 3 | Sky lighting, large open spaces, vegetation |
| Character / Organic | 3 | SSS, hair/fur, skin, eyes |
| Product / Hard Surface | 2 | Metals, glass, reflections |
| Volumetric / Atmosphere | 2 | Smoke, fire, fog, god rays |
| Hair / Fur | 2 | Strand rendering, anisotropic shading |

### 11.5 Animated Camera Patterns

For each scene, generate 30-50 frames with camera animations:

1. **Orbit**: 360-degree rotation around center of interest
2. **Dolly**: Forward/backward movement (depth change)
3. **Pan/Tilt**: Horizontal and vertical sweeps
4. **Dutch Angle**: Roll animation (tests rotation-invariance)
5. **Flythrough**: Moving through the scene (tests large motion vectors)

### 11.6 Training Pipeline

```
Blender Demo Files (.blend)
    ↓ (Blender Python API - bpy)
Scene Graph Extraction
    ├── Materials: mat.node_tree → bNodeTree → node types + values
    ├── Geometry: obj.to_mesh() → vertices, faces, UVs, normals
    ├── Lights: light.type, energy, color, transform
    └── Cameras: fov, clip, transform
    ↓ (Scene converter: Blender nodes → Mitsuba XML)
Mitsuba Scene Files (.xml)
    ↓ (Mitsuba 3 renderer)
Render Training Pairs
    ├── Noisy: mi.render(scene, spp=random(1,4))
    ├── Ground truth: mi.render(scene, spp=256-4096)
    └── AOV buffers: albedo, normal, depth, motion, cryptomatte
    ↓ (Mojo tile fingerprint computation)
Tile Fingerprints (23-dim per 8x8 tile)
    ↓ (Nabla training loop)
Omen Model Training
    ├── Loss: pred_loss + λ * sigreg_loss
    ├── Optimizer: AdamW (lr=1e-3)
    └── Schedule: cosine with warmup
```

---

## 12. Blender Shared Node System Architecture

### 12.1 Core Architecture

Based on exhaustive analysis of the Blender source code at `/home/riju279/Documents/Projects/MOJO/Cycles_mojo/blender/`.

```
.blend file
    ↓  DNA_material_types.h: Material { nodetree: *bNodeTree }
bNodeTree (NTREE_SHADER = 0)
    ↓  DNA_node_types.h: bNode, bNodeSocket, bNodeLink
    ↓  BKE_node.hh: bNodeType { gpu_fn, materialx_fn, exec_fn, ... }
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    INTERPRETATION LAYER                       │
├──────────────────┬──────────────────┬────────────────────────┤
│  CYCLES PATH     │  EEVEE PATH      │  EXPORT PATH           │
│                  │                  │                        │
│ shader.cpp:      │ gpu_material.cc: │ usd_writer_material.cc │
│ add_nodes()      │ GPU_material_    │ material.cc:           │
│ ↓                │ from_nodetree()  │ export_to_materialx()  │
│ b_node.is_type   │ ↓                │ ↓                      │
│ ("ShaderNode     │ ntreeGPU         │ materialx_fn           │
│  BsdfDiffuse")   │ MaterialNodes()  │ callback               │
│ ↓                │ ↓                │ ↓                      │
│ Cycles internal  │ gpu_fn callback  │ MaterialX document     │
│ ShaderGraph      │ per node type    │ ↓                      │
│ ↓                │ ↓                │ USD / glTF / Alembic   │
│ SVM/OSL compile  │ GLSL compile     │                        │
│ ↓                │ ↓                │                        │
│ CPU/GPU render   │ Real-time render │                        │
└──────────────────┴──────────────────┴────────────────────────┘
```

### 12.2 Key Source Files

#### Node System Core

| File | Purpose |
|------|---------|
| `source/blender/makesdna/DNA_node_types.h` | bNodeTree, bNode, bNodeSocket DNA structures |
| `source/blender/makesdna/DNA_material_types.h` | Material { nodetree: *bNodeTree } |
| `source/blender/blenkernel/BKE_node.hh` | bNodeType with execution callbacks |
| `source/blender/nodes/NOD_shader.h` | Shader node tree type, output node finder |
| `source/blender/nodes/shader/node_shader_tree.cc` | Shader tree registration, localization |

#### Cycles Conversion

| File | Purpose |
|------|---------|
| `intern/cycles/blender/sync.h` | BlenderSync class definition |
| `intern/cycles/blender/shader.cpp` | Node-by-node conversion (1000+ lines of if-else) |
| `intern/cycles/blender/session.h` | BlenderSession management |

#### EEVEE Conversion

| File | Purpose |
|------|---------|
| `source/blender/draw/engines/eevee/eevee_material.hh` | Closure system, material compilation |
| `source/blender/draw/engines/eevee/eevee_sync.hh` | Sync module (mesh, volume, curves) |
| `source/blender/gpu/intern/gpu_material.cc` | GPU_material_from_nodetree() |

#### Export

| File | Purpose |
|------|---------|
| `source/blender/io/usd/intern/usd_writer_material.cc` | USD material export |
| `source/blender/nodes/shader/materialx/material.cc` | MaterialX export |

#### Render Engine API

| File | Purpose |
|------|---------|
| `source/blender/render/RE_engine.h` | RenderEngineType, RenderEngine structs |
| `source/blender/render/RE_bake.h` | Bake API |
| `source/blender/depsgraph/DEG_depsgraph_query.hh` | Depsgraph query API |

### 12.3 Critical Architecture Details

#### Output Node Targeting

```c
// DNA_node_types.h
SHD_OUTPUT_ALL = 0,      // Use for all renderers
SHD_OUTPUT_EEVEE = 1,    // EEVEE-specific output
SHD_OUTPUT_CYCLES = 2,   // Cycles-specific output
```

Each renderer finds its output node via:
```cpp
// NOD_shader.h
bNode *ntreeShaderOutputNode(bNodeTree *ntree, int target);
```

#### Cycles Node Conversion Pattern

Cycles reads Blender nodes directly via RNA API, using a type-by-type if-else chain:
```cpp
// intern/cycles/blender/shader.cpp
if (b_node.is_type("ShaderNodeBsdfDiffuse"_ustr)) { ... }
else if (b_node.is_type("ShaderNodeBsdfMetallic"_ustr)) { ... }
else if (b_node.is_type("ShaderNodeBsdfAnisotropic"_ustr)) { ... }
// ~100+ node types mapped
```

#### EEVEE Node Conversion Pattern

EEVEE uses the GPU material compilation system:
```cpp
// gpu_material.cc
GPUMaterialFromNodeTreeResult GPU_material_from_nodetree(
    Scene *scene, bNodeTree *ntree, ...) {
  bNodeTree *localtree = bke::node_tree_add_tree(...);
  ntreeGPUMaterialNodes(localtree, mat);  // Calls gpu_fn per node
  // Compiles to GLSL shader
}
```

Each node type's `gpu_fn` callback generates GLSL code fragments.

#### MaterialX Export Pattern

```cpp
// materialx/material.cc
void export_to_materialx(bNodeTree *ntree, ...) {
  // Traverses node tree
  // Calls materialx_fn callback per node
  // Generates MaterialX document
}
```

### 12.4 For Omen: Third Interpretation Path

Omen needs to read the same `bNodeTree` and extract scene graph information. The recommended approach:

1. **Follow Cycles' pattern** (read via Python RNA API in Blender plugin)
2. **Don't need to convert to shaders** — we just need material type + parameters
3. **Use existing MaterialX export** as reference for scene graph extraction
4. **Extract**: material types (diffuse, metallic, glass, SSS, emission), roughness, metallic, albedo color, normal maps, geometry topology, light types/positions, camera parameters

---

## 13. Omen Blender Plugin Design

### 13.1 Render Engine Registration

```python
import bpy

class OmenEngine(bpy.types.RenderEngine):
    bl_idname = "OMEN_RENDER"
    bl_label = "Omen"
    bl_use_eevee_viewport = True       # EEVEE for interactive preview
    bl_use_postprocess = True           # Use Blender's compositor
    bl_use_shading_nodes_custom = True  # Custom shader node support

    def update_render_passes(self, scene, view_layer):
        """Declare all AOV passes Omen produces."""
        # Standard passes
        self.register_pass(scene, view_layer, "Combined", 4, "RGBA", 'COLOR')
        self.register_pass(scene, view_layer, "Depth", 1, "Z", 'VALUE')

        # AOV buffers for denoiser
        self.register_pass(scene, view_layer, "Diffuse Color", 3, "RGB", 'COLOR')
        self.register_pass(scene, view_layer, "Specular Color", 3, "RGB", 'COLOR')
        self.register_pass(scene, view_layer, "Normal", 3, "XYZ", 'VECTOR')
        self.register_pass(scene, view_layer, "Vector", 4, "XYZW", 'VECTOR')  # motion

        # Cryptomatte (material/object/asset)
        self.register_pass(scene, view_layer, "CryptoMaterial", 4, "RGBAAA", 'COLOR')
        self.register_pass(scene, view_layer, "CryptoObject", 4, "RGBAAA", 'COLOR')
        self.register_pass(scene, view_layer, "CryptoAsset", 4, "RGBAAA", 'COLOR')

    def render(self, depsgraph):
        """Final render (F12)."""
        scene = depsgraph.scene_eval

        # 1. Extract scene graph
        scene_graph = self._extract_scene_graph(depsgraph)

        # 2. Render via Mitsuba (training) or internal pipeline
        noisy_result = self._render_noisy(scene_graph, spp=scene.omen.spp)

        # 3. Extract AOV buffers
        aov_buffers = self._extract_aov(scene_graph)

        # 4. Denoise via JEPA model
        clean_result = self._denoise(noisy_result, aov_buffers, scene_graph)

        # 5. Return result to Blender
        result = self.begin_result(0, 0, scene.render.resolution_x,
                                    scene.render.resolution_y)
        layer = result.layers[0]
        layer.passes["Combined"].rect = clean_result
        self.end_result(result)

    def view_update(self, context, depsgraph):
        """Viewport update trigger."""
        pass

    def view_draw(self, context, depsgraph):
        """Viewport render."""
        pass

    def _extract_scene_graph(self, depsgraph):
        """Extract scene graph from Blender depsgraph."""
        scene_graph = {
            'cameras': [],
            'lights': [],
            'materials': [],
            'meshes': [],
            'objects': []
        }

        for obj in depsgraph.objects:
            if obj.type == 'MESH':
                mesh = obj.to_mesh()
                scene_graph['meshes'].append({
                    'name': obj.name,
                    'vertices': [v.co[:] for v in mesh.vertices],
                    'faces': [list(p.vertices) for p in mesh.polygons],
                    'normals': [n[:] for n in mesh.normals()],
                    'uvs': ([d.uv[:] for d in mesh.uv_layers.active.data]
                            if mesh.uv_layers else None),
                    'material_indices': [p.material_index for p in mesh.polygons],
                    'transform': [list(row) for row in obj.matrix_world],
                })
                obj.to_mesh_clear()

            elif obj.type == 'LIGHT':
                light = obj.data
                scene_graph['lights'].append({
                    'name': obj.name,
                    'type': light.type,
                    'energy': light.energy,
                    'color': list(light.color),
                    'transform': [list(row) for row in obj.matrix_world],
                })

            elif obj.type == 'CAMERA':
                cam = obj.data
                scene_graph['cameras'].append({
                    'name': obj.name,
                    'fov': cam.angle,
                    'clip_start': cam.clip_start,
                    'clip_end': cam.clip_end,
                    'transform': [list(row) for row in obj.matrix_world],
                })

        # Extract materials from shared node system
        for mat in bpy.data.materials:
            if mat.use_nodes and mat.node_tree:
                scene_graph['materials'].append(
                    self._extract_material_nodes(mat))

        return scene_graph

    def _extract_material_nodes(self, mat):
        """Read bNodeTree → Omen material representation."""
        ntree = mat.node_tree
        material = {
            'name': mat.name,
            'nodes': [],
            'links': [],
        }

        for node in ntree.nodes:
            node_data = {
                'type': node.bl_idname,
                'name': node.name,
                'location': node.location[:],
                'inputs': {},
            }

            # Read input socket values
            for socket in node.inputs:
                if socket.is_linked:
                    node_data['inputs'][socket.name] = {
                        'linked': True,
                        'from': socket.links[0].from_node.name
                    }
                else:
                    node_data['inputs'][socket.name] = {
                        'linked': False,
                        'value': self._get_socket_value(socket)
                    }

            material['nodes'].append(node_data)

        for link in ntree.links:
            material['links'].append({
                'from_node': link.from_node.name,
                'from_socket': link.from_socket.identifier,
                'to_node': link.to_node.name,
                'to_socket': link.to_socket.identifier,
            })

        return material

    def _get_socket_value(self, socket):
        """Extract default value from unlinked socket."""
        if socket.type == 'VALUE':
            return socket.default_value
        elif socket.type == 'RGBA':
            return list(socket.default_value)
        elif socket.type == 'VECTOR':
            return list(socket.default_value)
        elif socket.type == 'INT':
            return socket.default_value
        elif socket.type == 'BOOLEAN':
            return socket.default_value
        return None

def register():
    bpy.utils.register_class(OmenEngine)

def unregister():
    bpy.utils.unregister_class(OmenEngine)
```

### 13.2 Render Engine API Summary (from RE_engine.h)

| Callback | Purpose | When Called |
|----------|---------|------------|
| `update(engine, bmain, depsgraph)` | Initial scene sync | Scene changes |
| `render(engine, depsgraph)` | Final render | F12 / animation render |
| `render_frame_finish(engine)` | Post-render cleanup | After all view layers |
| `draw(engine, context, depsgraph)` | Display render result | During render() |
| `bake(engine, depsgraph, object, ...)` | Texture baking | Bake operation |
| `view_update(engine, context, depsgraph)` | Viewport sync | Scene changes in viewport |
| `view_draw(engine, context, depsgraph)` | Viewport render | Every viewport redraw |
| `update_render_passes(engine, scene, view_layer)` | Declare AOV passes | Before render |
| `update_script_node(engine, ntree, node)` | Script node update | OSL/script nodes |

### 13.3 Engine Type Flags

```c
RE_INTERNAL              // Built-in engine (not Python)
RE_USE_PREVIEW           // Supports material preview renders
RE_USE_POSTPROCESS       // Uses Blender's compositing
RE_USE_EEVEE_VIEWPORT    // Use EEVEE for viewport rendering
RE_USE_SHADING_NODES_CUSTOM  // Custom shading node support
RE_USE_GPU_CONTEXT       // Needs GPU context
RE_USE_SPHERICAL_STEREO  // Stereo rendering support
RE_USE_MATERIALX         // MaterialX support
```

### 13.4 Key Result API Functions

```c
// Create render result container
RenderResult *RE_engine_begin_result(engine, x, y, w, h, layername, viewname);

// Add custom pass to result
void RE_engine_add_pass(engine, name, channels, chan_id, layername);

// Submit completed result
void RE_engine_end_result(engine, result, cancel, highlight, merge_results);

// Register pass in update_render_passes callback
void RE_engine_register_pass(engine, scene, view_layer, name, channels, chanid, type);
```

---

## 14. Actionable Next Steps

### Phase 1: Core Implementation (Current)
- [ ] Implement tile-based MoE routing (strongest single contribution)
- [ ] Build U-Net backbone with AOV input channels
- [ ] Cornell box self-training pipeline
- [ ] Basic inference pipeline

### Phase 2: Training Data (After Core Works)
- [ ] Write Blender scene extraction script (Python API pattern above)
- [ ] Select 15 Blender demo scenes
- [ ] Generate animated camera sequences (30-50 frames per scene)
- [ ] Render training pairs via Mitsuba (1-4 spp noisy + 256-4096 spp GT)
- [ ] Extract all AOV buffers

### Phase 3: Experimental Validation
- [ ] Run ablation: MoE vs single FFN
- [ ] Run ablation: tile-based vs per-pixel routing
- [ ] Run ablation: scene graph vs raw AOV conditioning
- [ ] Compare against OIDN 2.x, OptiX denoiser
- [ ] Compute SSIM, PSNR, LPIPS, FLIP metrics
- [ ] Analyze failure cases (caustics, volumetrics, very low spp)

### Phase 4: Paper Writing
- [ ] Write Paper 1: Tile-based MoE Routing for MC Denoising
- [ ] Target EGSR 2026 or HPG 2026
- [ ] Include: architecture, ablations, baselines, qualitative comparisons
- [ ] Supplementary: expert activation visualizations, routing heatmaps

### Phase 5: Extensions (Post-Paper 1)
- [ ] Add JEPA scene conditioning → Paper 2
- [ ] Add motion experts → Paper 3
- [ ] Add MLA compression, FP8, DualPipe (performance section)
- [ ] Target SIGGRAPH 2027 main track

---

## Appendix A: Key Reference Papers

1. Bako et al., "Kernel-Predicting Convolutional Networks for Denoising Monte Carlo Renderings" (KPCN), SIGGRAPH 2017
2. Chaitanya et al., "Interactive Reconstruction of Monte Carlo Image Sequences", ACM TOG 2017
3. Disney Research, "Neural Denoising for Deep-Z Monte Carlo Renderings", 2024
4. "Neural Kernel Regression for Consistent Monte Carlo Denoising", SIGGRAPH 2024
5. "Real-time Monte Carlo Denoising with the Neural Bilateral Grid", 2024
6. AMD GPUOpen, "Neural Supersampling and Denoising for Real-time Path Tracing", 2024
7. Zamfir et al., "MoCE-IR: Mixture of Complexity Experts for Image Restoration", CVPR 2025
8. "Neural Denoising for Spectral Monte Carlo Rendering", Eurographics
9. "Adversarial Monte Carlo denoising with conditioned auxiliary feature modulation"
10. Maes et al., "LeWorldModel (LeWM)", 2026 (LeCun)
11. Assran et al., "I-JEPA", 2023
12. DeepSeek-AI, "DeepSeek-V3 Technical Report", 2024
13. Liu et al., "Swin Transformer", ICCV 2021
14. Gu et al., "Mamba: Linear-Time Sequence Modeling", 2023

## Appendix B: Blender Source Code Key Paths

```
blender/
├── source/blender/
│   ├── makesdna/
│   │   ├── DNA_node_types.h          # bNodeTree, bNode, bNodeSocket
│   │   ├── DNA_material_types.h      # Material { nodetree }
│   │   └── DNA_scene_types.h         # Scene structure
│   ├── blenkernel/
│   │   └── BKE_node.hh               # bNodeType with callbacks
│   ├── nodes/
│   │   ├── NOD_shader.h              # Shader tree API
│   │   ├── shader/
│   │   │   ├── node_shader_tree.cc   # Tree registration, localization
│   │   │   ├── nodes/                # 90+ individual shader node files
│   │   │   └── materialx/material.cc # MaterialX export
│   │   └── geometry/                 # Geometry nodes
│   ├── draw/engines/
│   │   ├── eevee/                    # EEVEE engine (current)
│   │   │   ├── eevee_material.hh     # Closure system
│   │   │   └── eevee_sync.hh         # Sync module
│   │   ├── workbench/                # Workbench engine
│   │   └── overlay/                  # Overlay engine
│   ├── gpu/intern/
│   │   └── gpu_material.cc           # GPU_material_from_nodetree()
│   ├── render/
│   │   ├── RE_engine.h               # RenderEngineType, RenderEngine
│   │   └── RE_bake.h                 # Bake API
│   ├── depsgraph/
│   │   └── DEG_depsgraph_query.hh    # Depsgraph query API
│   └── io/
│       └── usd/intern/
│           └── usd_writer_material.cc # USD material export
└── intern/cycles/
    └── blender/
        ├── sync.h                     # BlenderSync class
        ├── sync.cpp                   # Scene synchronization
        ├── shader.cpp                 # Node-by-node conversion
        ├── session.h                  # BlenderSession
        └── addon/
            ├── __init__.py            # Cycles Python registration
            ├── properties.py          # Cycles properties
            └── engine.py              # Cycles engine class
```
