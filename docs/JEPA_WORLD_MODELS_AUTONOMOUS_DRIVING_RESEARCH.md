# JEPA World Models & Autonomous Driving: Research Findings for Omen

**Date:** 2026-05-12
**Purpose:** Map the landscape of JEPA world models in autonomous driving/simulation to assess Omen's novelty and extract actionable insights.

---

## Table of Contents

1. [LeWorldModel (LeWM)](#1-leworldmodel-lewm)
2. [V-JEPA 2](#2-v-jepa-2)
3. [NVIDIA Cosmos](#3-nvidia-cosmos)
4. [Tesla FSD](#4-tesla-fsd)
5. [DriveMamba / TrajectoryMamba](#5-drivemamba--trajectorymamba)
6. [NVIDIA Isaac Sim / Omniverse](#6-nvidia-isaac-sim--omniverse)
7. [Generative World Renderer](#7-generative-world-renderer)
8. [Novelty Assessment for Omen](#8-novelty-assessment-for-omen)
9. [What Omen Can Learn](#9-what-omen-can-learn)
10. [References](#10-references)

---

## 1. LeWorldModel (LeWM)

**Paper:** Maes, Le Lidec, Scieur, LeCun, Balestriero. "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels." arXiv:2603.19312, March 2026.

### Architecture

| Component | Parameters | Details |
|-----------|-----------|---------|
| ViT-Tiny Encoder | ~5.5M | 12 layers, hidden=192, heads=3, mlp=768, patch=14 |
| Projector MLP | ~0.8M | 192 -> 2048 -> 192, BatchNorm1d |
| Action Encoder | ~0.16M | Conv1d(10,10,k=1) + MLP(10->768->192) |
| ARPredictor (6 ConditionalBlocks) | ~10.8M | depth=6, hidden=192, heads=16, dim_head=64, mlp=2048 |
| pred_proj MLP | ~0.8M | 192->2048->192, BatchNorm1d |
| SIGReg | 0 | All buffers, no learnable params |
| **Total** | **~15-18M** | Trains on single GPU in hours |

### Two Losses

1. **Next-embedding prediction loss:** MSE between predicted and actual future embeddings
2. **SIGReg (Sketched-Isotropic-Gaussian Regularizer):** Enforces Gaussian-distributed latent embeddings via Cramer-Wold theorem with projection-based dimensionality reduction. Prevents representation collapse without EMA, stop-gradients, or pretrained encoders.

Key insight: Reduces tunable loss hyperparameters from 6 to 1 compared to prior end-to-end JEPAs.

### Performance

- **48x faster planning** than foundation-model-based world models (full planning completes in <1 second)
- 18% higher success rate on PushT task vs PLDM
- Competitive with DINO-WM despite DINO-WM having access to additional proprioceptive information

### Surprise Detection (Critical for Omen)

LeWM implements **violation-of-expectation (VoE)** evaluation:

- **Surprise = prediction error** (MSE between predicted and actual embeddings)
- Tested across three perturbation types:
  - Unperturbed reference (low baseline surprise)
  - Visual perturbation (object color change) -- weak, not statistically significant
  - Physical perturbation (object teleportation) -- **pronounced spike, p<0.01** paired t-test
- The model is **more sensitive to physical perturbations than visual ones**
- No explicit threshold in codebase; surprise is a continuous signal used as planning cost

### Mapping to Render Frame Prediction

Omen's adaptation replaces:
- **Robot actions** -> **Scene deltas** (camera moves, object transforms, new emitters, light changes)
- **Physical perturbation detection** -> **Unexpected scene change requiring re-render**
- **Planning in latent space** -> **Frame prediction in latent space (skip path tracing)**

---

## 2. V-JEPA 2

**Paper:** Assran et al. (Meta). "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning." arXiv:2506.09985, June 2025.

### Architecture & Scale

| Aspect | Details |
|--------|---------|
| Parameters | 1.2 billion |
| Training data | 1M+ hours of internet video |
| Architecture | JEPA (latent prediction, no pixel reconstruction) |
| Training | Self-supervised, action-free pre-training |
| Fine-tuning | Post-training with V-JEPA 2-AC using 62 hours of robot data |

### Two-Stage Training

1. **Stage 1:** Action-free JEPA pre-training on 1M+ hours internet video -- learns physical world dynamics without action labels
2. **Stage 2:** Post-train V-JEPA 2-AC (action-conditioned) on 62 hours of unlabeled robot videos from Droid dataset

### Performance

- 77.3 top-1 accuracy on Something-Something v2 (motion understanding)
- 39.7 recall-at-5 on Epic-Kitchens-100 (human action anticipation, SOTA)
- 84.0 on PerceptionTest, 76.9 on TempCompass (video QA at 8B scale with LLM alignment)
- **Zero-shot robot planning:** 65-80% success rate on pick-and-place in unseen environments

### Physical Reasoning & Surprise

V-JEPA 2 introduced **IntPhys 2** benchmark:

- Evaluates ability to distinguish physically plausible vs implausible scenarios
- Uses violation-of-expectation paradigm from developmental psychology
- Generates paired videos: identical up to a point, then a physics-breaking event in one
- **Key finding:** Even V-JEPA 2 is near chance on many IntPhys 2 scenarios, showing the gap between current models and human intuitive physics
- V-JEPA reliably shows higher "surprise" (prediction error) for physically impossible events than pixel-generative or text-based models

### Relevance to Omen

V-JEPA 2 proves:
1. JEPA can scale to 1.2B parameters for video world modeling
2. Latent prediction transfers to zero-shot physical interaction
3. The violation-of-expectation paradigm (which Omen uses for surprise detection) is a validated methodology with dedicated benchmarks
4. However, V-JEPA 2 operates on natural video, not render engine outputs -- Omen's controlled rendering domain is a different (potentially easier) problem

---

## 3. NVIDIA Cosmos

**Paper:** NVIDIA (Ali et al., 87 authors). "World Simulation with Video Foundation Models for Physical AI." arXiv:2511.00062, October 2025.

### Architecture Evolution

| Version | Architecture | Scale | Key Feature |
|---------|-------------|-------|-------------|
| Cosmos-Predict1 | Diffusion Transformer | Multiple sizes | Initial video generation for Physical AI |
| Cosmos-Predict2 | Improved DiT | - | Better video quality |
| **Cosmos-Predict2.5** | **Flow-based (Rectified Flow) + DiT** | **2B and 14B** | Unifies Text2World, Image2World, Video2World |

### Technical Details

- **Flow-based architecture** using Rectified Flow (not standard diffusion)
- Solves RF dynamics using Unified Predictor-Corrector (UniPC) framework
- Trained on **200M curated video clips**
- **Reinforcement learning-based post-training** for alignment
- Leverages **Cosmos-Reason1** (Physical AI VLM) for richer text grounding
- **Cosmos-Transfer2.5:** ControlNet-style framework for Sim2Real and Real2Real world translation, 3.5x smaller than v1 but higher fidelity

### How It Relates to Omen

| Dimension | Cosmos | Omen |
|-----------|--------|------|
| Goal | Generate synthetic training data for Physical AI | Accelerate interactive path tracing |
| Approach | Flow-matching video generation in pixel/latent space | JEPA latent prediction (no pixel generation) |
| Scale | 2B-14B params | ~18M params (LeWM scale) |
| Training | 200M curated video clips | Rendered frames with scene deltas |
| Physics | Learned from video | Governed by render engine |
| Output | Full video frames | Latent embeddings (decode only when needed) |

**Key insight:** Cosmos generates pixels; Omen predicts latents. Omen's approach is orders of magnitude cheaper because it doesn't need to generate full frames -- it only needs to detect when prediction is insufficient (surprise).

---

## 4. Tesla FSD

### Architecture (Public Knowledge)

Tesla's FSD pipeline is not fully published, but public disclosures and patents reveal:

1. **BEV (Bird's-Eye View) Representation:** Multi-camera images transformed into unified top-down feature space via transformer-based attention
2. **Occupancy Network:** Predicts 3D volumetric voxel grid -- "is something there?" for every voxel around the car. Replaces explicit object detection, handles novel objects (debris, etc.)
3. **End-to-End Pipeline:** FSD v12+ moved from modular (detection -> planning -> control) to fully neural end-to-end
4. **Temporal Fusion:** Incorporates past frame information for stability and prediction
5. **Fleet-scale Training:** Millions of miles of driving data

### JEPA Usage: NOT Confirmed

- **No public evidence** Tesla uses JEPA specifically
- Tesla's world model appears closer to a **latent dynamics model** with BEV + occupancy as the latent state
- The occupancy network acts as a learned 3D scene representation (analogous to JEPA's encoder)
- Prediction happens in BEV/occupancy space rather than a JEPA latent space
- Tesla's patent (September 2025) reveals advances in FSD visualization and world model representation

### Relevance to Omen

| Dimension | Tesla FSD | Omen |
|-----------|----------|------|
| Input | Multi-camera video | Rendered frames |
| Latent space | BEV + Occupancy voxels | JEPA embeddings |
| Prediction | Future BEV state | Future frame embeddings |
| Surprise | Anomaly in occupancy predictions | JEPA prediction error |
| Scale | Billions of miles of data | Artist scenes (much smaller domain) |

---

## 5. DriveMamba / TrajectoryMamba

### DriveMamba

**Paper:** Su, Wu, Song, Zhang, Yang, Yan. "DriveMamba: Task-Centric Scalable State Space Model for Efficient End-to-End Autonomous Driving." arXiv:2602.13301, February 2026. **Accepted to ICLR 2026.**

#### Architecture

- **Replaces Transformer decoders** with **Unified Mamba Decoders** (Selective SSM)
- Single-stage architecture integrating:
  - Dynamic task relation modeling (perception, planning jointly)
  - Implicit view correspondence (no explicit BEV projection needed)
  - Long-term temporal fusion
- Both image features and task outputs converted to token-level sparse representations
- Sorted by instantiated 3D positions
- **Bidirectional trajectory-guided "local-to-global" scan** preserves spatial locality

#### Key Properties

| Property | Transformer Baseline (UniAD) | DriveMamba |
|----------|------------------------------|------------|
| Complexity | O(n^2) attention | O(n) linear |
| Decoder | Separate per task | Unified Mamba |
| View fusion | Explicit BEV projection | Implicit correspondence |
| Scalability | Limited by attention cost | Monotonically improves with layers |

#### Benchmarks

- Evaluated on **nuScenes** and **Bench2Drive** datasets
- Demonstrates superiority, generalizability, and efficiency over Transformer-based E2E-AD baselines

### TrajectoryMamba (Tamba)

**Paper:** Huang et al. "Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM." **CVPR 2025.**

#### Architecture

- **Hybrid attention-Mamba** encoder-decoder for motion forecasting
- Uses **Selective SSM** to replace self-attention in key parts of the architecture
- Reduces computational complexity from quadratic to linear
- Introduces **joint polyline encoding** strategy
- Evaluated on **Argoverse 1 and Argoverse 2** datasets
- SOTA performance with **significantly reduced FLOPs and model parameters**

### Relevance to Omen

DriveMamba's architecture choices are directly relevant:
1. **Mamba's O(n) complexity** vs Transformer's O(n^2) -- relevant if Omen scales to high-resolution latents
2. **Task-centric unified decoder** -- Omen could use a single Mamba-based predictor for both spatial and temporal prediction
3. **Implicit view correspondence** -- if Omen needs multi-view rendering prediction
4. However, DriveMamba operates on BEV features for driving; Omen operates on rendered frame embeddings for a different purpose

---

## 6. NVIDIA Isaac Sim / Omniverse

### Rendering Architecture

Isaac Sim is built on **NVIDIA Omniverse**, which uses:

1. **RTX Renderer** with two modes:
   - **RTX Realtime:** Rasterization + ray tracing hybrid
   - **RTX Path Tracing:** Full path tracing (used for high-fidelity sensor simulation)

2. **Neural Rendering (New):**
   - **NVIDIA Omniverse NuRec** + **3DGUT (3D Gaussian with Unscented Transforms)**
   - Reconstructs photorealistic 3D scenes from ~100 photos using COLMAP -> 3DGUT pipeline
   - Export as USD -> deploy in Isaac Sim or CARLA
   - **Interactive frame rates** for reconstructed real-world scenes

3. **RTX Neural Materials:** Neural representations for materials in path-traced scenes, accelerating material evaluation

### Do They Use World Models or Neural Rendering for Acceleration?

**No JEPA-style world models.** NVIDIA's acceleration approach is:
- **Hardware acceleration:** RTX cores for ray tracing
- **Neural supersampling/denoising:** DLSS-style upscaling from low-SPP renders
- **Neural reconstruction:** 3DGUT for real-world scene capture
- **Traditional path guiding:** Online-trained neural networks for light transport (not JEPA)

**Key gap:** NVIDIA does NOT use a world model to predict future frames and skip rendering. They rely on:
- Path tracing every frame (accelerated by RT cores)
- Denoising low-SPP outputs
- DLSS temporal accumulation for upscaling

This is exactly the gap Omen fills: **predict frames instead of rendering them, only render when surprised.**

---

## 7. Generative World Renderer

**Paper:** Huang et al. "Generative World Renderer." arXiv:2604.02329, April 2026.

### Overview

- Large-scale dynamic dataset from AAA games: **4M continuous frames (720p/30 FPS)**
- Synchronized RGB + 5 G-buffer channels
- **Dual-screen stitched capture method** for synchronized data
- Enables both **inverse rendering** (geometry/material decomposition) and **forward rendering** (G-buffer-guided video generation)
- VLM-based evaluation protocol for inverse rendering quality

### Relevance to Omen

This is the closest published work to Omen's domain:
- Uses **G-buffers as conditioning** for forward rendering (analogous to Omen's scene delta encoding)
- Operates on **rendered/game frames** rather than natural video
- Focuses on generation quality rather than prediction-for-acceleration
- Does NOT use JEPA; uses standard generative models (diffusion/GAN)

---

## 8. Novelty Assessment for Omen

### What Omen Does That Others Don't

| Aspect | Omen | Existing Work |
|--------|------|---------------|
| JEPA for render frame prediction | YES | NO (LeWM uses JEPA for control, not rendering) |
| Surprise-based adaptive rendering | YES | NO (NVIDIA renders every frame) |
| Scene delta conditioning (not actions) | YES | NO (LeWM uses robot actions) |
| Latent prediction to skip path tracing | YES | NO (Cosmos generates full pixels) |
| Mojo/Nabla implementation | YES | NO (all existing work is PyTorch) |

### Partial Overlaps

1. **LeWM surprise detection -> Omen surprise-based re-rendering:** Same mechanism, different application. LeWM detects physical anomalies for planning; Omen detects render surprises for adaptive rendering. **Novel application of an existing mechanism.**

2. **V-JEPA 2 video prediction -> Omen frame prediction:** Same JEPA framework, but V-JEPA 2 predicts natural video frames, Omen predicts rendered frames in a controlled domain (much more constrained/predictable). **Different domain, same family.**

3. **Generative World Renderer -> Omen:** Both work on rendered/game frames. GWR generates from G-buffers; Omen predicts from scene deltas in latent space. **Different approach to same domain.**

4. **NVIDIA DLSS/denoising -> Omen:** Both accelerate rendering. DLSS upscales low-SPP frames; Omen potentially skips frames entirely. **Complementary approaches, not competing.**

### Novelty Verdict

**Omen's core idea -- JEPA world model predicting render frames in latent space, with surprise-triggered adaptive path tracing -- is novel.** No published work combines:
1. JEPA architecture for render prediction
2. Surprise-based rendering budget allocation
3. Scene delta conditioning (replacing robot actions)
4. Latent-only prediction (no pixel generation needed)

The closest analogues are:
- LeWM (same architecture, different application)
- NVIDIA Cosmos (same goal of Physical AI simulation, different approach at 100x the scale)
- Neural supersampling (same rendering acceleration goal, different technique)

---

## 9. What Omen Can Learn

### From LeWM

1. **SIGReg is crucial** -- without it, representations collapse. Omen must implement SIGReg exactly.
2. **AdaLN-zero conditioning** is the right way to inject scene deltas into the predictor.
3. **48x planning speed** demonstrates that small JEPAs can be extremely efficient at inference -- encouraging for real-time render prediction.
4. **Surprise detection works reliably** for physical perturbations -- Omen should expect clean signals for unexpected scene changes.

### From V-JEPA 2

1. **Scale matters for natural video** but Omen's domain (rendered frames) is more constrained, so ~18M params may be sufficient.
2. **Two-stage training** (action-free pre-train, then action-conditioned fine-tune) could work for Omen: pre-train on rendered frames without scene changes, then fine-tune with scene deltas.
3. **IntPhys 2 benchmark methodology** provides a template for evaluating Omen's surprise detection: generate pairs of rendered sequences where one has an unexpected change.
4. **V-JEPA 2-AC** shows that adding action conditioning as a post-training step works -- Omen's scene delta conditioning follows the same pattern.

### From NVIDIA Cosmos

1. **Flow-based models** (rectified flow) may be worth exploring if Omen eventually needs to generate pixels, not just predict latents.
2. **RL-based post-training** for alignment is an advanced technique Omen could adopt to fine-tune surprise thresholds.
3. **200M video clips** training scale is not needed for Omen -- render engine output is deterministic and much simpler than natural video.
4. **Cosmos-Transfer2.5** (Sim2Real/Real2Real) shows the value of domain transfer -- if Omen ever needs to bridge different render engines.

### From DriveMamba

1. **Mamba's O(n) complexity** is attractive if Omen scales to higher resolutions or longer temporal windows.
2. **Unified decoder** for multiple tasks is a clean architecture -- Omen could have a single predictor handle both spatial and temporal prediction.
3. **Bidirectional scan strategies** could improve Omen's spatial prediction quality.
4. However, **Mamba is newer and less proven** than transformers for this use case. LeWM's transformer predictor works well at 10M params.

### From Tesla FSD

1. **Occupancy-based representation** is a form of 3D latent space -- analogous to Omen's latent embeddings but structured as voxels.
2. **Temporal fusion across frames** is essential for stability -- Omen already plans this via autoregressive rollout.
3. **End-to-end differentiability** from perception to control maps to Omen's goal of end-to-end from scene encoding to frame prediction.
4. Tesla's approach confirms that **learning-based world modeling** is the industry direction, not hand-crafted heuristics.

### From NVIDIA Isaac Sim

1. **NVIDIA does NOT have a world model for frame prediction** -- this is Omen's competitive niche.
2. **Neural denoising + DLSS** are complementary to Omen: even when Omen triggers a re-render, the result can be low-SPP + neural denoising.
3. **3DGUT neural reconstruction** shows that Gaussian-based scene representations work for interactive rates -- relevant if Omen explores hybrid rendering.
4. **USD scene format** is the industry standard -- Omen should ensure compatibility.

---

## 10. References

### Papers

1. **LeWorldModel (LeWM):** Maes et al. "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels." arXiv:2603.19312, March 2026. [https://arxiv.org/abs/2603.19312](https://arxiv.org/abs/2603.19312)

2. **V-JEPA 2:** Assran et al. "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning." arXiv:2506.09985, June 2025. [https://arxiv.org/abs/2506.09985](https://arxiv.org/abs/2506.09985)

3. **NVIDIA Cosmos-Predict2.5:** NVIDIA (Ali et al.). "World Simulation with Video Foundation Models for Physical AI." arXiv:2511.00062, October 2025. [https://arxiv.org/abs/2511.00062](https://arxiv.org/abs/2511.00062)

4. **DriveMamba:** Su et al. "DriveMamba: Task-Centric Scalable State Space Model for Efficient End-to-End Autonomous Driving." arXiv:2602.13301, February 2026. Accepted to ICLR 2026. [https://arxiv.org/abs/2602.13301](https://arxiv.org/abs/2602.13301)

5. **TrajectoryMamba:** Huang et al. "Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM." CVPR 2025. [https://github.com/YiZhou-H/Trajectory-Mamba-CVPR](https://github.com/YiZhou-H/Trajectory-Mamba-CVPR)

6. **Generative World Renderer:** Huang et al. "Generative World Renderer." arXiv:2604.02329, April 2026. [https://arxiv.org/abs/2604.02329](https://arxiv.org/abs/2604.02329)

7. **High-Fidelity 4x Neural Reconstruction of Real-time Path Traced Images:** Lao et al. WACV 2025. [https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/papers/Lao_High-Fidelity_4x_Neural_Reconstruction_of_Real-time_Path_Traced_Images_WACVW_2025_paper.pdf](https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/papers/Lao_High-Fidelity_4x_Neural_Reconstruction_of_Real-time_Path_Traced_Images_WACVW_2025_paper.pdf)

8. **Value-guided action planning with JEPA world models:** arXiv:2601.00844. [https://arxiv.org/abs/2601.00844](https://arxiv.org/abs/2601.00844)

### Code Repositories

- LeWM: [https://github.com/lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)
- V-JEPA 2: [https://github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- Cosmos-Predict2.5: [https://github.com/nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- TrajectoryMamba: [https://github.com/YiZhou-H/Trajectory-Mamba-CVPR](https://github.com/YiZhou-H/Trajectory-Mamba-CVPR)
- 3DGUT: [https://github.com/nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut)
- IntPhys 2: [https://github.com/facebookresearch/IntPhys](https://github.com/facebookresearch/IntPhys)

### Industry Sources

- NVIDIA Cosmos: [https://www.nvidia.com/en-us/ai/cosmos/](https://www.nvidia.com/en-us/ai/cosmos/)
- NVIDIA Isaac Sim: [https://developer.nvidia.com/isaac/sim](https://developer.nvidia.com/isaac/sim)
- V-JEPA 2 Blog: [https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- LeWM Project: [https://le-wm.github.io/](https://le-wm.github.io/)
- NVIDIA NuRec: [https://developer.nvidia.com/blog/how-to-instantly-render-real-world-scenes-in-interactive-simulation/](https://developer.nvidia.com/blog/how-to-instantly-render-real-world-scenes-in-interactive-simulation/)

---

## Quick Reference: Architecture Comparison Table

| System | Params | Architecture | Domain | Prediction Space | Surprise? | Year |
|--------|--------|-------------|--------|-----------------|-----------|------|
| **Omen (proposed)** | **~18M** | **JEPA (LeWM port)** | **Rendered frames** | **Latent** | **Yes (adaptive re-render)** | **2026** |
| LeWM | ~15M | JEPA + SIGReg | Control tasks | Latent | Yes (VoE) | 2026 |
| V-JEPA 2 | 1.2B | JEPA | Natural video | Latent | Yes (IntPhys 2) | 2025 |
| Cosmos-Predict2.5 | 2B/14B | Flow + DiT | Physical AI video | Pixel | No | 2025 |
| Tesla FSD | Unknown | BEV + Occupancy | Driving | BEV voxels | Implicit | 2024+ |
| DriveMamba | Unknown | Mamba (SSM) | E2E driving | BEV tokens | No | 2026 |
| TrajectoryMamba | Small | Attention-Mamba | Trajectory forecast | Trajectories | No | 2025 |
| GWR | Unknown | Diffusion/GAN | Game frames | Pixel | No | 2026 |
| NVIDIA Isaac Sim | N/A | RTX Path Tracing | Simulation | N/A (renders all) | No | Ongoing |
