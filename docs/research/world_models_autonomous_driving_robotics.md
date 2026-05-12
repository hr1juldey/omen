# World Models in Autonomous Driving & Robotics: Architecture Survey
## Relevance to Omen's JEPA Rendering Engine

Date: 2026-05-12

---

## 1. Tesla's World Model Architecture

### Architecture: Transformer + BEV + Occupancy Networks (NOT JEPA)

Tesla's FSD stack (v12-v14, 2024-2025) uses a **transformer-based end-to-end** architecture, not JEPA:

- **Backbone**: BEV (Bird's Eye View) + Transformer with ~1 billion parameters
- **3D Understanding**: Occupancy Networks — voxel-based 3D volumetric grid replacing traditional object detection. Predicts occupancy of every 3D point around the vehicle
- **World Model Component**: FSD v14 incorporates an implicit world model that predicts future states for planning through imagined scenarios
- **Multimodal (v14)**: Inputs expanded to video + audio + navigation + vehicle state; outputs include 3D Gaussian Reconstruction and language-based scene understanding
- **3D Gaussian Splatting**: New in v14 for high-fidelity 3D scene reconstruction
- **No explicit JEPA**: Tesla has NOT publicly confirmed using JEPA. Their approach shares philosophical similarities (predicting high-level representations) but uses a different architectural lineage

**Key Takeaway for Omen**: Tesla proves that voxel/occupancy representations + transformers can scale to real-time 3D understanding. Their move toward 3D Gaussian Splatting in v14 validates the importance of neural scene representations for rendering-quality output.

---

## 2. NVIDIA's World Model Stack (Cosmos + Isaac Sim + Omniverse)

### 2a. NVIDIA Cosmos — World Foundation Models (CES 2025, Updated GTC 2025)

The most directly relevant system to Omen. Cosmos provides **two parallel architectures**:

1. **Diffusion-based WFM**: Transformer-based denoiser for iterative noise removal in video/state generation. Generates 30-second physics-aware predictive video
2. **Autoregressive-based WFM**: Sequential token prediction for world state evolution

Both use **transformer backbone** for scalability. Key technical details:
- Post-trained on **20,000 hours of driving video** for AV-specific workflows
- Cosmos Predict WFM models future world states as video from multimodal inputs (text, video, start-end frame)
- Open models released March 2025 (GTC), updated December 2025 with Alpamayo-R1
- Paper: arXiv:2501.03575

**Key Takeaway for Omen**: NVIDIA's dual-architecture approach (diffusion + autoregressive) is exactly the space Omen's JEPA + AR predictor occupies. The transformer-based denoiser pattern in Cosmos could inform Omen's denoiser mode.

### 2b. NVIDIA Isaac Sim / Omniverse

Built on **Universal Scene Description (USD)**, not neural rendering:
- **Isaac Sim**: Robotics simulation on top of Omniverse — physically accurate, NOT learned
- **DRIVE Sim**: Autonomous vehicle simulation variant
- **Project GR00T**: Foundation model for humanoid robots trained in Isaac Sim
- Neural rendering integration comes through **Cosmos** (see above), not Isaac Sim directly

**Key Takeaway for Omen**: Omniverse/Isaac Sim represents the "ground truth simulation" path (physics-based), while Cosmos represents the "neural prediction" path. Omen sits at the intersection — using neural prediction (JEPA) to approximate physics-based rendering.

### 2c. NVIDIA Neural Rendering Research (NeRF, Instant-NGP, 3DGS)

Relevant techniques from NVIDIA's rendering research:
- **Instant-NGP**: Hash-grid encoding for real-time neural radiance fields — Omen could use similar spatial encoding for scene representation
- **3D Gaussian Splatting**: Real-time differentiable rendering using 3D Gaussians — Tesla adopted this in FSD v14
- **Neural denoising**: OIDN (OptiX Denoiser) — already used in Cycles, Omen's denoiser mode could learn from this pipeline
- **Differentiable rendering** (via Mitsuba 3): Omen already integrates with this

---

## 3. JEPA-Based World Models Beyond LeWorldModel

### V-JEPA 2 (Meta AI, 2025)
- First world model trained on video achieving SOTA visual understanding
- Predicts in **latent/embedding space** rather than pixel space
- Directly relevant to Omen's JEPA approach — validates predicting in compressed representation

### LeWorldModel (LeWM)
- First JEPA that trains stably end-to-end from raw pixels using only two loss terms
- Addresses key instability challenge in JEPA training
- Paper: arXiv:2603.19312
- Omen already has this referenced in `lewm_technical_reference.md`

### DriveWorld (CVPR 2024)
- **4D pre-trained scene understanding via world models** for autonomous driving
- Learns spatiotemporal representations for perception + prediction
- Paper: arXiv:2405.04390

### JEPA World Model for Agent Dynamics
- Practical implementation predicting future latent states in simulated environments
- GitHub: Lake-Wang/Deep_Learning_JEPA_World_Model

---

## 4. Mamba SSM in Autonomous Driving

Mamba (Selective State Space Model) is being actively adopted in autonomous driving, with several 2024-2025 papers:

| Model | Focus | Key Innovation |
|-------|-------|---------------|
| **DriveMamba** | End-to-end driving | Unified Mamba decoders replacing Transformers |
| **Trajectory Mamba** | Trajectory prediction | Selective SSM + Attention hybrid |
| **AutoMamba** | Semantic segmentation | Real-time perception with Mamba |
| **MS3M** | Motion forecasting | Multi-stage Mamba for vehicle trajectories |

**Key Takeaway for Omen**: Mamba's linear-time sequence modeling makes it attractive for real-time temporal prediction. If Omen's AR predictor (`arpredictor.py`) faces quadratic complexity issues with transformers, Mamba is a proven alternative in the driving domain. The hybrid Attention+Mamba pattern (Trajectory Mamba) is especially relevant.

---

## 5. Scene Understanding + Pixel Prediction

This is exactly the problem Omen faces. Key approaches:

### DriveWorld (CVPR 2024)
- World model-based 4D representation learning
- Spatiotemporal pre-training for scene understanding

### GAIA-1 / GAIA-2 (Wayve)
- **Generative transformer world model** (9B parameters)
- Inputs: video + text + action (multimodal)
- Generates realistic driving videos with fine-grained control
- Architecture: image tokenizer → world model transformer → video decoder
- Paper: arXiv:2309.17080
- **Directly relevant**: GAIA-1's pipeline (encode → predict in latent space → decode to pixels) mirrors Omen's JEPA architecture

### Berkeley DeepDrive
- Pixel-level semantic scene understanding and forecasting
- Addresses recognizing and foreseeing situational changes

### NeurIPS 2025 — Long-Horizon Prediction
- Addresses temporal stability for extended predictions
- Critical for Omen's animation mode

### Accident Anticipation via World Models (Nature Comms Eng, 2025)
- End-to-end scene generation for safety-critical prediction

---

## 6. Architectural Patterns Summary for Omen

### Pattern 1: Dual-Path Prediction (NVIDIA Cosmos)
- **Diffusion path**: For high-quality spatial prediction (denoising, single-frame refinement)
- **Autoregressive path**: For temporal prediction (animation, multi-frame coherence)
- **Omen already has this**: JEPA latent prediction + AR predictor + denoiser mode

### Pattern 2: Latent-Space Prediction (JEPA / V-JEPA 2)
- Predict future states in compressed representation, NOT pixel space
- Decode to pixels only when needed for final output
- **Omen already does this**: scene_encoder → latent prediction → decoder

### Pattern 3: 3D Scene Representation (Tesla / 3DGS)
- Occupancy networks or 3D Gaussian Splatting for spatial understanding
- Enables view-consistent rendering
- **Omen could adopt**: 3DGS-style scene encoding in `scene_encoder.py`

### Pattern 4: Hybrid SSM+Attention (Trajectory Mamba / DriveMamba)
- Use SSM for long-range temporal dependencies (linear time)
- Use attention for spatial relationships (quadratic but precise)
- **Omen could adopt**: Replace or augment AR predictor's transformer with Mamba

### Pattern 5: Multimodal Conditioning (GAIA-1 / Cosmos)
- Condition world model on text + action + prior frames
- Enables controllable generation
- **Omen could adopt**: Condition predictions on Blender scene parameters (lighting, materials, camera)

---

## 7. Key Papers & Resources

| Paper/System | Year | Key Relevance |
|-------------|------|--------------|
| NVIDIA Cosmos WFM (arXiv:2501.03575) | 2025 | Dual diffusion+AR architecture |
| V-JEPA 2 (Meta) | 2025 | Latent-space video world model |
| LeWorldModel (arXiv:2603.19312) | 2025 | Stable end-to-end JEPA |
| DriveWorld (arXiv:2405.04390) | 2024 | 4D scene understanding via world models |
| GAIA-1 (arXiv:2309.17080) | 2023 | Generative transformer world model for driving |
| DriveMamba (arXiv:2602.13301) | 2025 | Mamba replacing Transformers in driving |
| Trajectory Mamba (arXiv:2503.10898) | 2025 | Hybrid SSM+Attention for prediction |
| Tesla FSD v14 | 2025 | Occupancy + BEV + Transformer + 3DGS |
| Awesome World Models (GitHub) | 2024-25 | Curated list of AD world models |

---

## 8. What Omen Should Prioritize

1. **Short-term**: Study NVIDIA Cosmos's dual diffusion+AR architecture. Omen's JEPA + AR predictor + denoiser mode already mirrors this pattern — validate the approach.

2. **Medium-term**: Consider 3D Gaussian Splatting integration in `scene_encoder.py` for richer 3D scene understanding, following Tesla's v14 approach.

3. **Medium-term**: Evaluate Mamba SSM as a replacement for the transformer in `arpredictor.py` for more efficient temporal prediction, especially for animation mode.

4. **Long-term**: Explore multimodal conditioning (GAIA-1 style) where the world model is conditioned on Blender scene graph parameters, enabling controllable prediction.
