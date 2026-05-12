# Literature Survey: Latent Decoders, JEPA Dimensions, Diffusion Decoders, Neural Rendering Scaling, and Physics-Based Constraints

**Date**: 2026-05-12
**Scope**: 2024-2026 papers from arXiv, CVPR, SIGGRAPH, NeurIPS, ICML

---

## 1. Advanced Decoders for JEPA / Latent-to-Image (4K Reconstruction)

### 1.1 Diffusion-4K (CVPR 2025)
- **Paper**: "Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models" (Zhang et al., CVPR 2025)
- **ArXiv**: 2503.18352
- **Key Insight**: Direct 4K (3840x2160) synthesis from text prompts using latent diffusion. Uses wavelet-based frequency decomposition to avoid the Conv2dTranspose bottleneck at ultra-high resolutions. The decoder operates in a multi-scale wavelet domain rather than direct pixel-space upsampling.
- **Relevance**: Shows that 4K decoding from small latents is feasible without naive Conv2dTranspose stacking.

### 1.2 Latent Wavelet Diffusion (LWD) (2025)
- **Paper**: "Latent Wavelet Diffusion: Enabling 4K Image Synthesis for Free"
- **OpenReview**: 5og80LMVxG
- **Key Insight**: A lightweight plugin framework that enables ANY latent diffusion model to scale to 2K-4K. Uses wavelet transforms to decompose latent data into frequency bands, then decodes each band independently. This avoids the quadratic cost growth of standard decoders.
- **Relevance**: Directly relevant as an architecture pattern: decode in wavelet space, then inverse-wavelet to pixels.

### 1.3 LSRNA: Latent Space Super-Resolution (CVPR 2025)
- **Paper**: "LSRNA: Latent Space Super-Resolution for Higher-Resolution Image Generation with Diffusion Models" (Jeong et al., CVPR 2025)
- **ArXiv**: 2503.18446
- **Key Insight**: Maps low-resolution reference latents directly onto a high-resolution manifold using a learned super-resolution network, followed by Region-Adaptive Noise (RNA) injection. Avoids generating high-res latents from scratch.
- **Relevance**: Two-stage approach: small latent -> super-resolved latent -> decode. Could pair with JEPA encoder.

### 1.4 Latent Upscale Adapter (LUA) (2025)
- **Paper**: "Latent Upscale Adapter"
- **ArXiv**: 2511.10629
- **Key Insight**: Lightweight adapter module performing super-resolution directly on the generator's latent code before decoding. Designed as a plug-in for existing diffusion models (e.g., Stable Diffusion). Minimal overhead.
- **Relevance**: Shows that latent-space upscaling (before decoding) is more efficient than pixel-space upscaling (after decoding).

### 1.5 DC-AE: Deep Compression Autoencoder (2024)
- **Paper**: "Deep Compression Autoencoder for Efficient High-Resolution Generation" (MIT Han Lab / NVIDIA)
- **ArXiv**: 2410.10733
- **Key Insight**: Achieves 32x and 64x spatial compression with high reconstruction quality. Uses architectural innovations (grouped residual blocks, multi-scale decoding) to maintain quality at extreme compression ratios. Applied to NVIDIA's Sana model.
- **Relevance**: Directly addresses the problem of extreme latent compression while preserving decode quality.

### 1.6 WF-VAE: Wavelet-Driven Energy Flow VAE (CVPR 2025)
- **Paper**: "WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow" (CVPR 2025, Poster #32779)
- **Key Insight**: Explicitly ablates 4, 8, 16, and 32 latent channels. Shows 16 channels is the sweet spot for reconstruction quality vs compute. Uses wavelet-driven energy flow for efficient spatial compression (factor f=16).
- **Relevance**: Provides the most comprehensive ablation study on latent channel sizing.

### 1.7 Conv2dTranspose Alternatives
- **DySample (CVPR 2024)**: Dynamic point-sampling based upsampling. Lower latency and memory than Conv2dTranspose. No checkerboard artifacts.
- **Pixel Shuffle / Sub-pixel Convolution**: Rearranges (C*r^2, H, W) -> (C, H*r, W*r). No checkerboard artifacts. Efficient and learnable.
- **Efficient Pixel-Dense Feature Upsampling with Local Attenders (arXiv 2025, 2601.17950)**: Iterative upsampling with local attention that competes with cross-attention methods at lower cost.
- **Spectral Artifacts in Upsampling (ECCV 2024, arXiv 2311.17524)**: Shows that large spatial context during upsampling provides stable, high-quality predictions and mitigates checkerboard artifacts from Conv2dTranspose.

### 1.8 Multi-Scale Progressive Approaches
- **MSPG-SEN (arXiv 2025, 2508.16089)**: Two-flow feedback multi-scale progressive GAN. Progressive growing from low to high resolution.
- **Multi-Scale Frequency VAE-GAN (IEEE 2025)**: Frequency-domain-aware feature extraction combined with VAE-GAN for multi-scale decoding.

### 1.9 NeRF-Inspired Decoders
- **Neural Brain Fields (arXiv 2601.00012)**: Demonstrates NeRF-inspired coordinate-based decoding for ultra-high resolution generation. Operates at "any desired resolution" by querying continuous coordinate-based functions.
- **Relevance**: Coordinate-based decoders inherently avoid the fixed-grid upsampling bottleneck of Conv2dTranspose.

### Key Takeaway for 4K Decoding
The dominant strategy in 2024-2025 is: **(1)** compress to small latent, **(2)** optionally super-resolve in latent space, **(3)** decode using wavelet-based or multi-scale approaches rather than naive transposed convolutions. Wavelet-domain decoding (Diffusion-4K, LWD) and latent-space super-resolution (LSRNA, LUA) are the most efficient paths to 4K.

---

## 2. Optimal Latent Dimension Size for Scene Understanding

### 2.1 I-JEPA Original Architecture (Assran et al., CVPR 2023)
- **Paper**: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
- **Latent Dimensions**:
  - ViT-Small: **384** embedding dimension
  - ViT-Base: **768** embedding dimension
  - ViT-Huge/14: **1024** embedding dimension (patch size 14x14 = 196 spatial elements)
- **Predictor Width**: Matches embedding dimension (384, 768, or 1024)
- **Key Insight**: The predictor operates ENTIRELY in latent space (not pixel space). The embedding dimension determines the capacity for scene understanding. 768 dims (ViT-B) is sufficient for strong ImageNet performance; 1024 dims (ViT-H) for state-of-the-art.

### 2.2 V-JEPA (Meta AI, 2024)
- **Paper**: "V-JEPA: Latent Video Prediction for Visual Representation Learning"
- **Key Insight**: Extends I-JEPA to video. Masking strategies FORCE scene-level understanding (object permanence, physics). The same 768-1024 latent dimensions encode temporal and spatial understanding. Achieves 1.5-6x training efficiency over pixel-reconstruction methods.
- **Relevance**: Shows that ~768-1024 dims is sufficient for VIDEO-LEVEL scene understanding (not just single images).

### 2.3 A-JEPA (arXiv 2311.15830)
- **Paper**: "A-JEPA: Joint-Embedding Predictive Architecture Can Listen"
- **Key Insight**: Extends JEPA to audio spectrum. Uses similar latent dimensions (384-768). Demonstrates that the JEPA latent space is domain-agnostic.

### 2.4 Latent Diffusion over JEPA Embeddings (2025)
- **Paper**: "Latent Diffusion over JEPA Embeddings for Conformal Time-Series Prediction" (arXiv 2605.00126)
- **Key Insight**: JEPA encoder maps inputs into a **64-dimensional latent space**. A conditional latent bridge generates predictions. Shows that even 64 dims can suffice for JEPA-style prediction when the task is more constrained (time-series).
- **Relevance**: 64 dims is a LOWER BOUND; scene understanding requires more.

### 2.5 NeRF Feature Dimensions (Baseline Comparison)
- **Original NeRF (Mildenhall et al., ECCV 2020)**:
  - 8 fully-connected layers, **256 channels** per layer
  - Input: 3D coordinate (x,y,z) + viewing direction (theta, phi) = 5 dims
  - Output: RGB (3) + density (1)
  - Intermediate features: 256 dims
- **KiloNeRF (ICCV 2021)**: Decomposes into thousands of tiny MLPs, each with smaller hidden dims (64-128)
- **Keypoint NeRF (CASE 2022)**: Uses **128-dimensional latent** with 4 hidden layers of 512 width
- **Takeaway**: NeRFs use 128-256 dimensional feature representations internally.

### 2.6 3D Gaussian Splatting Feature Sizes (Baseline Comparison)
- **Per-Gaussian attributes**: 59 float values total
  - Position: 3
  - Scale: 3
  - Rotation (quaternion): 4
  - Opacity: 1
  - SH coefficients (degree 3): 48 (= 16 coeffs x 3 RGB channels)
- **Compact 3DGS (NeurIPS 2024)**: Anchor-level context model achieves **100x+ size reduction** while maintaining quality
- **Feature 3DGS**: Extends to arbitrary-dimension semantic features per Gaussian
- **DCSHARP (WACV 2026)**: Replaces traditional SH with direction cosine SH for more compact representation
- **Takeaway**: ~59 floats per Gaussian primitive, dominated by SH coefficients (81%). Compression can reduce 100x.

### 2.7 VAE Latent Channel Ablation (from WF-VAE, CVPR 2025)
| Latent Channels | Reconstruction Quality | Compute Cost | Use Case |
|----------------|----------------------|-------------|----------|
| 4 | Max compression, detail loss | Lowest | SD 1.x, fast inference |
| 8 | Good balance | Moderate | SDXL, modern models |
| 16 | High fidelity | Higher | Recommended sweet spot |
| 32 | Best quality | Highest | DA-VAE, high-fidelity VAEs |

### Key Takeaway on Latent Dimensions
For scene understanding:
- **64 dims**: Minimum for constrained prediction tasks (time-series)
- **384 dims**: ViT-Small level understanding
- **768 dims**: ViT-Base level, sufficient for strong scene understanding
- **1024 dims**: ViT-Huge level, state-of-the-art
- NeRFs use 128-256 internal features; 3DGS uses ~59 floats per primitive
- For a JEPA-style scene encoder paired with a renderer, **256-768 dims** is the recommended range

---

## 3. Diffusion-Based Decoders Paired with JEPA or Learned Representations

### 3.1 D-JEPA: Denoising with Joint-Embedding Predictive Architecture (2024)
- **Paper**: "Denoising with a Joint-Embedding Predictive Architecture"
- **ArXiv**: 2410.03755 (October 2024)
- **OpenReview**: d4njmzM7jf
- **Architecture**: Three identical Vision Transformers: context encoder, target encoder, and feature predictor. Integrates JEPA principles INTO generative/denoising modeling.
- **Key Result**: Achieves lower FID scores with fewer training epochs. Scales well with compute (more GFLOPs -> consistently better FID).
- **Relevance**: This is THE paper on combining JEPA with diffusion. It bridges self-supervised representation learning with generative modeling.

### 3.2 Latent Diffusion over JEPA Embeddings (2025)
- **Paper**: arXiv 2605.00126
- **Key Insight**: JEPA encoder produces 64-dim latents; a conditional latent bridge (diffusion-based) generates predictions. Four sampling modes for the bridge.
- **Relevance**: Shows a concrete JEPA-encoder + diffusion-decoder pipeline, though for time-series (not images).

### 3.3 Diffusion Transformers with Representation Autoencoders (2025)
- **Paper**: arXiv 2510.11690
- **Key Insight**: Explores latent generative modeling where a pretrained autoencoder maps pixels into latent space for the diffusion process. The autoencoder's representation quality directly impacts downstream generation quality.
- **Relevance**: Shows that the encoder quality matters as much as the decoder for diffusion-based latent generation.

### 3.4 Encoder-Decoder Diffusion Language Models (NeurIPS 2025)
- **Paper**: NeurIPS 2025 Poster #119836
- **Key Insight**: Encoder represents clean tokens; decoder performs denoising. Accelerates discrete diffusion inference by reducing the iterative denoising burden on the decoder.
- **Relevance**: The encoder/decoder split is analogous to JEPA-encoder + diffusion-decoder for images.

### 3.5 Prometheus: 3D-Aware Latent Diffusion (CVPR 2025)
- **Paper**: CVPR 2025 Poster #33710
- **Key Insight**: Feedforward decoder combined with latent diffusion for text-to-3D generation. Shows that feedforward decoders can match diffusion quality when conditioned on good latent representations.
- **Relevance**: Feedforward decoding from learned latents CAN achieve high quality.

### 3.6 Lightweight Decoders for Diffusion Models (2025)
- **Paper**: "Toward Lightweight and Fast Decoders for Diffusion Models" (arXiv 2503.04871)
- **Key Insight**: Investigates replacing heavy VAE decoders with lightweight alternatives for both image and video diffusion. Studies the quality-speed tradeoff systematically.
- **Relevance**: Directly addresses decoder efficiency.

### 3.7 Diffusion Decoder Compute Cost at 4K
- **DiTFastAtten (arXiv 2406.08552)**: Attention compression that reduces compute cost, with GREATER savings at higher resolutions. Critical for 4K where attention is O(n^2).
- **DistriFusion (2024)**: Distributed parallel inference for high-resolution diffusion. Distributes the massive compute across multiple GPUs.
- **Edge-SD-SR (CVPR 2025, Poster #33249)**: First parameter-efficient diffusion model for super-resolution (~169M params). Targets edge devices.
- **NanoSD (arXiv 2601.09823)**: Pareto-optimal diffusion models distilled from SD 1.5 for real-time edge inference.
- **Rule of Thumb**: Per-token decode cost ~ 2x parameter count in FLOPs (transformer-based). Diffusion adds 20-50 iterative denoising steps on top.

### 3.8 Diffusion vs Feedforward Decoder Comparison
| Aspect | Diffusion Decoder | Feedforward (VAE/GAN) Decoder |
|--------|-------------------|-------------------------------|
| Speed | Slow (20-50 iterative steps) | Fast (single forward pass) |
| Quality | Higher fidelity, sharper details | Good but can blur fine details |
| Memory | Higher footprint | Lower requirements |
| 4K Feasibility | Very expensive (~50x feedforward) | Practical with wavelet/multi-scale |
| Use Case | Offline high-quality rendering | Real-time/interactive rendering |

### Key Takeaway on Diffusion Decoders
D-JEPA (2024) is the first paper to formally integrate JEPA with diffusion. The trend is toward hybrid architectures: JEPA-style encoders producing latent representations, then diffusion-based or feedforward decoders mapping to pixels. For 4K, feedforward decoders with wavelet/multi-scale architectures are far more practical than iterative diffusion decoding.

---

## 4. Model Scaling for Scene-Level Render Prediction

### 4.1 Grendel: Scaling Up 3DGS Training (arXiv 2024, 2406.18533)
- **Key Insight**: Distributed training system for large-scale 3DGS. The simple **sqrt(batch_size) scaling rule** for learning rates is highly effective. Shows that 3DGS scales well with compute when properly distributed.

### 4.2 Compact 3DGS with Anchor Level Context Model (NeurIPS 2024)
- **Key Insight**: Achieves **100x+ size reduction** compared to vanilla 3DGS while maintaining rendering quality. Uses context modeling at the anchor level.
- **Relevance**: Production neural renderers don't need to be as large as vanilla 3DGS.

### 4.3 LODGE: Level-of-Detail Large-Scale Gaussian Splatting (NeurIPS 2025)
- **Key Insight**: LOD method for 3DGS enabling real-time rendering of large-scale scenes. Addresses the key challenge that naive 3DGS scales poorly with scene size.
- **Relevance**: Shows that hierarchical/multi-resolution approaches are essential for scaling.

### 4.4 GameNGen: Diffusion Models as Real-Time Game Engines (Google DeepMind, 2024)
- **Paper**: arXiv 2408.14837
- **Key Insight**: First game engine powered entirely by a neural model. Uses a diffusion model to predict next frames in real-time. Demonstrates that neural rendering can be interactive.
- **Relevance**: Shows the scale of model needed for real-time scene-level prediction.

### 4.5 NVIDIA RTX Kit (GDC 2025)
- **Key Insight**: Production neural rendering suite including neural shaders, neural radiance caching, and DLSS 4. Integrated into Unreal Engine via NvRTX branch. Shows that neural rendering is production-ready.
- **Relevance**: Industry standard for production neural rendering.

### 4.6 Real-Time Neural Appearance Models (NVIDIA, ACM 2024)
- **Paper**: ACM DL 10.1145/3659577
- **Key Insight**: Complete system for real-time rendering of complex appearance using compact neural networks. Shows that production neural renderers can be relatively small (a few MB) when properly designed.

### 4.7 Model Size Benchmarks
| System | Model Size | Resolution | Speed |
|--------|-----------|------------|-------|
| Vanilla 3DGS | ~200MB-1GB+ (scene-dependent) | Real-time | 30-100+ FPS |
| Compact 3DGS (NeurIPS 2024) | ~2-10MB (100x reduction) | Real-time | Comparable |
| GameNGen | Large (diffusion-based) | 20+ FPS | Interactive |
| NVIDIA RTX Neural Rendering | Small (shader-level) | Real-time | 60+ FPS |
| Edge-SD-SR (CVPR 2025) | ~169M params | Variable | Edge-optimized |

### Key Takeaway on Model Scaling
Production neural renderers can be surprisingly compact (2-10MB for compressed 3DGS). The key is hierarchical representation (LOD, anchor-based compression) rather than brute-force model size. For scene-level render prediction, the encoder (JEPA) needs 768-1024 dims, and the decoder can be relatively lightweight if using multi-scale/wavelet approaches.

---

## 5. Physically-Based / Energy-Conserving Neural Rendering

### 5.1 PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields (CVPR 2025)
- **Paper**: Wu et al., arXiv 2412.09680
- **Venue**: CVPR 2025 (Poster #34210)
- **Key Contributions**:
  1. **Conservation of Energy Loss**: Explicitly penalizes BRDFs that reflect more energy than received. Makes the neural rendering physically valid.
  2. **NDF-weighted Specular Loss**: Promotes better specular decomposition by weighting the Normal Distribution Function.
- **Architecture**: Built on NeILF++. Jointly estimates geometry, materials, and lighting from posed images.
- **Key Insight**: You CAN enforce energy conservation as a differentiable loss in neural rendering. This eliminates physically invalid artifacts (over-bright reflections, energy gain).
- **Relevance**: Directly applicable to any neural renderer. The energy conservation loss is simple to implement and dramatically improves physical plausibility.

### 5.2 Neural Inverse Rendering with Physics-Based Light Transport (CMU PhD Thesis, 2025)
- **Author**: Battal, Carnegie Mellon University Robotics Institute
- **Key Insight**: Demonstrates combining physics-based light transport modeling with neural scene representations and specialized cameras. The thesis argues that neural rendering MUST respect light transport equations for reliable results.
- **Relevance**: Comprehensive theoretical framework for physics-informed neural rendering.

### 5.3 V-JEPA Intuitive Physics (Meta AI, 2024-2025)
- **Key Insight** (from LeCun's posts): V-JEPA, trained entirely self-supervised on video, develops an understanding of **intuitive physics** (object permanence, gravity, collisions) without any physics supervision. This is emergent from the JEPA latent prediction objective.
- **Relevance**: Shows that JEPA-style latent prediction naturally encodes physical constraints. No explicit physics losses needed for basic physical understanding.

### 5.4 Energy-Based Models and JEPA (LeCun, 2023-2024)
- **Paper**: "Introduction to Latent Variable Energy-Based Models" (arXiv 2306.02572, published in Journal of Physics A)
- **Key Insight**: JEPA operates as an Energy-Based Model (EBM) on representations. Valid predictions map to LOW energy; invalid/impossible predictions map to HIGH energy. This provides a natural framework for encoding physical constraints: physically impossible scene states would have high energy.
- **Relevance**: The EBM framework in JEPA IS the mechanism for enforcing physical constraints. Rather than explicit loss terms, the energy landscape naturally penalizes physically invalid predictions.

### 5.5 LeJEPA Variant (2024-2025)
- **Key Insight**: A mathematically grounded JEPA variant that replaces training heuristics with explicit regularization. Provides better theoretical guarantees on the learned latent space.
- **Relevance**: Could be extended to include physical regularization terms.

### 5.6 Analytical & Neural Approaches to Physically Based Rendering (SIGGRAPH Asia 2023)
- **Key Insight**: Bridging analytical path tracing with neural techniques. Shows that neural rendering and physics-based rendering are converging -- the best results combine both.

### Key Takeaway on Physics-Based Constraints
Two complementary approaches exist:
1. **Explicit physics losses** (PBR-NeRF style): Add energy conservation, BRDF validity, and light transport losses directly to the training objective. Proven to work, simple to implement.
2. **Implicit physics from JEPA** (V-JEPA style): The JEPA prediction objective naturally learns physical constraints from data. No explicit physics needed, but less controllable.

The recommended approach for a production system: **Use JEPA-style latent prediction for scene understanding, then enforce explicit energy conservation losses (from PBR-NeRF) in the decoder/rendering stage.**

---

## Summary: Recommended Architecture for JEPA-Based Scene Rendering

Based on the surveyed literature:

### Encoder (Scene Understanding)
- **Architecture**: JEPA/V-JEPA style Vision Transformer
- **Latent dims**: 768 (ViT-Base) for balanced performance, 1024 (ViT-Huge) for maximum quality
- **Patch size**: 14x14
- **Training**: Self-supervised latent prediction (no pixel reconstruction)

### Latent Space
- **Dimension**: 768-1024 for semantic understanding
- **Compression**: If pairing with a VAE decoder, use 16 latent channels at the spatial bottleneck (WF-VAE ablation)
- **Optional latent super-resolution**: LUA or LSRNA for scaling up before decoding

### Decoder (Latent-to-4K Image)
- **Recommended**: Wavelet-based decoder (Diffusion-4K / LWD pattern) or multi-scale progressive decoder
- **NOT recommended**: Naive Conv2dTranspose stacking (checkerboard artifacts, poor scaling)
- **Alternatives**: Pixel Shuffle + convolution, DySample for dynamic upsampling
- **For 4K**: Two-stage (latent SR + decode) is more efficient than single-stage

### Physics Constraints
- **Encoder stage**: Implicit via JEPA energy-based formulation (physical impossibilities = high energy)
- **Decoder/rendering stage**: Explicit energy conservation loss (PBR-NeRF pattern)
- **Combined**: Best of both worlds -- learned physical understanding + enforced physical validity

### Model Size Target
- **Encoder**: ~86M params (ViT-Base) to ~632M params (ViT-Huge)
- **Decoder**: ~20-50M params with wavelet/multi-scale architecture
- **Total**: ~100-700M params depending on quality target
- **Compressed 3DGS for scene**: 2-10MB (100x compressed)
