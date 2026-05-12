# Mamba SSM vs Transformer for U-Net Bottleneck in Image Denoising

**Date:** 2026-05-12
**Context:** Omen render engine — deciding architecture for the U-Net bottleneck block at 4K resolution (3840x2160).

---

## 1. Key Mamba-Based Denoising/Restoration Architectures (2024-2025)

| Architecture | Venue | Task | Key Innovation |
|---|---|---|---|
| **MambaIR** | ECCV 2024 | Denoising, SR, deblurring | Residual State Space Block (RSB) + local enhancement + channel attention |
| **MambaIRv2** | CVPR 2025 | Denoising, SR, JPEG artifact reduction | Attentive State-Space Equation — non-causal modeling with single scan |
| **MaIR** | CVPR 2025 | Denoising, SR, deblurring | Locality- and continuity-preserving Mamba |
| **MambaVision** | CVPR 2025 | Vision backbone (classification, detection, segmentation) | Hybrid Mamba-Transformer from NVIDIA Labs |
| **VMamba** | NeurIPS 2024 | General 2D vision | Cross-scan module for 2D spatial traversal |
| **PlainMamba** | 2024 | Visual recognition | Non-hierarchical SSM, simplified 2D flattening |
| **MambaOut** | CVPR 2025 | Vision benchmarking | Paper argues CNN+MLP beats Mamba on ImageNet; Mamba better for long-sequence autoregressive tasks |
| **Swin-UMamba** | MICCAI 2024 | Medical segmentation | ImageNet-pretrained Mamba encoder in U-Net shape |
| **U-Mamba** | 2024 | Medical segmentation | First U-shaped Mamba model |

---

## 2. Computational Complexity Comparison

### At Full 4K Resolution (3840x2160 = 8,294,400 pixels)

| Metric | Self-Attention | Mamba SSM |
|---|---|---|
| Time complexity | O(n^2) | O(n) |
| Memory for n=8.3M tokens | ~68.8 trillion entries (infeasible) | Linear in n (feasible) |
| Practical at 4K? | **NO** — requires heavy patching/windowing | **YES** — processes full sequence |

### At U-Net Bottleneck (H/16 x W/16 = 240x135 = 32,400 tokens)

| Metric | Self-Attention | Mamba SSM |
|---|---|---|
| n^2 operations | ~1.05 billion | N/A |
| n operations | N/A | ~32,400 |
| Memory (attention matrix) | ~32,400^2 = ~1.05B entries (~4GB float32) | Constant hidden state (~few MB) |
| Practical? | **Borderline** — needs windowed/shifted attention (like Swin) | **Trivial** — full global context in O(n) |
| Wall-clock estimate | Slower by ~10-30x vs Mamba at this n | Fast |

### Critical Insight: At the bottleneck, n=32,400 is SMALL enough for windowed attention to work well, but also small enough that Mamba's O(n) advantage is modest. The advantage of Mamba at bottleneck scale is primarily **memory efficiency**, not raw FLOPs.

---

## 3. Quality Comparison: Mamba vs Transformer for Pixel Restoration

| Metric | Transformer (Restormer/NAFNet) | Mamba (MambaIR/MambaIRv2) |
|---|---|---|
| SIDD denoising PSNR | ~40.30 dB (NAFNet) | Competitive, within 0.1-0.3 dB |
| Gaussian denoising | Strong baseline | MambaIRv2 matches or slightly exceeds |
| Super-resolution | HAT/SRFormer SOTA | MambaIRv2 beats SRFormer by +0.35 dB PSNR with 9.3% fewer params |
| Convergence speed | Well-established | Faster training convergence claimed |
| Parameter efficiency | Baseline | Fewer parameters for comparable quality |

**Verdict:** MambaIRv2 reaches parity or slightly exceeds transformer baselines on standard benchmarks. Quality gap is negligible for practical purposes.

---

## 4. Global Scene Understanding (Caustics, Indirect Lighting)

| Capability | Transformer (Attention) | Mamba (SSM) |
|---|---|---|
| True global context | YES — every token attends every token | PARTIAL — sequential state propagation; distant dependencies decay |
| Non-causal bidirectional reasoning | Native (bidirectional attention) | Requires architectural additions (bidirectional scan, attentive SSM in MambaIRv2) |
| Precise spatial recall | STRONG — explicit position-aware attention | WEAKER — Harvard study shows SSMs inferior at precise copying/recall |
| Caustics from distant glass | Good — long-range pixel correlations | Adequate but may miss fine-grained long-range patterns |
| Indirect lighting bounce | Good — global aggregation | Good — summed state can capture accumulated light transport |
| Hidden state receptive field | N/A (full field) | Each hidden channel has finite effective receptive field (LongMamba finding) |
| Unidirectional limitation | N/A | S6 is unidirectional; 2D vision needs cross-scan (4 directions) or more |

**Key limitation from NeurIPS 2025 paper:** Mamba's difficulty stems not from the SSM module itself but from the **nonlinear convolution preceding it**, which fuses token information before the state update. This limits precise long-range token recall.

**Key finding from MambaOut (CVPR 2025):** Mamba's advantage is strongest for tasks requiring **long-sequence modeling with autoregressive properties**. For discriminative vision tasks at moderate resolutions, CNN+MLP can match or beat Mamba. However, image denoising at high resolution IS a long-sequence modeling task, so Mamba remains relevant.

---

## 5. 2D Vision Mamba Variants for Handling 2D Spatial Data

| Variant | 2D Strategy | Relevance to Omen |
|---|---|---|
| **VMamba** (NeurIPS 2024) | Cross-scan: 4 directional scans (left-right, right-left, top-down, bottom-up) | Most mature 2D Mamba; good reference |
| **2D-CrossScan** (AAAI 2025) | Multi-path spatially consistent hidden state propagation | Improved over VMamba's scanning |
| **PlainMamba** | Continuous 2D zigzag flattening | Simpler but less effective |
| **MambaVision** (NVIDIA, CVPR 2025) | Hybrid: Mamba blocks in early/mid stages, Transformer in final stage | **Directly relevant** — NVIDIA's own hybrid approach |
| **MambaIRv2** | Single-scan attentive SSM with non-causal modeling | Best for restoration tasks specifically |

---

## 6. Comparison Summary Table

| Criterion | Pure Transformer | Pure Mamba SSM | Hybrid (Mamba+Transformer) |
|---|---|---|---|
| Bottleneck complexity (n=32,400) | O(n^2) — manageable with windowing | O(n) — trivial | O(n) for Mamba stages, O(n^2) for transformer stage |
| Memory at bottleneck | ~4GB for full attention | ~few MB | Moderate |
| Denoising quality (PSNR) | Baseline SOTA | Matches SOTA | Can exceed both |
| Global context | Excellent | Good (with cross-scan) | Excellent |
| Precise long-range recall | Strong | Weaker | Strong (transformer catches what Mamba misses) |
| Caustics/indirect lighting | Good | Adequate | Best of both |
| Implementation maturity | Very mature | Emerging (2024-2025) | Emerging |
| Training stability | Well-understood | Newer, less documented | Needs tuning |
| Inference speed | Slower | 2-5x faster | Balanced |

---

## 7. Recommendation for Omen

### USE A HYBRID APPROACH: Mamba in the encoder/decoder, Transformer in the bottleneck.

**Rationale:**

1. **At the bottleneck (H/16 x W/16 = 32,400 tokens), self-attention IS feasible.** Windowed attention with shift (Swin-style) at this resolution costs ~1.05B operations — expensive but manageable on modern GPUs. Full attention would be ~4GB memory which is tight but possible on 8GB+ VRAM.

2. **Mamba's O(n) advantage is least impactful at the bottleneck** because n is already downsampled. The real win for Mamba is at higher resolutions (encoder/decoder paths) where n is large. But since Omen uses convolutions in the encoder/decoder already, this is less relevant.

3. **Global context matters most at the bottleneck.** Caustics from distant glass, multi-bounce indirect lighting, and inter-reflections require true global token interaction. Transformers handle this natively; Mamba relies on sequential state propagation that may miss fine-grained long-range correlations.

4. **MambaVision (NVIDIA, CVPR 2025) validates this hybrid approach.** NVIDIA's own architecture uses Mamba blocks in early/mid stages and adds Transformer layers in the final stage for tasks requiring global reasoning. This is exactly the pattern Omen should follow.

5. **MambaIRv2 shows Mamba alone is competitive for denoising**, but it adds significant architectural complexity (attentive SSM, non-causal modeling) to compensate for inherent SSM limitations. A simple transformer at the bottleneck achieves the same effect with less complexity.

### Proposed Architecture:

```
Encoder (Conv/ResNet)  →  Bottleneck  →  Decoder (Conv/ResNet)
                              │
                    ┌─────────┴──────────┐
                    │  2-4 Swin Transformer │
                    │  blocks (windowed     │
                    │  attention at H/16)   │
                    │  + optional Mamba     │
                    │  cross-scan block     │
                    └──────────────────────┘
```

### Specific Parameters:
- **Bottleneck resolution:** 240x135 for 4K input
- **Transformer blocks:** 2-4 Swin Transformer blocks, window size 8x8
- **Window attention cost:** 8x8=64 tokens per window, ~30x17=510 windows = trivial
- **Optional:** Add 1 VMamba cross-scan block before transformer for state aggregation
- **Skip Mamba in encoder/decoder:** Convolutions are sufficient at those levels; Mamba adds complexity without proportional benefit

### If GPU memory is very tight (<6GB VRAM):
- Replace transformer with MambaIRv2-style attentive SSM blocks
- Accept ~0.1-0.3 dB PSNR tradeoff for significantly lower memory
- Use 4-direction cross-scan (VMamba-style) for 2D spatial coverage

---

## Sources

- [MambaIR (ECCV 2024)](https://arxiv.org/html/2402.15648v3)
- [MambaIR GitHub (MambaIR + MambaIRv2)](https://github.com/csguoh/MambaIR)
- [MambaIRv2 (CVPR 2025)](https://arxiv.org/abs/2411.15269)
- [MaIR (CVPR 2025)](https://arxiv.org/pdf/2412.20066)
- [MambaVision (CVPR 2025)](https://arxiv.org/abs/2407.08083)
- [MambaVision GitHub (NVIDIA)](https://github.com/nvlabs/mambavision)
- [VMamba (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/baa2da9ae4bfed26520bb61d259a3653-Paper-Conference.pdf)
- [PlainMamba](https://arxiv.org/abs/2403.17695)
- [MambaOut (CVPR 2025)](https://arxiv.org/abs/2405.07992)
- [Swin-UMamba (MICCAI 2024)](https://arxiv.org/html/2402.03302v1)
- [2D-CrossScan (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/38855)
- [Essential Difficulties of Mamba (NeurIPS 2025)](https://ins.sjtu.edu.cn/people/xuzhiqin/pub/Mamba_NIPS_2025.pdf)
- [LongMamba](https://arxiv.org/html/2504.16053v1)
- [MobileMamba (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/He_MobileMamba_Lightweight_Multi-Receptive_Visual_Mamba_Network_CVPR_2025_paper.pdf)
- [Transformers Better at Copying (Harvard)](http://kempnerinstitute.harvard.edu/research/deeper-learning/repeat-after-me-transformers-are-better-than-state-space-models-at-copying/)
- [Mamba vs Transformers Efficiency](https://galileo.ai/blog/mamba-linear-scaling-transformers)
