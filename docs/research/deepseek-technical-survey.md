# DeepSeek Technical Breakthroughs — Complete Survey

## Omen Research Survey — May 2026

---

## Paper Timeline & Cost-to-Performance

| Paper | Date | Total Params | Active Params | Training Cost | Key Innovation |
|-------|------|-------------|---------------|---------------|----------------|
| DeepSeek LLM (67B dense) | Jan 2024 | 67B | 67B | ~$10M+ | Baseline dense model |
| DeepSeekMoE | Jan 2024 | 16.4B | 2.4B | Much cheaper | Fine-grained expert + shared expert |
| DeepSeek-V2 | May 2024 | 236B | 21B | 42.5% cheaper than 67B dense | MLA + DeepSeekMoE |
| DeepSeek-V3 | Dec 2024 | 671B | 37B | $5.6M (2.788M H800 GPU-hrs) | Auxiliary-loss-free + MTP + FP8 |
| DeepSeek-R1 | Jan 2025 | 671B | 37B | $5.6M base + ~$294K RL | Pure RL reasoning, no SFT needed |

### The "DeepSeek Shock"

| Model | Training Cost | Comparable To | Cost Ratio |
|-------|--------------|---------------|------------|
| DeepSeek-V2 (236B) | ~$6M | Llama-3-70B | ~20% cost |
| DeepSeek-V3 (671B) | $5.6M | GPT-4o, Claude 3.5 | ~3-5% cost |
| DeepSeek-R1 (671B) | $5.6M + $294K RL | OpenAI o1 | ~3% cost |
| R1-Distill-32B | ~$100K | GPT-4o on math | ~0.1% cost |

V3 trained for **$5.6M** and matched models costing $100M-1B to train. R1 achieved OpenAI o1-level reasoning at **3% of the cost**.

---

## Innovation 1: Multi-Head Latent Attention (MLA)

**Source**: [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (May 2024)

### The Problem

Standard Multi-Head Attention (MHA) caches K and V matrices per token per layer. For a model with `n_h` heads, `d_h` head dim, at sequence length L:

```
KV cache per layer = 2 × n_h × d_h × L × bytes_per_element
```

At 128K context with 128 heads × 128 dim = **2GB KV cache per layer**. For a 60-layer model: 120GB of KV cache alone.

### The Solution: Low-Rank Joint KV Compression

MLA compresses K and V into a **single low-rank latent vector**:

```
Standard MHA:  cache stores [K_1, V_1, K_2, V_2, ...]  → 2 × n_h × d_h per token
MLA:           cache stores [c_KV]                       → d_c per token (d_c << 2×n_h×d_h)
```

**Mechanism:**

1. **Down-projection (compression):**
   ```
   c_KV = W_DKV(h_t)    # W_DKV is (d_c × d_model), d_c = 512 vs d_model = 4096
   ```

2. **Up-projection (reconstruction at inference):**
   ```
   (k_t, v_t) = W_UK(c_KV) × W_K    # reconstruct K and V from latent
   # W_UK is (d_model × d_c), W_K is (d_h × d_model)
   ```

3. **Query also compressed:**
   ```
   c_Q = W_DQ(h_t)       # down-project query
   q_t = W_UQ(c_Q) × W_Q  # up-project for attention
   ```

### RoPE Complication & Decoupled Solution

RoPE (Rotary Position Embedding) is applied to Q and K. But if K is in compressed latent form, RoPE can't be applied to the latent without losing the compression benefit.

**DeepSeek's solution: Decoupled RoPE**
- Apply RoPE to a **separate small projection** of K (not the compressed latent)
- `q_t` gets both: standard attention q from latent + RoPE query from separate projection
- `k_t` gets both: reconstructed K from latent + RoPE key from separate projection
- The RoPE portion is small (`d_R = 64` dims) so it doesn't dominate the cache

```
q_rope = W_Q_R(h_t)     # small separate projection for RoPE query (d_R dims)
k_rope = W_K_R(h_t)     # small separate projection for RoPE key (d_R dims)

# Final attention uses both:
attention = softmax( (q × k^T + q_rope × k_rope^T) / sqrt(d) ) × v
```

### Cache Savings

```
MHA KV cache per token: 2 × n_h × d_h = 2 × 128 × 128 = 32,768 elements
MLA KV cache per token: d_c + d_R = 512 + 64 = 576 elements
Reduction: 93.3%  (from DeepSeek-V2 paper)

At 128K context: 128K × 576 × 2 bytes ≈ 150MB vs 2GB+ for MHA
```

### Key Insight: MLA = LoRA Applied to Attention Cache

MLA borrows from LoRA (Low-Rank Adaptation): instead of caching full-rank K,V matrices, cache a low-rank latent and reconstruct on-the-fly. The "adaptation" IS the cache.

### Relevance to Omen

Omen's U-Net encoder/decoder processes features at full resolution. MLA-style compression could:
- Reduce memory for skip connections (6GB at 4K)
- Compress feature maps stored between encoder and decoder passes
- Enable longer temporal history in ARPredictor without memory blowup

---

## Innovation 2: DeepSeekMoE — Fine-Grained Expert Segmentation + Shared Experts

**Source**: [DeepSeekMoE](https://arxiv.org/abs/2401.06066) (Jan 2024, ACL 2024)

### The Problem with Standard MoE

1. **Knowledge Hybridity**: Each expert handles diverse knowledge → not specialized enough
2. **Knowledge Redundancy**: Multiple experts learn overlapping information → wasted params
3. **Load Imbalance**: Some experts get over-selected, others underutilized

### Solution 1: Fine-Grained Expert Segmentation

Instead of N large experts, use N×F smaller experts, activating the same total params:

```
Standard MoE:  16 experts, top-2 routing  → 2 × 4096 = 8,192 active params
DeepSeekMoE:   64 experts, top-8 routing  → 8 × 512 = 4,096 active params (same compute)
```

Each expert is more specialized (narrower focus), routing is more precise.

### Solution 2: Shared Expert Isolation

Add K "shared experts" that are **always active** (no routing). They capture common knowledge:

- **Shared experts**: learn universal patterns (syntax, common reasoning)
- **Routed experts**: learn specialized patterns (domain-specific knowledge)

```
DeepSeek-V2: 2 shared + 64 routed, top-6 routing
DeepSeek-V3: 1 shared + 256 routed, top-8 routing
```

### Result

DeepSeekMoE-16B matches DeepSeek-67B (dense) performance at ~40% of the compute.

### Relevance to Omen

MoE concepts could apply to Omen's tier system:
- Shared experts = base denoising capability (always active)
- Routed experts = scene-specific features (glass caustics, volumetrics, hair, SSS)
- Fine-grained segmentation = specialized kernels for different noise patterns
- At inference, only activate relevant experts → Omen-Fast uses fewer experts, Omen-High uses more

---

## Innovation 3: Auxiliary-Loss-Free Load Balancing

**Source**: [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (Dec 2024)

### The Problem

MoE routing needs load balancing (all experts should get similar usage). Traditional approach:

```
L_total = L_main + α × L_aux_balance
```

- Small α: insufficient balancing → some experts idle
- Large α: hurts model performance → gradient interference with main objective
- This is a **fundamental tradeoff** in all MoE models before V3

### The Solution: Bias-Based Dynamic Adjustment

Instead of a loss term, maintain a **bias vector** `b_i` for each expert:

```
Routing: p_i = softmax(topK_scores + b_i)
```

**Update rule** (after each training step):
```
if expert i overloaded (usage > target):  b_i -= γ   # discourage selection
if expert i underloaded (usage < target): b_i += γ   # encourage selection
```

- γ (step size) ≈ 0.001 per step
- **b_i does NOT participate in gradient computation** → zero interference with training loss

### Why This Matters

1. No gradient interference → model quality is NOT compromised for balance
2. Simpler training — no hyperparameter tuning of α
3. More stable — no loss spikes from auxiliary loss

### Relevance to Omen

If Omen adopts MoE for scene-type specialization:
- Auxiliary-loss-free balancing avoids quality/performance tradeoff
- Bias adjustment is simple to implement in Nabla autograd
- No extra loss term simplifies the already multi-objective training (L_denoise + L_energy + L_sigreg)

---

## Innovation 4: Multi-Token Prediction (MTP)

**Source**: [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (Dec 2024)

### The Problem

Standard LLM training: predict next token only. Wastes information from future tokens.

### The Solution

Predict D future tokens simultaneously using sequential prediction heads:

```
Main head:    predict token t+1 (standard)
MTP head 1:   predict token t+2 (conditioned on t+1 prediction)
MTP head 2:   predict token t+3 (conditioned on t+1, t+2)
...
```

Each MTP head shares the main trunk but has its own output projection.

### Training Loss

```
L = L_main + λ × Σ(L_mtp_i)    where λ ≈ 0.3 per head
```

### Speculative Decoding Speedup

MTP enables **1.8× faster inference** via speculative decoding (draft-then-verify).

### Relevance to Omen

MTP maps directly to Omen's temporal prediction:
- ARPredictor already predicts future frames → MTP gives multi-frame lookahead
- Instead of predicting frame N+1 only, predict N+1, N+2, N+3 simultaneously
- Enables speculative rendering: draft 3 future frames, verify with 1spp render
- MTP heads share the ARPredictor trunk → minimal extra cost

---

## Innovation 5: FP8 Mixed Precision Training + DualPipe

**Source**: [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (Dec 2024)

### FP8 Strategy

```
Forward pass:   FP8 for linear layers (matmuls), BF16 for attention softmax
Backward pass:  BF16 for gradient computation, FP8 for gradient communication
Quantization:   Online per-tile scaling (not per-tensor)
Formats:        FP8 E4M3 for forward, E5M2 for backward
```

### DualPipe — Overlapping Communication

Pipeline parallelism that overlaps forward and backward communication:
- While GPU computes forward on chunk N, it sends backward gradients for chunk N-K
- Eliminates pipeline bubbles → near-linear scaling across GPUs

### Relevance to Omen

Omen targets BF16 precision (per spec). FP8 could:
- Halve VRAM for activations (4.5GB → ~2.25GB at 4K)
- Speed up U-Net inference by 1.5-2× with FP8 matmuls
- Per-tile quantization aligns with Omen's tiled processing strategy
- DualPipe concept applies to overlapping Mitsuba rendering with JEPA inference

---

## Innovation 6: DeepSeek-R1 — Pure RL Reasoning

**Source**: [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (Jan 2025), [Nature](https://www.nature.com/articles/s41586-025-09422-z)

### The Breakthrough

**DeepSeek-R1-Zero**: Skip SFT entirely, train reasoning with ONLY reinforcement learning.

```
Standard approach:  SFT on human CoT data → RL on top
R1-Zero approach:   Skip SFT, go straight to RL with GRPO
```

### GRPO (Group Relative Policy Optimization)

Instead of PPO with value function (expensive), use **group-relative rewards**:

1. Sample G outputs for each prompt
2. Score each output with reward model
3. Normalize rewards within group (relative ranking)
4. Update policy to increase probability of high-reward outputs

No value function needed → **50% less memory** during RL training.

### Emergent Chain-of-Thought

Without ANY human CoT data, the model discovers reasoning patterns:
- **Self-verification**: "let me check this again"
- **Self-correction**: "wait, that's wrong, let me recalculate"
- **Exploration**: trying multiple approaches before committing

Average response length grew from 100 to **8,000+ tokens** during RL training.

### Distillation to Smaller Models

```
R1-Distill-Qwen-32B:  beats GPT-4o on several math benchmarks
R1-Distill-Llama-8B:  strong reasoning in small footprint
```

Key finding: **distillation from R1 is MORE effective than RL on small models directly**.

### Relevance to Omen

R1's approach validates Omen's self-supervised training strategy:
- Omen generates its own training data (render pairs) — no human annotation needed
- R1 proves pure RL/self-supervised can match supervised approaches
- Distillation concept: train a large Omen-High model, distill to Omen-Fast
- GRPO-style group rewards could apply to denoising quality scoring
- Emergent behavior: JEPA surprise detection is analogous to R1's self-verification

---

## Cross-Cutting Relevance to Omen Architecture

| DeepSeek Innovation | Omen Application | Impact |
|---------------------|------------------|--------|
| MLA (93.3% KV cache reduction) | Compress U-Net skip connections at 4K | 6GB → ~400MB skip memory |
| Fine-grained MoE | Scene-type specialized denoising experts | Activate only relevant experts per scene |
| Shared experts | Base denoising always active | Common patterns free, specialized on demand |
| Auxiliary-loss-free | MoE load balancing without quality loss | Simpler multi-objective training |
| Multi-Token Prediction | Multi-frame temporal lookahead | 1.8× faster speculative rendering |
| FP8 + DualPipe | Half VRAM, overlap render+denoise | 4.5GB → 2.25GB, concurrent pipeline |
| GRPO / Pure RL | Self-supervised JEPA training | No human annotation for denoising quality |
| Distillation | Omen-High → Omen-Fast model compression | Train big, deploy small |
| Decoupled RoPE | Position encoding in compressed space | Apply spatial encoding to latent features |

---

## Sources

- [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434) — MLA + DeepSeekMoE
- [DeepSeekMoE Paper](https://arxiv.org/abs/2401.06066) — Fine-grained expert segmentation
- [DeepSeek-V3 Paper](https://arxiv.org/abs/2412.19437) — Auxiliary-loss-free + MTP + FP8
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948) — Pure RL reasoning
- [MLA Architecture Detail — Sebastian Raschka](https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/)
- [MLA Deep Dive — Lior Sinai](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html)
- [MLA Analysis — dataturbo/Medium](https://dataturbo.medium.com/deepseek-technical-analysis-2-mla-74bdb87d4ad2)
- [Auxiliary-Loss-Free Load Balancing — GoPubby](https://ai.gopubby.com/deepseek-v3-explained-3-auxiliary-loss-free-load-balancing-4beeb734ab1f)
- [DeepSeek-V3 MoE + MTP — Yugen.ai](https://medium.com/yugen-ai-technology-blog/deepseek-v3-advances-in-moe-load-balancing-and-multi-token-prediction-training-f6d68c59749c)
- [DeepSeek-R1 Cost Analysis — VentureBeat](https://venturebeat.com/ai/deepseek-r1s-bold-bet-on-reinforcement-learning-how-it-outpaced-openai-at-3-of-the-cost)
- [DeepSeek-R1 in Nature](https://www.nature.com/articles/s41586-025-09422-z)
