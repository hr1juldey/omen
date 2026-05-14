# Omen — Honest Architecture Audit

> Source: Hermes conversation 2026-05-13. Full technical critique of the Omen
> scene-aware JEPA render accelerator. The goal is not to cut features — it is
> to keep everything but implement it properly.

---

## Context: Why This Matters

Constitutional AI self-play (generate → critique → revise → train on revisions)
requires a model good enough to **meaningfully** critique its own output.
Below ~10-15B parameters, self-critique is garbage for general language tasks.

But this is only true for **general language tasks**. The self-critique threshold
is lower in bounded domains. A 3B model CAN meaningfully critique whether a
denoised render preserved edges or hallucinated detail — because the domain is
constrained, the evaluation criteria are clear, and "good vs. bad" is measurable
(SSIM, PSNR, confidence maps).

Constitutional-style training SHOULD work at smaller scales for Omen's domain.
The reason nobody has done it:

1. **Academic incentives**: nobody publishes "small model in narrow domain" papers
2. **Commercial incentives**: Anthropic isn't going to publish a recipe for making
   Claude-equivalent small models
3. **The knowledge gap**: you need someone who understands both the neuroscience
   AND the domain AND the ML — that intersection is rare

---

## PART 1: What Omen Is Doing CORRECT

These are things that are genuinely right and should be kept.

### 1.1 JEPA Latent Prediction Instead of Pixel Reconstruction

This is the **single most important correct decision**. The model doesn't
reconstruct pixels — it predicts in latent space what a clean render looks like.
This IS predictive coding. This IS what cortical columns do. This is the core
insight that separates Omen from every DLSS/OptiX competitor.

### 1.2 Scene Graph Conditioning

Giving the model knowledge of geometry, materials, lights, BVH — not just noisy
pixels — is structural context. The brain does this. You don't look at a noisy
image and guess blindly — you know the 3D world. Omen knows the 3D world.

### 1.3 Online Self-Supervised Learning During Rendering

The model learns from every render pair (noisy → clean). This is continual
learning. Not offline training on a static dataset. The right paradigm, even if
the implementation has gaps.

### 1.4 Surprise Detection as a Signal

`temporal.py` computes z-scored surprise from MSE deviation. The signal exists.
The concept is right. Detecting when the scene has changed enough that the model
needs to adapt.

### 1.5 The Confidence Head

Per-pixel confidence estimation is uncertainty quantification. The brain does
this constantly — you're more uncertain about things in your peripheral vision,
in low light, in unfamiliar contexts.

### 1.6 The Closed-Loop Architecture

Mitsuba renders → Omen denoises → Omen trains on the pair. This is an embodied
loop. Not iid training. The model is in continuous interaction with its
environment. Fundamentally different from how GPT was trained.

**Score: 6/6 correct ideas.**

---

## PART 2: What Omen Is Doing WRONG

Actual mistakes — things that will cause problems.

### 2.1 One Optimizer for All 23 Experts

```python
# trainer.py
self.optimizer = nn.optim.AdamW(model, lr=5e-5, weight_decay=1e-3)
```

This single line undermines the entire MoE architecture. You built 23
specialized experts with cryptomatte routing — and then you train them all with
the **same learning rate**. The caustics expert that has converged gets updated
as aggressively as the volumetrics expert that just started learning.

**The fix is trivial.** Per-expert parameter groups with individual learning
rates. ~30 lines. But without it, MoE is decoration — the experts can't actually
specialize because they all learn at the same rate and interfere with each other.

**Engineering detail:**

```python
# What exists (WRONG):
optimizer = AdamW(model.parameters(), lr=5e-5)

# What should exist:
param_groups = []
for i, expert in enumerate(model.experts):
    # Each expert gets its own lr based on update count / convergence
    param_groups.append({
        'params': expert.parameters(),
        'lr': per_expert_lr(i),  # e.g. cosine decay per-expert
    })
# Shared components (scene encoder, etc.) get their own group
param_groups.append({
    'params': model.shared.parameters(),
    'lr': 5e-5,
})
optimizer = AdamW(param_groups)
```

### 2.2 The Replay Buffer Is Cosmetic

```python
# lora_manager.py
self._replay_buffer = deque(maxlen=50)
REPLAY_SAMPLES_PER_STEP = 3
```

50 items. 3 samples per step. This is not sleep replay. This is a band-aid.
The hippocampus doesn't keep 50 memories — it keeps a dense episodic trace and
replays it interleaved with thousands of old memories during NREM sleep.

**The buffer should be:**
- At minimum: **500+ items**, stratified by material type
- Replay ratio: **1:1** with new data (not 3 per step)
- Explicitly interleaved: every training step should have **50% new data,
  50% replay**

**Engineering detail:**

```
Current: deque(maxlen=50), 3 samples/step, no stratification
         → buffer is FIFO, oldest evicted regardless of importance
         → no guarantee of material diversity

Required:
  - StratifiedBuffer: maintains per-material-type sub-buffers
  - Sampling: randomly pick material type → randomly pick item from that type
  - Ratio: 1 new sample : 1 replay sample per training step
  - Size: 500 minimum (covers ~10 diverse scenes × ~50 tiles each)
```

### 2.3 SIGReg Is Overengineered

```python
# sigreg.py — 17 quantile knots, Epps-Pulley two-sample test,
# 1024 random projections for slice Wasserstein
```

Preventing variance collapse in the JEPA latent space is correct. But using an
academic-grade statistical test as a differentiable component is wrong. At 8M
parameters, a simple cosine similarity penalty between latent vectors would
achieve the same thing. SIGReg is ~400 lines solving a problem that ~20 lines
could solve at this scale.

**Engineering detail:**

```python
# What exists: SIGReg with Epps-Pulley two-sample test
#   - 17 quantile knots
#   - 1024 random projections for slice Wasserstein
#   - Custom Mojo GPU kernel
#   - ~400 lines total

# What would work at 8M params:
def simple_variance_reg(latent, eps=1e-6):
    std = latent.std(axis=0)
    return -nb.mean(nb.log(std + eps))  # prevent collapse, ~3 lines
```

### 2.4 The Surprise Signal Is Computed and Thrown Away

```python
# temporal.py
is_surprise = z_score > SURPRISE_THRESHOLD  # → used only for re-anchoring
# NEVER connected to the optimizer
# NEVER modulates learning rate
# NEVER gates which experts get updated
```

You built a neuromodulation signal. It's the right signal. And then you didn't
wire it to anything that matters. The brain uses surprise (acetylcholine) to
**increase plasticity**. Your surprise detector doesn't touch the optimizer.

**Engineering detail:**

```python
# What exists: surprise → boolean flag → re-anchor only

# What should exist: surprise → modulate learning rate
for param_group in optimizer.param_groups:
    base_lr = param_group['base_lr']
    # Surprise increases learning rate (acetylcholine analog)
    param_group['lr'] = base_lr * (1.0 + surprise_scale * z_score)
```

**Score: 4/4 wrong implementations identified.**

---

## PART 3: What Omen Is Doing CORRECT but in the WRONG WAY

The most important ones. Ideas that are right but execution defeats the purpose.

### 3.1 MoE Routing on Tile Fingerprints Instead of Scene Graph Properties

You route experts based on cryptomatte-style tile fingerprints (pixel-level
material statistics). This works, but you're **ignoring your own best asset** —
the scene graph.

You KNOW what material each pixel is.
You KNOW what light type is illuminating it.
You're **deriving from pixels what you already have as structured data.**

```python
# Current: tile fingerprint → route to expert
# Better:  scene_graph.material_id + light_type → route to expert
```

The scene graph is your structural context advantage. Using pixel-level routing
is like **having a textbook open during an exam but answering from memory
anyway.**

**Engineering detail:**

```
Current pipeline:
  render → extract tile pixels → compute fingerprint (mean/std of RGBA,
  albedo, normal) → route to expert

What should happen:
  scene_graph → {material_id, light_type, geometry_tag} per tile
  → direct routing table lookup → expert assignment

The scene graph from Blender/Mitsuba contains:
  - Per-object material assignments (exact shader type + parameters)
  - Per-face geometry tags (smooth/sharp, manifold/non-manifold)
  - Per-light type (point, area, spot, sun, environment)
  - BVH structure (spatial coherence)

This is GROUND TRUTH. Cryptomatte fingerprints are a LOSSY APPROXIMATION
of the same information, derived after rendering instead of before.

Why this matters for training:
  - Pixel-derived routing introduces noise into expert assignments
  - Noisy routing → noisy gradients → slower/no expert specialization
  - Scene-graph routing → deterministic expert assignments → clean gradients
  - Clean gradients → faster convergence, better specialization
```

### 3.2 LoRA Adaptation Instead of Architectural Dual Memory

`lora_manager.py` implements fast adaptation through LoRA adapters with a cache
and replay buffer. This is trying to be the hippocampus. But LoRA is the wrong
abstraction:

- LoRA adds low-rank matrices to existing weights → it's a **perturbation**,
  not a separate memory system
- The "fast store" (LoRA adapter) and "slow store" (base model) share the same
  forward pass → they **can't have different learning rates architecturally**
- Consolidation is **binary**: an adapter is either in `_training_cache` or in
  `_consolidated` — there's no graded consolidation

**The right implementation:** a small separate "episodic" network (literally a
2-layer MLP with ~100K params) that receives the noisy render + scene graph and
produces a **correction** to the main model's output. Fast to train,
architecturally separate, doesn't touch the main model during inference unless
the scene is novel.

**Engineering detail:**

```
LoRA problems:
  1. LoRA(A, B) modifies W as W + AB — same forward pass, same gradient flow
  2. Can't have different optimizers for base vs. adapter weights
  3. Consolidation = merging AB into W — binary, no graded transfer
  4. Multiple scenes compete for the same weight space via low-rank slots

Episodic correction network:
  class EpisodicCorrection(nn.Module):
      def __init__(self, dim=192, hidden=256):
          self.net = nn.Sequential(
              nn.Linear(dim * 2, hidden),  # [main_output, scene_context]
              nn.SiLU(),
              nn.Linear(hidden, dim),
          )
      def forward(self, main_output, scene_context):
          return main_output + self.net(cat([main_output, scene_context]))

  - Separate parameters, separate optimizer, separate learning rate
  - Can be 10x faster to train (100K params vs full model)
  - Graded consolidation: gradually reduce correction magnitude
  - When scene is familiar, correction → 0 (no inference overhead)
```

### 3.3 The ARPredictor Is Solving Tomorrow's Problem Today

```python
# arpredictor.py — 6-layer ConditionalBlock with AdaLN-zero
# Ported from LeWorldModel for temporal prediction
```

The first mode is denoising a **single frame**. The ARPredictor is designed for
temporal sequences — predicting frame N+1 from frames 1..N. You've built the
temporal architecture **before validating the spatial architecture works.**

**The correct order:**
1. Validate single-frame denoising with scene conditioning (no ARPredictor)
2. Validate confidence-guided adaptive sampling (no ARPredictor)
3. THEN add temporal prediction with ARPredictor

Right now the ARPredictor adds ~2M parameters and complexity to a model that
hasn't proven its core thesis yet.

**Engineering detail:**

```
ARPredictor costs:
  - ~2M parameters (4 ConditionalBlocks × ~500K each)
  - SceneDeltaEncoder: ~155K params
  - Positional embeddings: small
  - Total: ~2.2M params added before spatial denoising is validated

When ARPredictor IS needed (animation mode):
  - Input: history window [latent_{N-3}, ..., latent_{N-1}, current]
  - Conditioning: scene delta embedding
  - Output: predicted next latent
  - Used to: guide temporal coherence, reduce flicker

When ARPredictor is NOT needed (single-frame denoising):
  - Every parameter adds inference latency
  - Every ConditionalBlock adds attention computation
  - SceneDeltaEncoder processes deltas that don't exist for single frames
```

### 3.4 Multi-Mode Architecture Before Single-Mode Validation

You built four modes (denoiser, accelerator, upscaler, temporal) simultaneously.
Each has its own pipeline in `src/omen/modes/`. The result: **four half-validated
modes instead of one fully-validated mode.**

The biological parallel: evolution doesn't evolve four capabilities
simultaneously. It evolves one, validates it, then builds on it. Your caudate
nucleus wasn't built the same time as your prefrontal cortex — one came first,
proved useful, and then the other developed on top.

**Engineering detail:**

```
Current mode files:
  src/omen/modes/denoiser.py    — render pipeline with training + SSIM gate
  src/omen/modes/adaptive.py    — confidence-guided sample allocation
  src/omen/modes/multires.py    — progressive resolution upscaling
  src/omen/modes/animation.py   — delegates to temporal.py for delta/surprise

All share helpers from denoiser.py (_render_with_aov, etc.)

Problem: each mode needs its own validation pipeline:
  - Denoiser: SSIM/PSNR vs OIDN/OptiX on 10+ scenes
  - Adaptive: sample allocation efficiency vs uniform sampling
  - Multires: quality vs progressive rendering at different resolutions
  - Animation: temporal coherence (flicker metrics) across frame sequences

None of these have been validated on real rendered data yet.
```

**Score: 4/4 correct-but-wrong-way identified.**

---

## PART 4: What Omen Is Doing TOO MUCH to Achieve TOO LITTLE

The painful part. The honest assessment.

### 4.1 MoE at 8M Parameters Is Premature

```python
# moe.py: 23 experts, top-k routing, auxiliary-loss-free load balancing
```

MoE matters when your model is so large that activating all parameters every
forward pass is too expensive. At 8M parameters, the **ENTIRE model fits in L2
cache**. There's no sparsity benefit. Every expert is ~350K parameters. You're
paying the routing overhead (computing fingerprints, dispatching, gathering) for
a model that could just be dense.

MoE at 8M parameters is like **putting a traffic light at the end of your
driveway.** Technically correct. Practically unnecessary.

**When does MoE become worth it?** Around **100M+ parameters**, where activating
all weights every forward pass starts to matter. Build the dense model first.
Prove it works. Then add MoE when you hit scale constraints.

**Engineering detail:**

```
MoE overhead at 8M params:
  - Tile fingerprint computation: Conv1d or Linear pass per tile
  - Top-k routing: softmax over 23 experts per tile
  - Expert dispatch: scatter tiles to experts
  - Expert computation: only k experts run (but model is small enough
    that ALL experts running would be cheaper than the routing overhead)
  - Gather: reassemble tiles from expert outputs

Dense equivalent at 8M params:
  - Single forward pass through one large FFN
  - No routing, no scatter/gather, no load balancing
  - SIMPLER code, FEWER bugs, FASTER inference

Cost ratio at 8M:
  Routing overhead ≈ 15-20% of total compute
  Sparsity savings ≈ 60-70% (top-2 of 23 experts)
  Net: ~50-55% of dense compute... for a model that fits in cache anyway

Cost ratio at 100M:
  Routing overhead ≈ 2-5% of total compute
  Sparsity savings ≈ 60-70%
  Net: significant actual savings
```

### 4.2 MLA (Multi-head Latent Attention) with Skip Connections

```python
# mla_skip.py — multi-head latent attention
```

MLA compresses the KV cache for **long sequences**. Your sequences are image
tiles — small, fixed-size contexts. MLA is solving a context-length problem that
rendering doesn't have. Standard multi-head attention would be simpler and
equally effective for tile-size contexts.

**Engineering detail:**

```
MLA purpose (from DeepSeek-V2):
  - Compress KV cache from (num_heads × seq_len × head_dim)
  - To (kv_latent_dim × seq_len)
  - Savings scale with sequence length

Rendering context:
  - Tiles are 8×8 = 64 tokens (or 16×16 = 256 tokens)
  - Sequence length is FIXED and SMALL
  - KV cache compression saves ~nothing at seq_len=64

Standard MHA at seq_len=64:
  - Attention matrix: 64×64 = 4096 entries
  - Trivially fits in SRAM
  - No compression needed

MLA adds:
  - Compression/decompression projections (extra parameters)
  - Information loss in latent space
  - Complexity for no benefit
```

### 4.3 Three-Tier Config System

```python
# tier_config.py — Fast (~4M), Medium (~16M), Beast (~64M)
```

You designed three model sizes **before validating any of them works.** The tier
system adds configuration complexity, testing burden, and decision fatigue. You
should have ONE size — the smallest one — validated end-to-end, before even
thinking about scaling up.

### 4.4 The Blender Addon Exists but the Model Hasn't Been Trained

```python
# src/omen-blender/ — engine.py, bridge.py, panel.py, properties.py
```

You built a complete Blender integration — custom render engine, UI panels,
properties — for a model that hasn't been trained on real data yet. This is
beautiful engineering in the wrong order.

**HOWEVER** (user correction from MSG 56): The Blender addon IS the data
pipeline. 4 spp = training data, 256 spp = ground truth. The addon IS the
training loop. This is not cart-before-horse — this is a deliberate design
decision. The addon must stay.

### 4.5 SIGReg's Statistical Machinery

```
Epps-Pulley two-sample test: designed for testing whether two
samples come from the same distribution in academic statistics.

Using it as a differentiable loss component in an 8M parameter
neural network is like using a mass spectrometer to check if
your coffee has sugar.
```

A simple variance regularization term — `loss_reg = -log(std(latent) + eps)` —
would prevent collapse at this scale. SIGReg is for a paper. Simple
regularization is for a working system.

**Score: 5/5 too-much-too-little identified.**

---

## PART 5: The Scorecard

```
CORRECT ideas:         6/6   (JEPA, scene graph, online learning,
                                surprise, confidence, closed loop)

WRONG implementations: 4/4   (single optimizer, cosmetic replay,
                                overengineered SIGReg, surprise not wired)

CORRECT but WRONG WAY: 4/4   (routing source, LoRA vs dual memory,
                                ARPredictor premature, multi-mode)

TOO MUCH → TOO LITTLE: 5/5   (MoE at 8M, MLA for tiles, 3 tiers,
                                Blender before training, SIGReg complexity)
```

---

## PART 6: What Omen V1 Should Actually Be

Strip it down to what works. Keep everything — but implement in the right order.

```
Omen V1 — Minimum Viable Biological:

Architecture:
  - Scene graph encoder: simple GNN, not ViT+Mamba (scene graph is
    structured data — use a graph network, not a vision transformer)
  - Render feature encoder: 3-layer Conv2d (already correct)
  - Cross-attention fusion (already correct)
  - Decoder: Conv2dTranspose (already correct)
  - Confidence head (already correct)
  - NO MoE, NO MLA, NO ARPredictor
  - Target: ~3M parameters, dense, single-mode (denoiser)

Training:
  - Per-parameter-group learning rates
    (not per-expert — no experts in V1)
  - Surprise signal → modulate learning rate directly
  - Replay buffer: 500 items, stratified, 1:1 replay ratio
  - Simple variance regularization (not SIGReg)
  - Online continual learning (already correct)

Validation:
  - Train on 10 diverse scenes
  - Test on 5 held-out scenes
  - Measure: SSIM, PSNR, confidence calibration
  - Compare against: OptiX Denoiser, OIDN
  - If this doesn't beat them, scaling won't help

After V1 works:
  - Add MoE (now the model is big enough to benefit)
  - Add ARPredictor (now spatial prediction is validated)
  - Add tier system (now you know what to scale)
  - Add Blender integration polish (now there's something to integrate)
```

This would be ~3,000 lines of code instead of ~12,000. It would validate the
core thesis in weeks instead of months. And every piece of complexity you
stripped out gets added back WITH justification — because the simpler version
proved it was necessary.

---

## PART 7: The Biological Lesson for Your Process

You know this from mol bio. When you're cloning a gene:

```
Wrong approach:
  1. Design 10 primer pairs
  2. Build 5 expression vectors
  3. Order 3 fluorescent tags
  4. Plan the entire Western blot pipeline
  5. Then run the PCR for the first time
  6. PCR fails → everything else was wasted

Right approach:
  1. Run the PCR with one primer pair
  2. See if you get a band
  3. Then design the vector
  4. Then clone
  5. Then express
  6. Each step validated before the next
```

Omen right now is approach #1. You built the entire experimental pipeline
before running the PCR. The MoE, the tiers, the Blender addon, the ARPredictor,
the SIGReg — all of these are downstream of a core hypothesis (scene-aware JEPA
denoising works) that hasn't been tested yet.

**Strip to the core. Validate. Then add complexity with justification.**

That's what the brain does too — it doesn't grow the prefrontal cortex before
the brainstem works. Evolution validates each layer before building the next.
Your architecture should do the same.

---

## Appendix: User's Position (from conversation)

> "I don't want to cut things down. I want to keep them, but implement them
> properly. I cannot cut down the Blender add-on part because all the training
> data is actually Blender scenes. Whenever I open Blender and try to render
> with this add-on, the data flows. That is my data pipeline. So I cannot cut
> the Blender add-on because whenever I use the add-on, it actually trains the
> model. Because at 4 samples per pixel, it is my training data. And at 256
> samples per pixel, it is my ground truth."

The critique acknowledges this. The Blender addon IS the data pipeline. The
critique is about **implementation order**, not about cutting features. Keep
everything. Fix the implementation.
