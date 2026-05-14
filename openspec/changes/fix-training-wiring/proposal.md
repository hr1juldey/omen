# Proposal: fix-training-wiring

## Problem

Omen's model has all the right architectural components (JEPA, MoE, ARPredictor,
SIGReg, surprise detection, LoRA), but the training wiring has 8 specific bugs
that prevent meaningful training on real Blender scenes:

**Part 2 — Wrong Implementations (4 bugs):**
1. Single optimizer for all 23 MoE experts (same lr for converged + learning experts)
2. Cosmetic replay buffer (50 items, no stratification, no 1:1 ratio)
3. Overengineered SIGReg (17 knots, 1024 projections for an 8M param model)
4. Surprise signal computed but never wired to optimizer/learning rate

**Part 3 — Correct Ideas, Wrong Execution (4 bugs):**
5. MoE routes from pixel fingerprints instead of scene graph ground truth
6. LoRA adapters instead of architecturally separate episodic correction network
7. ARPredictor active before spatial denoising is validated
8. Four modes built simultaneously before any single mode is validated

## Solution

Build the **full product with component switches**. Every component exists in the
model always, but each can be enabled/disabled independently. This lets us:

- Start with minimal V1 (dense denoiser, ~3M params active)
- Unlock components one at a time as each is validated
- Keep all code stable, all weights stable across switches
- Control switches from API now, Blender addon UI later

### OmenConfig — The Control Panel

A serializable dataclass that controls which components are active in forward
and training passes. Disabled components use identity passthrough (parameters
exist but are unused). Parameters are never removed — only gradients and
forward contributions are gated.

### Key Architectural Changes

1. **Multiple optimizers** (Nabla pattern from LoRA example) — one per expert
   group with independent learning rates
2. **EpisodicCorrection network** — separate 2-layer MLP (~100K params) with
   own optimizer, replaces LoRA as the fast-adaptation mechanism
3. **Scene-graph routing** — uses material_ids + light_type_ids from
   scene_extractor.py instead of pixel-derived fingerprints
4. **Simple variance regularization** — replaces SIGReg with
   `-log(std(latent) + eps)` (3 lines vs 400)
5. **Surprise → lr modulation** — wires z_score to per-component learning rate
6. **Stratified replay buffer** — 500 items, per-scene stratification, 1:1 ratio
7. **ARPredictor switch** — OFF for V1 denoiser, ON when animation mode is
   unlocked; passthrough returns current latent unchanged when disabled
8. **Mode switches** — denoiser ON by default; adaptive/multires/temporal OFF
   until denoiser is validated

## Scope

Files modified:
- NEW: `src/omen/config.py` — OmenConfig dataclass + preset configs
- MODIFY: `src/omen/model/jepa.py` — config-gated forward + compute_loss
- MODIFY: `src/omen/model/moe.py` — scene-graph routing + config switch
- MODIFY: `src/omen/model/arpredictor.py` — config switch (passthrough when off)
- MODIFY: `src/omen/model/sigreg.py` — simple_var_reg switch
- NEW: `src/omen/model/episodic.py` — EpisodicCorrection network
- MODIFY: `src/omen/training/trainer.py` — multi-optimizer + surprise lr
- MODIFY: `src/omen/modes/lora_manager.py` → rewrite to `replay.py` — stratified buffer
- MODIFY: `src/omen/temporal.py` — expose z_score for lr modulation
- MODIFY: `src/omen/jepa_bridge.py` — config propagation + ARPredictor history
- MODIFY: `src/omen/modes/denoiser.py` — config-gated pipeline

## Out of Scope

- Part 4 critiques (MoE at 8M, MLA, tier system) — acknowledged but controversial
- Blender addon UI for switches (later change)
- New training data or validation benchmarks
- Removing any existing code (everything stays, just gated)

## Success Criteria

- All 8 wiring bugs fixed
- V1 config produces a working dense denoiser (~3M active params)
- Can train on real Blender scenes (4spp noisy → 256spp clean)
- Switching any component ON/OFF doesn't break existing weights
- All existing tests pass
- Checkpoint format unchanged (full model always serialized)
