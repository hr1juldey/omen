# Spec: Training Wiring Fixes

## Component Switch Specification

### SW-001: OmenConfig Dataclass

**Requirement**: A serializable configuration dataclass that controls every
component's activation state.

**Behavior**:
- Config can be serialized/deserialized (saved with checkpoints)
- Changing a switch at runtime does not require model re-initialization
- All parameters exist regardless of switch state
- Disabled components contribute zero to forward pass and receive zero gradients

**Preset Configs**:
- `v1_dense()` — dense denoiser only (MoE OFF, AR OFF, SIGReg OFF)
- `v1_moe()` — MoE with scene-graph routing unlocked
- `v1_animation()` — ARPredictor + temporal mode unlocked
- `full()` — everything enabled

---

### SW-002: MoE Component Switch

**Requirement**: MoE can be toggled ON/OFF. When OFF, all tokens pass through
the shared expert (dense FFN). When ON, full 23-expert routing activates.

**Sub-switches**: material_experts, light_experts, geometry_experts, motion_experts
Each sub-switch controls whether its ExpertGroup contributes. Disabled groups
return zero contribution.

**Routing Switch**: `scene_graph_routing` controls whether routing uses
scene graph ground truth (material_id + light_type) or pixel fingerprints.

---

### SW-003: ARPredictor Component Switch

**Requirement**: ARPredictor can be toggled ON/OFF. When OFF, returns current
latent unchanged (identity passthrough). When ON, processes history window
through ConditionalBlocks with AdaLN-zero conditioning.

**Dependencies**: When ON, `scene_delta_encoder` should also be ON for delta
conditioning. When OFF, history buffer is not maintained.

**Animation Integration**: ARPredictor switch is independent from mode switches
but typically ON when `mode_temporal = True`.

---

### SW-004: Regularization Switch

**Requirement**: Two mutually exclusive regularization options:
- `simple_var_reg` (V1 default): `-log(std(latent, axis=0) + eps)`
- `sigreg`: Full Epps-Pulley SIGReg loss

Both can be OFF (no regularization loss). Only one should be ON at a time.

---

### SW-005: Episodic Correction Switch

**Requirement**: EpisodicCorrection network can be toggled ON/OFF. When OFF,
main model output passes through unchanged. When ON, adds a learned correction
from a separate small network with its own optimizer at 400x higher lr.

**Backward Compatibility**: `lora` switch exists for legacy LoRA path but is
OFF by default. Episodic correction replaces LoRA as the adaptation mechanism.

---

## Training Wiring Specifications

### TW-001: Per-Component Optimizers

**Requirement**: Each component group has its own AdamW optimizer with
independent learning rate.

**Component Groups**:
1. Encoder group (scene + render + cross_attn + confidence): shared optimizer, lr=5e-5
2. Decoder: own optimizer, lr=5e-5
3. Shared expert (dense FFN): own optimizer, lr=5e-5
4. Material experts: own optimizer, lr=5e-5 (only active when MoE ON)
5. Light experts: own optimizer, lr=3e-5 (only active when MoE ON)
6. Geometry experts: own optimizer, lr=4e-5 (only active when MoE ON)
7. Motion experts: own optimizer, lr=5e-5 (only active when MoE ON)
8. ARPredictor + delta encoder: shared optimizer, lr=5e-5 (only active when AR ON)
9. Episodic correction: own optimizer, lr=2e-2 (always active when enabled)

**Nabla Pattern**: Each optimizer is `nn.optim.AdamW(component.parameters(), lr=...)`.
Step returns updated model: `model = optimizer.step()`.

---

### TW-002: Surprise → Learning Rate Modulation

**Requirement**: z_score from `temporal.detect_surprise()` modulates the learning
rate of all active optimizers.

**Formula**: `lr = base_lr * (1.0 + surprise_lr_scale * min(z_score, 5.0))`

**Gate**: Only active when `config.training.surprise_lr_modulation = True`.

**Signal Source**: `temporal.detect_surprise()` already returns z_score. The
trainer receives it as a parameter to `train_step()`.

---

### TW-003: Stratified Replay Buffer

**Requirement**: Replace flat deque(maxlen=50) with stratified buffer.

**Properties**:
- Total capacity: 500 items (configurable)
- Per-scene sub-buffers with automatic trimming
- Sampling: stratified by scene (pick random scene, then random item)
- Replay ratio: 1:1 (50% new data, 50% replay per training step)
- Only replays from OTHER scenes (prevents overfitting to current scene)

---

## Routing Specifications

### RT-001: Scene-Graph Routing

**Requirement**: When `scene_graph_routing = True`, route experts using structured
scene data instead of pixel-derived fingerprints.

**Input**: `scene_graph.geometry.material_ids` (per-face) + `scene_graph.lights.params[:, 0]` (light type IDs)

**Pipeline**:
1. Material IDs rasterized to per-pixel map via Mitsuba AOV
2. Downsampled to 8×8 tile grid (dominant material + light per tile)
3. Material ID → nn.Embedding(6, 64), Light type → nn.Embedding(4, 64)
4. Concatenate + Linear(128, 23) → routing logits
5. Feed to existing ExpertGroup routing

**Fallback**: When `scene_graph_routing = False`, uses current pixel fingerprint path.
