# Design: fix-training-wiring

## Architecture Overview

```
                        OmenConfig
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   OmenJEPA             OmenTrainer          Mode Pipeline
  (forward gated)     (optimizer gated)     (mode gated)
```

Every configurable component follows the same pattern:
1. Component always exists (parameters initialized)
2. Config switch controls forward behavior (identity passthrough when OFF)
3. Config switch controls gradient flow (no grad for OFF components)
4. Config switch controls optimizer (separate optimizer per component group)

---

## 1. OmenConfig (`src/omen/config.py`)

```python
@dataclass
class ComponentSwitches:
    """Toggle individual model components ON/OFF."""
    # --- Always-on core (V1) ---
    scene_encoder: bool = True
    render_encoder: bool = True
    cross_attention: bool = True
    decoder: bool = True
    confidence_head: bool = True

    # --- MoE system ---
    moe: bool = False                    # Master MoE switch
    moe_materials: bool = True           # Sub-switches (only if moe=True)
    moe_lights: bool = True
    moe_geometry: bool = True
    moe_motion: bool = True
    scene_graph_routing: bool = False    # True = scene graph, False = pixel fp

    # --- Temporal ---
    ar_predictor: bool = False           # OFF for V1, ON for animation
    scene_delta_encoder: bool = False    # Needs ARPredictor

    # --- Regularization ---
    sigreg: bool = False                 # Full SIGReg (for paper)
    simple_var_reg: bool = True          # -log(std + eps) for working system

    # --- Adaptation ---
    episodic_correction: bool = True     # Separate correction MLP
    lora: bool = False                   # Legacy LoRA (replaced by episodic)

    # --- Other ---
    mla_skip: bool = False               # Standard MHA when OFF

@dataclass
class TrainingSwitches:
    """Training-specific switches."""
    per_component_lr: bool = True        # Separate optimizers per group
    surprise_lr_modulation: bool = True   # Wire z_score to lr
    stratified_replay: bool = True
    replay_size: int = 500
    replay_ratio: float = 0.5            # 50% new, 50% replay
    surprise_lr_scale: float = 2.0

@dataclass
class ModeSwitches:
    """Mode pipeline switches."""
    denoiser: bool = True
    adaptive: bool = False
    multires: bool = False
    temporal: bool = False

@dataclass
class OmenConfig:
    components: ComponentSwitches = field(default_factory=ComponentSwitches)
    training: TrainingSwitches = field(default_factory=TrainingSwitches)
    modes: ModeSwitches = field(default_factory=ModeSwitches)
    tier: str = "fast"                   # fast / medium / beast

    @staticmethod
    def v1_dense() -> "OmenConfig":
        """V1: Dense denoiser, no MoE, no AR, no SIGReg."""
        return OmenConfig()

    @staticmethod
    def v1_moe() -> "OmenConfig":
        """After V1 validated: unlock MoE with scene-graph routing."""
        cfg = OmenConfig()
        cfg.components.moe = True
        cfg.components.scene_graph_routing = True
        return cfg

    @staticmethod
    def v1_animation() -> "OmenConfig":
        """After MoE validated: unlock temporal prediction."""
        cfg = OmenConfig.v1_moe()
        cfg.components.ar_predictor = True
        cfg.components.scene_delta_encoder = True
        cfg.modes.temporal = True
        return cfg

    @staticmethod
    def full() -> "OmenConfig":
        """Everything enabled."""
        cfg = OmenConfig()
        cfg.components.moe = True
        cfg.components.scene_graph_routing = True
        cfg.components.ar_predictor = True
        cfg.components.scene_delta_encoder = True
        cfg.components.sigreg = True
        cfg.components.simple_var_reg = False
        cfg.modes.adaptive = True
        cfg.modes.multires = True
        cfg.modes.temporal = True
        return cfg
```

---

## 2. Component Switch Behavior

### 2.1 MoE Switch (`moe.py`)

```
MoE OFF:  x → SharedExpert → output          (dense FFN, ~350K params)
MoE ON:   x → fingerprint/routing → top-k    (full 23-expert routing)
           experts → combine → shared → output
```

When `config.moe = False`, `TileMoERouter.forward()` returns
`self.shared(x)` directly. No routing computation, no expert dispatch.

When `config.moe = True` but sub-switches are OFF (e.g. `moe_materials = False`),
the corresponding `ExpertGroup.forward()` returns zero contribution.
Only enabled expert groups are activated.

### 2.2 ARPredictor Switch (`arpredictor.py`)

```
AR OFF:  current_latent → predicted_latent (identity, no history needed)
AR ON:   history + current_latent → ConditionalBlocks → predicted_latent
```

When `config.ar_predictor = False`:
- `ARPredictor.forward()` returns `current_latent` unchanged
- `SceneDeltaEncoder` is not called (no delta computed)
- History buffer is not maintained

When `config.ar_predictor = True`:
- Full temporal prediction with ConditionalBlocks + AdaLN-zero
- SceneDeltaEncoder encodes frame-to-frame deltas
- History window maintained in bridge/animation mode

The ARPredictor switch is tied to animation mode:
- `mode_denoiser` only → AR OFF (single frame)
- `mode_temporal` → AR ON (multi-frame prediction)
- But can be forced ON/OFF independently via config

### 2.3 SIGReg / Simple Var Reg Switch (`sigreg.py`)

```
simple_var_reg ON:  loss_reg = -log(std(latent, axis=0) + eps)
sigreg ON:          loss_reg = SIGReg.forward(embeddings)
both OFF:           loss_reg = 0
```

When `simple_var_reg = True` and `sigreg = False` (V1 default):
```python
def simple_variance_regularization(latent, eps=1e-6):
    std = latent.std(axis=0)
    return -nb.mean(nb.log(std + eps))
```

### 2.4 EpisodicCorrection Switch (`episodic.py`)

```
episodic ON:   output = main_output + episodic_net(main_output, scene_context)
episodic OFF:  output = main_output (correction = 0)
```

```python
class EpisodicCorrection(nn.Module):
    """Separate fast-adaptation network (~100K params).

    Own optimizer, own (higher) learning rate.
    Architecturally independent from main model.
    """
    def __init__(self, dim=192, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, main_output, scene_context, enabled=True):
        if not enabled:
            return main_output  # identity passthrough
        correction = self.net(nb.concat([main_output, scene_context], axis=-1))
        return main_output + correction
```

Separate optimizer for episodic params:
```python
episodic_opt = nn.optim.AdamW(
    episodic_net.parameters(),
    lr=2e-2,  # 400x higher than base (matches Nabla LoRA example pattern)
)
```

---

## 3. Routing: Scene Graph vs Pixel Fingerprint

### Current (pixel fingerprint):
```
aux buffer (H,W,10) → _compute_fingerprints() → 23-dim vector per tile
                                                     │
                                                     ▼
                                              TileMoERouter gates
```

### Fixed (scene-graph routing):
```
scene_graph → material_ids (per-face), light_type_ids
                    │
                    ▼
         project to tile grid (8×8 tiles)
         → per-tile: dominant material_id + dominant light_type
                    │
                    ▼
         embedding lookup → routing logits
                    │
                    ▼
         TileMoERouter.forward(x, routing_logits)
```

The scene graph already contains:
- `geometry.material_ids` — per-face uint32 (0=diffuse, 1=rough, 2=glossy, 3=glass, 4=metal)
- `lights.params` — (L,7) with type_id in column 0 (0=point, 1=area, 2=environment)

New routing approach:
1. Rasterize material_ids and light_type_ids into per-pixel maps (via Mitsuba AOV)
2. Downsample to 8×8 tile grid → per-tile dominant material + light type
3. Embed material_id + light_type → routing logits via nn.Embedding + nn.Linear
4. Feed to existing ExpertGroup routing

When `config.scene_graph_routing = False`, falls back to current pixel fingerprint.

---

## 4. Multi-Optimizer Training

### Nabla Constraint
Nabla's `AdamW(model, lr=...)` takes a SINGLE learning rate — no parameter
groups like PyTorch. Solution: **multiple AdamW instances, one per component group**.

This is the canonical Nabla pattern — the LoRA example creates a separate
optimizer for adapter params at lr=2e-2 vs base lr=5e-5.

### Optimizer Layout

```
Component Group          Optimizer          Base LR
─────────────────────────────────────────────────────
scene_encoder            encoder_opt        5e-5
render_encoder           encoder_opt        5e-5  (shared)
cross_attention          encoder_opt        5e-5  (shared)
decoder                  decoder_opt        5e-5
confidence_head          encoder_opt        5e-5  (shared)
shared_expert            shared_expert_opt  5e-5
material_experts         material_opt       5e-5  (if moe ON)
light_experts            light_opt          3e-5  (if moe ON)
geometry_experts         geo_opt            4e-5  (if moe ON)
motion_experts           motion_opt         5e-5  (if moe ON)
ar_predictor             ar_opt             5e-5  (if ar ON)
scene_delta_encoder      ar_opt             5e-5  (shared with AR)
episodic_correction      episodic_opt       2e-2  (always separate)
```

### Surprise → LR Modulation

```python
for name, (component, optimizer) in active_optimizers.items():
    base_lr = base_lrs[name]
    if config.training.surprise_lr_modulation and z_score > 0:
        lr = base_lr * (1.0 + config.training.surprise_lr_scale * min(z_score, 5.0))
    else:
        lr = base_lr
    # Update optimizer's lr (Nabla: recreate or adjust)
    optimizer.lr = lr
```

### Training Step with Switches

```python
def train_step(self, noisy, gt, scene_graph, z_score=0.0):
    self.model.train()

    # Forward pass (respects component switches)
    predicted_latent = self.model.encode(scene_graph, noisy)
    target_latent = self.model.encode(scene_graph, gt)

    # Loss (respects regularization switches)
    total_loss, pred_loss, reg_loss = self.model.compute_loss(
        predicted_latent, target_latent, self.config
    )

    # Backward
    total_loss.backward()
    _clip_grad_norm(self.model.parameters(), DEFAULT_GRADIENT_CLIP)

    # Per-component optimizer steps
    for name, optimizer in self._active_optimizers():
        lr = self._compute_lr(name, z_score)
        optimizer.lr = lr
        self.model = optimizer.step()  # Nabla: returns updated model

    self.model.zero_grad()
    self.iteration += 1
```

---

## 5. Stratified Replay Buffer (`replay.py`)

Replaces `lora_manager.py`'s flat `deque(maxlen=50)`.

```python
class StratifiedReplayBuffer:
    """Stratified replay buffer for continual learning.

    Maintains per-scene sub-buffers. Sampling ensures diversity
    across scenes (sleep-like replay).
    """
    def __init__(self, max_size=500, replay_ratio=0.5):
        self.max_size = max_size
        self.replay_ratio = replay_ratio
        self._buffers: dict[str, deque] = {}  # scene_hash → deque

    def add(self, scene_hash, noisy, gt):
        if scene_hash not in self._buffers:
            self._buffers[scene_hash] = deque(maxlen=self.max_per_scene())
        self._buffers[scene_hash].append((noisy, gt))
        self._trim()

    def sample(self, current_scene, count):
        """Sample from OTHER scenes for replay interleaving."""
        other_scenes = [h for h in self._buffers if h != current_scene]
        if not other_scenes:
            return []
        # Stratified: pick random scene, then random item from that scene
        samples = []
        for _ in range(count):
            scene = random.choice(other_scenes)
            buf = self._buffers[scene]
            if buf:
                samples.append(random.choice(list(buf)))
        return samples

    def replay_ratio_count(self, new_count):
        """How many replay samples for N new samples (1:1 ratio)."""
        return int(new_count * self.replay_ratio / (1 - self.replay_ratio))
```

---

## 6. Forward Pass with All Switches

```python
class OmenJEPA(nn.Module):
    def __init__(self, config: OmenConfig = None):
        super().__init__()
        self.config = config or OmenConfig()
        c = self.config.components

        # --- Core (always created) ---
        self.scene_encoder = SceneGraphEncoder()
        self.render_encoder = RenderFeatureEncoder()
        self.cross_attn = CrossAttentionFusion()
        self.decoder = Decoder()
        self.confidence_head = ConfidenceHead()

        # --- MoE (always created, gated by config) ---
        self.moe = TileMoERouter(192)
        self.dense_ffn = SharedExpert(192)  # Fallback when MoE OFF

        # --- ARPredictor (always created, gated by config) ---
        self.ar_predictor = ARPredictor()
        self.delta_encoder = SceneDeltaEncoder()

        # --- Regularization (always created, gated by config) ---
        self.sigreg = SIGRegLoss()
        self.var_reg = lambda x: -nb.mean(nb.log(x.std(axis=0) + 1e-6))

        # --- Episodic correction (always created, gated by config) ---
        self.episodic = EpisodicCorrection()

        # --- Routing lookup (for scene-graph routing) ---
        if c.scene_graph_routing:
            self.mat_embed = nn.Embedding(6, 64)   # 6 material types
            self.light_embed = nn.Embedding(4, 64)  # 4 light types
            self.route_proj = nn.Linear(128, 23)     # → routing logits

    def encode(self, scene_graph, noisy_render):
        scene_latent = self.scene_encoder.forward(scene_graph)
        render_latent = self.render_encoder.forward(noisy_render)
        fused = self.cross_attn.forward(render_latent, scene_latent)
        return fused, scene_latent

    def forward(self, scene_graph, noisy_render,
                history=None, delta=None, aux=None):
        c = self.config.components
        latent, scene_ctx = self.encode(scene_graph, noisy_render)

        # --- MoE / Dense FFN ---
        if c.moe:
            if c.scene_graph_routing:
                routing = self._route_from_scene_graph(scene_graph, aux)
            else:
                routing = self._route_from_fingerprints(aux)
            features = self.moe.forward(latent.unsqueeze(1), routing)
            latent = features.squeeze(1)
        else:
            latent = self.dense_ffn.forward(latent)

        # --- ARPredictor ---
        if c.ar_predictor and history is not None:
            if c.scene_delta_encoder and delta is not None:
                delta_emb = self.delta_encoder.forward(delta)
            else:
                delta_emb = nb.zeros((1, 192))
            latent = self.ar_predictor.forward(history, latent, delta_emb)
        # else: identity passthrough (latent unchanged)

        # --- Decode ---
        h, w = noisy_render.shape[1], noisy_render.shape[2]
        decoded = self.decoder.forward(latent, h, w)

        # --- Confidence ---
        if c.confidence_head:
            confidence = self.confidence_head.forward(latent)
        else:
            confidence = None

        # --- Episodic correction ---
        if c.episodic_correction:
            decoded = self.episodic.forward(decoded, scene_ctx, enabled=True)
        # else: decoded passes through unchanged

        return decoded, confidence, latent

    def compute_loss(self, predicted, target, config=None):
        c = (config or self.config).components
        pred_loss = F.mse_loss(predicted, target)

        if c.sigreg:
            reg_loss = self.sigreg.forward(predicted)
        elif c.simple_var_reg:
            reg_loss = -nb.mean(nb.log(predicted.std(axis=0) + 1e-6))
        else:
            reg_loss = nb.constant(0.0)

        return pred_loss + 0.09 * reg_loss, pred_loss, reg_loss
```

---

## 7. ARPredictor Switch for Animation

The ARPredictor is critical for animation mode but unnecessary for single-frame
denoising. The switch design:

```
┌────────────────────────────────────────────────────────────────┐
│                     ARPredictor Switch Matrix                  │
├──────────────┬─────────────┬──────────────────────────────────┤
│ Mode         │ AR Switch   │ Behavior                         │
├──────────────┼─────────────┼──────────────────────────────────┤
│ denoiser     │ OFF (V1)    │ Identity passthrough             │
│              │             │ No history maintained            │
│              │             │ No delta computed                │
├──────────────┼─────────────┼──────────────────────────────────┤
│ denoiser+    │ ON          │ Temporal-aware denoising         │
│ temporal     │             │ Uses history for coherence       │
│              │             │ Predicts next latent for smooth  │
├──────────────┼─────────────┼──────────────────────────────────┤
│ animation    │ ON          │ Full temporal prediction         │
│              │             │ ARPredictor drives frame predict │
│              │             │ SceneDeltaEncoder active         │
│              │             │ Jump-cut detection active        │
└──────────────┴─────────────┴──────────────────────────────────┘
```

When AR is OFF:
- `ARPredictor.forward(history, current, delta_emb)` returns `current` unchanged
- History buffer in bridge is not populated
- Delta encoder is not called
- ~2.2M params exist but contribute zero to forward/backward

When AR is ON:
- Full ConditionalBlock transformer processes history + current
- SceneDeltaEncoder encodes frame deltas
- Prediction used for temporal coherence
- Surprise detection drives re-anchoring AND learning rate modulation

---

## 8. File Change Summary

| File | Change | LOC Impact |
|------|--------|------------|
| `config.py` | NEW: OmenConfig dataclass + presets | ~80 lines |
| `jepa.py` | MODIFY: config-gated forward, multi-optimizer | ~40 lines changed |
| `moe.py` | MODIFY: scene-graph routing path + config switch | ~50 lines added |
| `arpredictor.py` | MODIFY: passthrough when disabled | ~10 lines |
| `sigreg.py` | MODIFY: simple_var_reg as alternative | ~15 lines |
| `episodic.py` | NEW: EpisodicCorrection network | ~40 lines |
| `trainer.py` | MODIFY: multi-optimizer, surprise lr, config | ~60 lines changed |
| `replay.py` | NEW: StratifiedReplayBuffer | ~80 lines |
| `lora_manager.py` | MODIFY: keep as compat shim to replay.py | ~10 lines |
| `temporal.py` | MODIFY: expose z_score, add lr modulation helper | ~15 lines |
| `jepa_bridge.py` | MODIFY: config propagation, ARPredictor history | ~30 lines |
| `denoiser.py` | MODIFY: config-gated render pipeline | ~20 lines |

Total: ~450 lines new/modified across 12 files
