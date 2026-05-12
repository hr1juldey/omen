## Context

Omen Mitsuba integrator skeleton exists (previous change). Current state:
- `OmenIntegrator` registered with Mitsuba plugin system
- Standard path tracing working via `mi.render()`
- Placeholder parameters: `jepa_model`, `use_gpu`

**What's missing:** Actual JEPA implementation in Mojo with scene conditioning.

**Critical constraint discovered:** Mitsuba's path tracer is C++ code (`src/integrators/path.cpp`). Python `mi.render()` is a binding - cannot inject into sampling loop. JEPA must work via multi-pass rendering: render → extract → JEPA → render → merge.

**Why JEPA is different from 2D denoisers:**
- OptiX/OIDN: only see 2D pixels + normals
- Omen JEPA: sees exact 3D scene (geometry, materials, lights) from Mitsuba
- Self-training advantage: render YOUR scene at 4spp + 256spp → perfect pairs, unlimited data

**Test scene:** Cornell box (`mi.cornell_box()`) renders in 2s at 256×256. Target: same quality in <500ms with 4-8× fewer samples via adaptive mode.

## Goals / Non-Goals

**Goals:**
- Implement scene graph extraction from Mitsuba Python API
- Create Mojo JEPA kernels with C ABI interface
- Implement 3 rendering modes (denoiser, adaptive, multires)
- Self-training protocol using Cornell box
- Zero-copy GPU buffer passing between Mitsuba and Mojo

**Non-Goals:**
- Modifying Mitsuba C++ path tracer source
- Per-pixel adaptive sampling within single render (C++ limitation)
- Material node compilation (use Mitsuba's BSDFs)

## Decisions

### Decision 1: JEPA architecture in Mojo/Nabla (LeWM port)

**Choice:** Direct Mojo/Nabla port of LeWorldModel (Maes et al. 2026, LeCun) with scene-delta conditioning replacing robot actions

**Rationale:**
- LeWM proves JEPA works as world model with only 2 losses (prediction + SIGReg)
- ~18M parameters, trains in hours on single GPU
- No EMA, no pretrained encoders, no auxiliary losses (unlike I-JEPA/V-JEPA)
- AdaLN-zero conditioning lets scene deltas modulate transformer attention/MLP
- Nabla provides autograd, GPU kernels, and SPMD for all tensor ops in pure Mojo

**Exact architecture from LeWM source (ported to Mojo/Nabla):**

```
Component              Params    Mojo File                    Nabla Ops
─────────────────────────────────────────────────────────────────────────
ViT-Tiny encoder       5.5M      scene_encoder.mojo           Linear, LayerNorm, GELU
  hidden=192, heads=3, depth=12, patch=14, img=224
Projector MLP          789K      scene_encoder.mojo           Linear, BatchNorm1d, GELU
  192 → 2048 → 192 (with BatchNorm)
SceneDeltaEncoder      155K      scene_delta_encoder.mojo     Conv1d, Linear, SiLU
  (replaces LeWM's action_encoder)
  Conv1d(input, smoothed, k=1) + MLP(smoothed → 4*emb → emb)
ARPredictor            10.8M     arpredictor.mojo             Linear, SiLU, scaled_dot_product
  Transformer(depth=6, heads=16, dim_head=64, mlp_dim=2048)
  Uses ConditionalBlock with AdaLN-zero conditioning
pred_proj MLP          789K      arpredictor.mojo             Linear, BatchNorm1d, GELU
  192 → 2048 → 192 (with BatchNorm)
SIGReg                 0         sigreg.mojo                  randn, matmul, cos, sin
  knots=17, num_proj=1024, Epps-Pulley statistic (buffers only)
Decoder                ~2M       decoder.mojo                 Linear, ConvTranspose2d
  Latent → RGB (added for Omen, LeWM operates in latent space only)
─────────────────────────────────────────────────────────────────────────
TOTAL                  ~20M
```

**Critical component: ConditionalBlock (AdaLN-zero)**
```mojo
# Ported from LeWM module.py ConditionalBlock
struct ConditionalBlock:
    var attn: Attention           # dim=192, heads=16, dim_head=64
    var mlp: FeedForward          # dim=192, hidden=2048
    var norm1: LayerNorm          # elementwise_affine=False
    var norm2: LayerNorm          # elementwise_affine=False
    var adaLN: Sequential         # SiLU + Linear(192, 1152)

    def forward(mut self, x: Tensor, c: Tensor) -> Tensor:
        # c = scene delta embedding (conditioning signal)
        # AdaLN produces 6 modulation params: shift/scale/gate × 2
        mods = self.adaLN(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa = mods[0], mods[1], mods[2]
        shift_mlp, scale_mlp, gate_mlp = mods[3], mods[4], mods[5]

        # Modulate norm: x * (1 + scale) + shift
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```

**Critical component: SIGReg loss (prevents representation collapse)**
```mojo
# Ported from LeWM module.py SIGReg
struct SIGReg:
    var t: Tensor          # linspace(0, 3, 17) - evaluation points
    var phi: Tensor        # exp(-t²/2) - Gaussian characteristic function
    var weights: Tensor    # trapezoidal weights * phi
    var num_proj: Int = 1024

    def forward(mut self, proj: Tensor) -> Tensor:
        # proj: (T, B, D) - the embeddings to regularize
        # Sample random projections A ~ N(0,1), normalize to unit length
        A = randn(proj.shape[-1], self.num_proj)
        A = A / A.norm(p=2, dim=0)

        # Project embeddings onto random directions, evaluate at Gaussian points
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (T, B, num_proj, knots)

        # Measure deviation from Gaussian characteristic function
        err = (x_t.cos().mean(dim=-3) - self.phi).square() + x_t.sin().mean(dim=-3).square()
        statistic = (err @ self.weights) * proj.shape[-2]
        return statistic.mean()
```

**Training loss (from LeWM train.py lejepa_forward):**
```
pred_loss = MSE(predict(emb[:, :ctx_len], delta_emb[:, :ctx_len]), emb[:, n_preds:])
sigreg_loss = SIGReg(emb.transpose(0, 1))
total_loss = pred_loss + λ * sigreg_loss   # λ=0.09 (from lewm.yaml), configurable
```

**Why this over 2D U-Net denoiser:**
- Scene conditioning via AdaLN-zero: scene delta modulates every attention/MLP layer
- Only 2 loss terms (not 6+ like other JEPAs) - minimal hyperparameter tuning
- SIGReg prevents collapse without EMA/target network (simpler implementation in Nabla)
- History truncation `emb[:, -HS:]` enables constant-memory prediction regardless of sequence length

### Decision 2: Multi-pass rendering strategy

**Choice:** Full-image re-renders at different spp, not tile-based

**Rationale:**
- Mitsuba Python API doesn't support crop rendering
- `mi.render()` always renders full frame
- Merge happens after both renders complete (not per-tile)

**Mode 2 adaptive flow:**
```python
# PASS 1: Preview (4 spp)
preview = mi.render(scene, spp=4)
confidence = jepa.predict_confidence(scene_graph, preview)

# PASS 2: High-spp (128 spp)
high_spp = mi.render(scene, spp=128)

# Merge: per-pixel based on confidence
output = jepa.merge_adaptive(preview, high_spp, confidence)
```

**Future C++ integrator could do tile-based, but Python API limits us to full-image.**

### Decision 3: Scene graph representation

**Choice:** Flat arrays with metadata, not recursive graphs

**Rationale:**
- Simpler C ABI (no pointers to pointers)
- Mojo can load directly via UnsafePointer
- Variable-length scenes handled with size fields

**C struct layout:**
```c
typedef struct {
    float* vertices;    // [N_verts×3]
    int    n_verts;
    int* faces;        // [N_faces×3]
    int    n_faces;
    int*   material_ids; // [N_faces]
} Geometry;

typedef struct {
    float* diffuse;
    float  roughness;
    float  metallic;
    // ... 8 params total
} Material;
```

### Decision 4: Self-training on Cornell box

**Choice:** Render same scene at multiple spp levels for training pairs

**Protocol:**
1. Render Cornell box at 4 spp → noisy input
2. Render Cornell box at 256 spp → ground truth
3. Train: (noisy + scene_graph) → ground truth
4. Repeat for 1000 frames (different camera angles, light positions)

**Why this works:**
- Same scene = consistent 3D structure
- JEPA learns "Cornell box with red wall at position X looks like Y"
- Transferable to similar scenes (boxes, indoor rooms)

### Decision 5: Mojo C ABI interface + GPU kernel interface

**Choice:** C ABI for Python bridge (`@register_function`), Mojo GPU kernels internally using `vectorize` and `TileTensor` — NO raw for/if loops in kernels

**Rationale:**
- C ABI is stable across Mojo versions and compatible with ctypes
- GPU kernels use `TileTensor` with layout declarations (not raw pointers)
- Zero-copy: `DeviceBuffer(ctx, raw_ptr, count, owning=False)` wraps Mitsuba's GPU memory
- `vectorize[simd_width]` replaces dirty for-loops — processes SIMD-width elements per invocation
- `comptime for` for compile-time unrolled loops (attention heads, tile dimensions)
- SIMD masking (`select`, `mask`) replaces if-guard bounds checks

**C ABI entry point (called from Python via ctypes):**
```mojo
@register_function
def omen_denoise(
    scene: SceneGraph,
    obs: RenderObservation,
    output_rgba: UnsafePointer[C_float],
    gpu_device_id: Int,
) -> Int:
    var ctx = DeviceContext()
    # Zero-copy wrap of Mitsuba's GPU buffer
    var input_buf = DeviceBuffer[DType.float32](ctx, obs.rgba_ptr, obs.size, owning=False)
    var output_buf = ctx.enqueue_create_buffer[DType.float32](obs.size)

    comptime layout = row_major[obs.height, obs.width, 4]()
    var input_tensor = TileTensor(input_buf, layout)
    var output_tensor = TileTensor(output_buf, layout)

    # Launch GPU kernel (bind comptime params first)
    comptime kernel = denoise_kernel[type_of(layout)]
    ctx.enqueue_function[kernel](
        input_tensor, output_tensor, scene_graph_tensor,
        grid_dim=(ceildiv(obs.width, 16), ceildiv(obs.height, 16)),
        block_dim=(16, 16),
    )
    ctx.synchronize()
    ctx.enqueue_copy(output_buf, UnsafePointer[C_float](output_rgba))
    return 0
```

**GPU kernel — uses vectorize, NOT dirty for/if:**
```mojo
def denoise_kernel[LT: TensorLayout](
    input: TileTensor[DType.float32, LT, MutAnyOrigin],
    output: TileTensor[DType.float32, LT, MutAnyOrigin],
    scene: TileTensor[DType.float32, LT, MutAnyOrigin],
):
    comptime assert input.flat_rank == 3, "expected H×W×4 tensor"
    var row = global_idx.y
    var col = global_idx.x
    # Bounds-safe via SIMD select — no if-guard needed
    var in_bounds = (row < Int(input.dim[0]())) & (col < Int(input.dim[1]()))
    var val = input[row, col]
    # Process: scene-conditioned denoise via JEPA latent
    val = val * scene[row, col]  # placeholder for actual JEPA inference
    output[row, col] = val
```

### Decision 6: JEPA world model for animation (LeWM port to Mojo/Nabla)

**Choice:** Autoregressive JEPA predictor - direct port of LeWorldModel (Maes et al. 2026, LeCun) to Mojo/Nabla

**Rationale:**
- LeWM proves JEPA works as world model with only 2 losses (prediction + SIGReg)
- ~18M parameters, trains in hours on single GPU, plans 48x faster than foundation models
- Replace LeWM's "robot actions" (Embedder: Conv1d + MLP) with "scene deltas" (SceneDeltaEncoder)
- Predict frames WITHOUT path tracing, only render on surprise

**Architecture (Mojo/Nabla) - exact port from LeWM source:**
```mojo
# Ported from LeWM jepa.py + module.py
struct OmenWorldModel:
    var encoder: ViTEncoder              # HuggingFace ViT-Tiny → CLS token
    var projector: MLP                   # 192 → 2048 → 192 (BatchNorm1d)
    var predictor: ARPredictor           # 6-layer ConditionalBlock transformer
    var scene_delta_encoder: Embedder    # Replaces action_encoder (Conv1d + MLP)
    var pred_proj: MLP                   # 192 → 2048 → 192 (BatchNorm1d)
    var decoder: ImageDecoder            # Latent → RGB (Omen addition)
    var sigreg: SIGReg                   # knots=17, num_proj=1024

    def encode(mut self, info: Dict) -> Dict:
        pixels = info["pixels"].float()          # (B, T, C, H, W)
        b = pixels.shape[0]
        pixels_flat = pixels.reshape(b * t, ...)  # Flatten for ViT
        output = self.encoder(pixels_flat)
        cls_tokens = output[:, 0]                 # CLS token extraction
        emb = self.projector(cls_tokens)
        info["emb"] = emb.reshape(b, t, -1)      # (B, T, 192)
        # Optional binding — no if/else, use Optional chaining
        var delta = info.get("scene_delta")
        match delta:
            case Some(d):
                info["delta_emb"] = self.scene_delta_encoder(d)
            case None:
                pass
        return info

    def predict(mut self, emb: Tensor, delta_emb: Tensor) -> Tensor:
        # emb: (B, T, D=192) delta_emb: (B, T, delta_dim)
        # Nabla handles vectorized matmul internally
        preds = self.predictor(emb, delta_emb)    # ConditionalBlock transformer
        preds = self.pred_proj(preds.reshape(-1, 192))
        return preds.reshape(emb.shape[0], emb.shape[1], -1)

    def rollout(mut self, info: Dict, delta_sequence: Tensor, history_size: Int = 3) -> Dict:
        # Encode initial frames
        var current = self.encode(info)
        var emb = current["emb"]

        comptime HS = history_size
        # Autoregressive rollout — sequential by nature (each step depends on previous)
        # Inner tensor ops are SIMD-native via Nabla (matmul, attention, etc.)
        for t in range(num_steps):
            var delta_emb = self.scene_delta_encoder(delta_sequence[:, t])
            var emb_trunc = emb[:, -HS:]                    # (B, min(T,HS), D)
            var delta_trunc = delta_emb[:, -HS:]            # Match history window
            var pred_emb = self.predict(emb_trunc, delta_trunc)[:, -1:]  # (B, 1, D)
            emb = concat([emb, pred_emb], dim=1)            # Append prediction

        return {"predicted_emb": emb, ...}
```

**SceneDeltaEncoder (replaces LeWM's action_encoder):**
```mojo
# Ported from LeWM module.py Embedder class
# Original: Conv1d(action_dim, smoothed, k=1) + MLP(smoothed → 4*emb → emb)
struct SceneDeltaEncoder:
    var patch_embed: Conv1d   # (delta_dim, smoothed_dim, kernel_size=1)
    var embed: Sequential     # Linear(smoothed, 4*emb) + SiLU + Linear(4*emb, emb)

    def forward(mut self, x: Tensor) -> Tensor:
        # x: (B, T, delta_dim) where delta_dim encodes:
        #   camera_delta (3 trans + 4 quaternion = 7)
        #   + object_deltas (per-object: 7 × num_objects)
        #   + light_deltas (per-light: 6 × num_lights)
        #   + birth_events (type + position + size = 8)
        #   + material_deltas (per-material: 4 × num_materials)
        x = x.float().permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)
```

**FrameState and animation data structures:**
```mojo
struct SceneDelta:
    var camera_delta: TransformDelta       # Translation(3) + quaternion(4) = 7 floats
    var object_deltas: List[ObjectDelta]   # Per-object: obj_id + transform_delta(7)
    var light_deltas: List[LightDelta]     # Per-light: id + intensity + color + pos = 6
    var birth_events: List[SceneElement]   # New fluid/smoke: type + position + size = 8
    var material_deltas: List[MaterialDelta]  # Per-material: id + param_deltas(4)

    def to_tensor(mut self) -> Tensor:
        # Flatten into fixed-size vector for SceneDeltaEncoder
        # Pad to max_objects/max_lights with zeros
        ...

struct FrameState:
    var latent: Tensor                      # (1, 192) current frame embedding
    var history: CircularBuffer[Tensor]     # Last H=3 frame latents
    var scene_graph_hash: String            # Topology hash (face connectivity, not positions)
    var surprise_score: Float32             # MSE(predicted, actual) > 2σ threshold
```

**Animation render flow:**
```
Frame 0 (Anchor):
  mi.render(scene, spp=1) → dirty_0 [H,W,4]
  encode(dirty_0, scene_graph) → latent_0
  denoise(latent_0) → clean_0
  history.push(latent_0)

Frame 1..N:
  mi.render(scene, spp=1) → dirty_N [H,W,4]  (ALWAYS render 1spp, ~30ms)
  scene_delta = compute_delta(frame[N-1], frame[N])
  delta_emb = SceneDeltaEncoder(scene_delta.to_tensor())
  latent_N = encode(dirty_N, scene_graph)
  emb_trunc = history.last(H)                  # history[-H:]
  delta_trunc = delta_emb.last(H)              # match window
  predicted_clean = ARPredictor(emb_trunc, delta_trunc)[:, -1:]
  rgb = decoder(predicted_clean) → output

  IF surprise > 2σ:
    mi.render(scene, spp=4) → denoise() → re-anchor
    history.push(actual_high_quality_latent)
  ELSE:
    history.push(predicted_latent)
```

**Surprise detection (from LeWM paper Section 4.4):**
```
surprise = MSE(predicted_latent, actual_latent)
threshold = running_mean + 2 * running_std  (adaptive threshold)
Auto-surprise for: new fluid/smoke (birth event), new light, material type change
Jump cut detection: camera translation > 1 unit OR rotation > 45° → clear history
```

**Why this is Omen's alpha:** Every frame gets a 1spp render (~30ms at 256x256) for geometric truth. JEPA world model uses history + dirty render + scene delta to produce clean output. No "prediction from nothing" - always grounded in actual render data. 1spp renders are fast enough for real-time but provide exact geometry/occlusion that prediction alone cannot.

### Decision 7: Mitsuba as JEPA training gym (differentiable rendering via Dr.Jit + Nabla autograd)

**Choice:** Use Mitsuba's differentiable rendering pipeline (`*_ad_*` variants with Dr.Jit autodiff) as a closed-loop training environment for the Mojo/Nabla JEPA model. Nabla's autograd replaces PyTorch for weight updates; Dr.Jit handles gradient flow through Mitsuba's path tracer.

**Rationale:**
- Mitsuba's `cuda_ad_rgb` variant provides full autodiff through the rendering equation via Dr.Jit
- Dr.Jit's `dr.backward(loss)` propagates gradients from pixel loss back through the path tracer
- Dr.Jit's `drjit.opt.Adam(lr=...)` optimizer updates scene parameters with gradient scheduling
- `dr.enable_grad(param)` + `dr.schedule(param)` required for tracking differentiable variables
- Forward-mode (`dr.forward()`) available for computing Jacobian-vector products (useful for sensitivity analysis)
- `@dr.wrap(source='torch', target='drjit')` decorator bridges PyTorch ↔ Dr.Jit autograd
- For Mojo/Nabla: Nabla's own autograd handles JEPA weight updates; Dr.Jit handles Mitsuba's rendering gradients
- This makes Mitsuba a "physics gym" for JEPA - like FLIP fluid simulations predicting n+1 from n

**The gym loop (like FLIP fluid sim predicting n-1 from n):**
```
┌──────────────────────────────────────────────────────────────────┐
│                    MITSUBA JEPA GYM (Mojo/Nabla)                 │
│                                                                   │
│  ┌─────────┐    forward       ┌──────────┐                       │
│  │  Scene  │─────────────────▶│  Mitsuba │ 1spp dirty            │
│  │  Graph  │  dr.enable_grad  │  Path    │──────┐                │
│  │ (mi)    │  on parameters   │  Tracer  │      │                │
│  └────▲────┘                  └──────────┘      │                │
│       │                                          ▼                │
│       │                                ┌──────────────────┐      │
│       │                                │  Nabla JEPA      │      │
│       │   Nabla autograd               │  OmenWorldModel  │      │
│       │   (weight updates)             │  .encode()       │      │
│       │                                │  .predict()      │      │
│       │                                │  .decoder()      │      │
│       │                                └──────┬───────────┘      │
│       │                                       │                  │
│       │                                       ▼                  │
│       │                                ┌──────────────┐          │
│       │   Dr.Jit autograd              │  L_pred +     │          │
│       │   (through path tracer)        │  λ*L_sigreg   │          │
│       │                                │  vs 256spp GT │          │
│       │                                └──────┬───────┘          │
│       │                                       │                  │
│       └───────────────────────────────────────┘                  │
│                                                                   │
│  Two autograd systems, one training loop:                         │
│  1. Dr.Jit: loss → path tracer → scene parameter gradients       │
│  2. Nabla: loss → JEPA weights → model weight gradients          │
│  3. Bridge: Dr.Jit tensor → Mojo UnsafePointer → Nabla Tensor    │
│                                                                   │
│  Like FLIP: frame N state → predict N+1 → compare → learn        │
└──────────────────────────────────────────────────────────────────┘
```

**Why this is more than just "generate training pairs":**
- **Gradient signal**: Not just "pixel is wrong" but "pixel is wrong because light at position X has wrong contribution"
- **Closed-loop**: JEPA iteratively improves by "practicing" rendering in Mitsuba's physics
- **Unlimited data**: Every scene, every camera angle, every light configuration is a training sample
- **Physical understanding**: JEPA learns the rendering equation's inverse, not just pixel statistics
- **Dr.Jit forward mode**: `dr.forward()` can compute per-parameter sensitivity for active learning (which scene params matter most for training)

**Dr.Jit autodiff API (exact usage for Omen):**
```python
# Reverse-mode AD (primary training path)
import drjit as dr
from drjit.opt import Adam

# 1. Mark scene parameters as differentiable
param = mi.Float32Array(...)
dr.enable_grad(param)
dr.set_label(param, "scene_param")

# 2. Render through differentiable pipeline
image = mi.render(scene, spp=1, params=param)

# 3. Compute loss
loss = dr.mean(dr.square(image - ground_truth))
dr.set_label(loss, "loss")

# 4. Backward pass
dr.backward(loss)

# 5. Read gradients
grad = dr.grad(param)

# 6. Optimizer step (Dr.Jit Adam)
opt = Adam(lr=0.05)
opt[param] = param
opt.update()  # applies Adam step to param
```

**Forward-mode AD (sensitivity analysis):**
```python
# Forward-mode: compute d(loss)/d(param) for specific directions
dr.enable_grad(param)
image = mi.render(scene, spp=1)
loss = compute_loss(image)

# Seed the gradient of param with a specific direction
dr.set_grad(param, mi.Float32Array([...direction...]))

# Forward pass propagates gradients forward through the computation
dr.forward(loss)

# Read the resulting gradient of the loss
loss_grad = dr.grad(loss)
```

**Training protocol (Mojo/Nabla for JEPA weights, Python/Dr.Jit for Mitsuba rendering):**
```python
import drjit as dr
from drjit.opt import Adam as DrJitAdam

# Phase 1: Generate training data via Mitsuba's differentiable renderer
# (Python side - Dr.Jit handles the path tracer autodiff)
for iteration in range(1000):
    # 1. Mitsuba renders dirty (1spp) + ground truth (256spp)
    dirty = mi.render(scene, spp=1)   # Dr.Jit tensor (cuda_ad_rgb)
    gt = mi.render(scene, spp=256)    # Ground truth reference

    # 2. Extract Dr.Jit tensor data → pass to Mojo/Nabla JEPA via C ABI
    #    dirty_tensor.data.ptr → UnsafePointer → Nabla Tensor
    predicted_clean = jepa_bridge.denoise(scene_graph, dirty)

    # 3. Loss computation (Nabla autograd for JEPA weights)
    #    pred_loss = MSE(predicted_clean, gt)
    #    sigreg_loss = SIGReg(jepa.embeddings)
    #    total_loss = pred_loss + λ * sigreg_loss
    #    Nabla handles: total_loss.backward() → weight gradients

    # 4. Nabla optimizer step (inside Mojo, not Dr.Jit)
    #    nabla_adam.step(jepa_params)
    #    JEPA weight updates happen entirely in Mojo/Nabla

    # Gradient flow:
    # loss → [Nabla autograd] → JEPA weights (Mojo side)
    # loss → [Dr.Jit autograd] → scene params (Python side, optional)
    # No PyTorch needed - Nabla replaces it for the model side
```

## Data Flow Diagrams

### Mode 1: Denoiser

```
Mitsuba Scene
    │
    ├─> scene_extractor.extract()
    │   └─> Geometry[], Material[], Light[], Camera
    │
    ├─> mi.render(spp=4)
    │   └─> noisy_rgba [H, W, 4]
    │
    └─> jepa_bridge.denoise(scene, noisy)
        ├─> Wrap as UnsafePointer
        ├─> Call omen_denoise()
        └─> return clean_rgba [H, W, 4]
```

### Mode 2: Adaptive

```
Mitsuba Scene
    │
    ├─> PASS 1: mi.render(spp=4)
    │   └─> preview [H, W, 4]
    │
    ├─> scene_extractor.extract()
    │   └─> scene_graph
    │
    ├─> jepa_bridge.predict_confidence(scene, preview)
    │   ├─> confidence [H, W, 1] (0=uncertain, 1=confident)
    │   └─> jepa_denoised [H, W, 4]
    │
    ├─> PASS 2: mi.render(spp=128)
    │   └─> high_spp [H, W, 4]
    │
    └─> merge_adaptive(jepa_denoised, high_spp, confidence)
        └─> output [H, W, 4]
```

### Mode 3: Multi-Res

```
Mitsuba Scene
    │
    ├─> PASS 1: mi.render(spp=256, resolution=25%)
    │   └─> low_res_clean [H/4, W/4, 4]
    │
    ├─> PASS 2: mi.render(spp=4, resolution=100%)
    │   └─> high_res_noisy [H, W, 4]
    │
    ├─> scene_extractor.extract()
    │   └─> scene_graph
    │
    └─> jepa_bridge.merge_multires(scene, low_res_clean, high_res_noisy)
        └─> output [H, W, 4]
```

## Component Architecture

### Python Side (`src/omen_integrator/`)

```
scene_extractor.py:
  extract_scene_graph(mi.Scene) -> SceneGraph
    ├─> extract_geometry() -> Geometry[]
    ├─> extract_materials() -> Material[]
    ├─> extract_lights() -> Light[]
    └─> extract_camera() -> Camera

jepa_bridge.py:
  class JEPABridge:
    load() -> None
    denoise(scene, noisy_rgba) -> clean_rgba
    predict_confidence(scene, noisy_rgba) -> (clean_rgba, confidence)
    merge_multires(scene, low_res, high_res) -> merged

modes/denoiser.py:
  render_denoiser(scene, spp=4) -> clean

modes/adaptive.py:
  render_adaptive(scene, spp_target=128) -> clean
    ├─> preview_pass(spp=4)
    ├─> confidence_prediction()
    ├─> high_spp_pass(spp=128)
    └─> merge_by_confidence()

modes/multires.py:
  render_multires(scene, scale=4) -> clean
    ├─> low_res_high_qual_pass(res=0.25, spp=256)
    ├─> high_res_noisy_pass(res=1.0, spp=4)
    └─> scene_guided_merge()
```

### Mojo Side (`jepa_kernels/`) — LeWM Architecture Port

```
C_ABI.mojo:
  @register_function def omen_denoise(...)
  @register_function def omen_predict_confidence(...)
  @register_function def omen_merge_multires(...)
  @register_function def omen_train_step(...)     # NEW: training via C ABI

scene_encoder.mojo:                               # ViT-Tiny (5.5M params)
  struct ViTEncoder:                               # Port of LeWM's encoder
    hidden=192, heads=3, depth=12, patch=14
    encode(pixels: Tensor[B,C,H,W]) -> cls_tokens  # CLS token extraction
  struct Projector:                                # 192 → 2048 → 192 (BatchNorm1d)
    project(cls_tokens) -> embeddings

scene_delta_encoder.mojo:                          # Replaces LeWM's action_encoder (155K params)
  struct SceneDeltaEncoder:
    patch_embed: Conv1d(delta_dim, smoothed, k=1)
    embed: Sequential[Linear(smoothed, 4*emb), SiLU, Linear(4*emb, emb)]
    forward(delta_tensor) -> delta_embeddings

arpredictor.mojo:                                  # Core predictor (10.8M params)
  struct ConditionalBlock:                          # AdaLN-zero conditioning
    attn: Attention(dim=192, heads=16, dim_head=64)
    mlp: FeedForward(dim=192, hidden=2048)
    adaLN: Sequential[SiLU, Linear(192, 1152)]    # 6 modulation params
    forward(x, conditioning) -> Tensor
  struct ARPredictor:                               # 6-layer ConditionalBlock transformer
    layers: List[ConditionalBlock]                  # depth=6
    forward(emb, delta_emb) -> predictions
  struct pred_proj: MLP                             # 192 → 2048 → 192 (789K params)

sigreg.mojo:                                       # SIGReg loss (0 learnable params)
  struct SIGReg:
    t: Tensor          # linspace(0, 3, 17)
    phi: Tensor        # exp(-t²/2) Gaussian characteristic function
    weights: Tensor    # trapezoidal weights * phi
    num_proj: Int = 1024
    forward(embeddings) -> loss                     # Epps-Pulley statistic

world_model.mojo:                                   # Top-level OmenWorldModel (~20M params)
  struct OmenWorldModel:
    encoder: ViTEncoder
    projector: Projector
    predictor: ARPredictor
    scene_delta_encoder: SceneDeltaEncoder
    pred_proj: MLP
    decoder: ImageDecoder                           # Latent → RGB (Omen addition)
    sigreg: SIGReg
    encode(info) -> Dict                            # pixels → CLS → embeddings
    predict(emb, delta_emb) -> predictions          # history + deltas → predicted latent
    rollout(info, deltas, history_size=3) -> Dict   # autoregressive N-step rollout

image_encoder.mojo:                                 # Patch extraction + encoding
  struct ImageEncoder:
    extract_patches(image, patch_size=14) -> Tensor # 8×8 pixel patches
    encode_patches(patches) -> Tensor               # Strided convolutions

jepa.mojo:                                          # Mode-specific inference
  struct JEPAModel:
    var world_model: OmenWorldModel
    def denoise(scene, noisy) -> clean
    def predict_confidence(scene, noisy) -> (clean, confidence)
    def merge_multires(scene, low_res, high_res) -> merged
    def train_step(scene, noisy, gt) -> loss         # Nabla autograd training

confidence.mojo:
  struct ConfidenceHead:
    MLP layers: latent → confidence (sigmoid, 0-1)

multires.mojo:
  struct MultiResMerge:
    merge(low_res, high_res, scene) -> merged        # Geometry-aware upsampling

checkpoint.mojo:
  struct CheckpointManager:
    save(model, optimizer, iteration) -> None         # Nabla state_dict serialization
    load(path) -> (model, optimizer, iteration)       # Resume from checkpoint
    validate_checkpoint(path) -> Bool                 # Architecture hash verification
```

## Training Protocol

### Self-Training Data Generation (Python + Dr.Jit for rendering, Nabla for model)

```python
# Python side: generate training pairs via Mitsuba's differentiable renderer
def generate_training_pair(scene, seed):
    mi.set_seed(seed)
    noisy = mi.render(scene, spp=4)     # Dr.Jit tensor (cuda_ad_rgb)

    mi.set_seed(seed)
    gt = mi.render(scene, spp=256)      # Ground truth reference

    return noisy, gt                    # Passed to Mojo/Nabla via C ABI

# Python side: generate temporal training pairs for animation prediction
def generate_temporal_pair(scene, frame_T, frame_T1):
    dirty_T = mi.render(scene, spp=1, sensor=frame_T.sensor)
    dirty_T1 = mi.render(scene, spp=1, sensor=frame_T1.sensor)
    gt_T1 = mi.render(scene, spp=256, sensor=frame_T1.sensor)

    scene_delta = compute_delta(frame_T, frame_T1)  # camera + object + light deltas
    return dirty_T, dirty_T1, gt_T1, scene_delta
```

### LeWM Training Loss (Mojo/Nabla side)
```mojo
# Inside jepa.mojo or world_model.mojo
# Exact loss from LeWM train.py lejepa_forward

def train_step(model: OmenWorldModel, batch: TrainingBatch) -> Float32:
    # 1. Encode frames
    info = model.encode(batch)

    # 2. Predict next embeddings from history + scene deltas
    #    emb[:, :ctx_len] + delta_emb[:, :ctx_len] → predicted emb[:, n_preds:]
    pred = model.predict(info["emb"][:, :ctx_len], info["delta_emb"][:, :ctx_len])

    # 3. Prediction loss: MSE vs actual future embeddings
    target = info["emb"][:, n_preds:]
    pred_loss = mse_loss(pred, target)

    # 4. SIGReg: prevent representation collapse (Gaussian regularizer)
    sigreg_loss = model.sigreg(info["emb"].transpose(0, 1))

    # 5. Total loss (only 2 terms, no EMA, no pretrained encoder needed)
    total = pred_loss + lambda_weight * sigreg_loss

    # 6. Nabla autograd backward pass
    total.backward()  # Nabla handles gradient computation

    # 7. Nabla Adam optimizer step (inside Mojo)
    # optimizer.step(model.parameters())

    return total
```

### Cornell Box Training Schedule

**Phase 1: Bootstrap denoiser (frames 1-100)**
- Render Cornell box at 4spp + 256spp (Python/Dr.Jit)
- Train JEPA denoiser in Nabla: `pred_loss = MSE(predict(dirty + scene), gt)`
- SIGReg regularization on embeddings
- Target: SSIM > 0.95 vs 256spp

**Phase 2: Confidence head (frames 101-200)**
- Render Cornell box 8× at 4spp → variance map (Python/Dr.Jit)
- Train confidence head in Nabla: MSE vs variance
- Target: predict uncertainty (high variance = low confidence)

**Phase 3: Multi-res merge (frames 201-300)**
- Render at 25% res 256spp + 100% res 4spp (Python/Dr.Jit)
- Render at 100% res 256spp (ground truth)
- Train merge model in Nabla: L1 loss vs ground truth
- Target: PSNR > 30dB vs ground truth

**Phase 4: Temporal prediction (frames 301-500)**
- Render consecutive animation frames at 1spp + 256spp (Python/Dr.Jit)
- Compute scene deltas (camera orbit, light changes)
- Train ARPredictor: `L_pred + λ*L_sigreg` on temporal pairs
- Target: predicted frame SSIM > 0.85 vs 256spp GT

## File Structure

```
jepa_kernels/
├── C_ABI.mojo                    # C interface, SceneGraph/RenderObservation structs
├── scene_encoder.mojo             # ViT-Tiny encoder (5.5M params): CLS token extraction
├── scene_delta_encoder.mojo       # Replaces LeWM's action_encoder (155K params)
│                                  # Conv1d + MLP for camera/object/light/birth/material deltas
├── arpredictor.mojo               # Core predictor (10.8M params): 6-layer ConditionalBlock
│                                  # AdaLN-zero conditioning, Attention(192, 16 heads, dim_head=64)
├── sigreg.mojo                    # SIGReg loss (0 learnable params)
│                                  # Epps-Pulley statistic, knots=17, num_proj=1024
├── world_model.mojo               # Top-level OmenWorldModel (~20M total)
│                                  # encode(), predict(), rollout() with history truncation
├── image_encoder.mojo             # Patch extraction + encoding for dirty renders
├── jepa.mojo                      # Mode-specific inference (denoise, adaptive, multires, train)
├── confidence.mojo                # Confidence head (Mode 2): MLP → sigmoid [0,1]
├── multires.mojo                  # Multi-resolution merge (Mode 3): geometry-aware upsampling
├── checkpoint.mojo                # Save/load model weights via Nabla state_dict, optimizer state
└── build.sh                       # Compile to libomen.so via mojo build

src/omen_integrator/
├── __init__.py                    # Updated: mode parameter
├── scene_extractor.py             # NEW: Mitsuba scene extraction
├── jepa_bridge.py                 # NEW: Ctypes bridge to Mojo .so
├── model_store.py                 # NEW: Model cache, checkpoint management
├── modes/
│   ├── __init__.py
│   ├── denoiser.py                # NEW: Mode 1 orchestration
│   ├── adaptive.py                # NEW: Mode 2 orchestration
│   ├── multires.py                # NEW: Mode 3 orchestration
│   └── animation.py               # NEW: Mode 4 temporal prediction pipeline
└── core.py                        # Updated: dispatch to modes

training/
├── jepa_gym.py                    # NEW: Differentiable training loop (Dr.Jit + Nabla)
├── cornell_box_trainer.py         # NEW: Training data generation (4spp + 256spp pairs)
└── temporal_trainer.py             # NEW: Animation training data (consecutive frames + deltas)

tests/
├── test_scene_extractor.py        # Test Cornell box extraction
├── test_jepa_bridge.py            # Test C ABI loading
├── test_model_store.py            # Test checkpoint save/load
├── test_cornell_denoise.py        # Test Mode 1 on Cornell
├── test_cornell_adaptive.py       # Test Mode 2 on Cornell
├── test_cornell_multires.py       # Test Mode 3 on Cornell
└── test_temporal_prediction.py    # Test Mode 4 animation prediction
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Mojo compilation fails | Provide prebuilt .so for Linux/macOS/Windows in releases |
| C ABI version mismatch | Include version field in structs, graceful degradation |
| Zero-copy buffer incompatibility | Fall back to memcpy if UnsafePointer wrapping fails |
| Training takes too long | Start with pre-trained model, fine-tune on user scenes |
| Multi-pass is slow (3× renders) | JEPA speedup >3× makes it worth it (4-8× fewer samples) |
| Scene extraction incomplete | Support Mitsuba primitives first, extend to custom BSDFs |

## Implementation Phases

### Phase 1: Scene Extraction (Week 1-2)
- [ ] scene_extractor.py from Mitsuba API
- [ ] Test: extract Cornell box geometry (2 boxes, 1 light)
- [ ] Test: extract materials (PrincipledBSDF, RoughBSDF)
- [ ] Test: extract camera (fov, transform)

### Phase 2: C ABI Bridge (Week 2-3)
- [ ] Create C header `omen_bridge.h` with structs
- [ ] Mojo `C_ABI.mojo` with `@register_function`
- [ ] Python `jepa_bridge.py` with ctypes
- [ ] Test: load dummy .so, call simple function

### Phase 3: Mode 1 - Denoiser (Week 4-6)
- [ ] Mojo image encoder (Nabla convolutions)
- [ ] Mojo scene encoder (transformer)
- [ ] Mojo JEPA cross-attention
- [ ] Mojo decoder
- [ ] Training: Cornell box 4spp → 256spp
- [ ] Test: denoise Cornell box, verify SSIM > 0.95

### Phase 4: Mode 2 - Adaptive (Week 7-9)
- [ ] Mojo confidence head
- [ ] Python multi-pass orchestration
- [ ] Adaptive merge in Mojo
- [ ] Training: variance → confidence labels
- [ ] Test: verify 4-8× sample reduction

### Phase 5: Mode 3 - Multi-Res (Week 10-12)
- [ ] Mojo multi-resolution merge
- [ ] Python resolution change orchestration
- [ ] Training: multi-res pairs
- [ ] Test: verify 8-16× speedup

## Model Lifecycle & Continuous Learning

### Storage Architecture

```
~/.cache/omen/models/
├── base_v0.omen                    # Pre-trained (Cornell box + variants)
├── base_v0.omen.meta.json          # Architecture metadata
├── base_v1_local.omen              # Aggregated improvements (local only)
├── scene_index.json                # Feature vector database for similarity
└── scenes/
    └── <scene_hash>/               # SHA256 of geometry+materials+lights
        ├── fine_tuned.omen         # Scene-specific weights
        ├── fine_tuned.omen.meta.json
        └── training_log.json       # SSIM, iterations, timestamps

<project_dir>/.omen/models/         # Project-local cache (optional)
└── <scene_hash>/
    └── fine_tuned.omen             # Overrides global cache if present
```

### Training → Checkpoint Flow (Mojo/Nabla)

```mojo
# Inside training loop (Mojo/Nabla, not Python)
# Data generation happens in Python/Dr.Jit, model training in Mojo/Nabla

def training_loop(model: OmenWorldModel, data_iter: DataIterator):
    var optimizer = NablaAdamW(lr=5e-5, weight_decay=1e-3)  # From lewm.yaml config
    var ckpt_mgr = CheckpointManager()

    for iteration in range(total_iterations):
        # Get training pair (generated by Python side, passed via C ABI)
        var batch = data_iter.next()

        # Forward + backward in Nabla
        var loss = train_step(model, batch)

        # Nabla optimizer step
        optimizer.step(model.trainable_params())

        # Checkpoint every 10 iterations (Nabla state_dict serialization)
        if iteration % 10 == 0:
            ckpt_mgr.save(model, optimizer, iteration, loss)
            ckpt_mgr.update_symlink("latest.omen")

    ckpt_mgr.save(model, optimizer, total_iterations, loss)
```

### Model Selection Strategy

```
User renders scene
    ↓
Compute scene hash (geometry + materials + lights)
    ↓
Check project-local cache → Found? Load it
    ↓ No
Check global cache → Found? Load it
    ↓ No
Compute scene feature vector
    ↓
Query scene_index.json for similar scenes (cosine > 0.85)
    ↓ Found
Load similar model → Quick adaptation (10 iters) → Save as fine-tuned
    ↓ Not found
Load base model → Full fine-tuning (50 iters) → Save as fine-tuned
```

### Continuous Learning Modes

| Mode | Privacy | Data Flow | Benefit |
|------|----------|-----------|---------|
| **Base only** | Default | None | Works offline, no training needed |
| **Local aggregation** | Private | Local only | Models improve from your scenes |
| **Anonymous contribution** | Opt-in | Weight deltas only | Improves Omen for everyone |

### GPU Rendering Configuration

```python
class JEPABridge:
    def __init__(self):
        # Detect Mitsuba backend
        self.variant = mi.variant()
        self.is_mitsuba_gpu = self.variant.startswith(('cuda', 'metal', 'llvm'))

        # Detect JEPA device
        self.jepa_device_id = self._detect_gpu_device()

        # Configure buffer passing
        if self.is_mitsuba_gpu and self.jepa_device_id >= 0:
            self.buffer_mode = "zero_copy"  # Fast path!
        elif self.jepa_device_id >= 0:
            self.buffer_mode = "memcpy_cpu_to_gpu"  # Fallback
        else:
            self.buffer_mode = "cpu"  # CPU inference

    def denoise(self, scene, noisy_rgba):
        if self.buffer_mode == "zero_copy":
            # Wrap GPU pointer directly, no memcpy
            ptr = noisy_rgba.data.ptr
            return self._omen_denoise_gpu(scene, ptr, self.jepa_device_id)
        elif self.buffer_mode == "memcpy_cpu_to_gpu":
            # Copy CPU render to GPU first
            gpu_buffer = self._copy_to_gpu(noisy_rgba)
            result = self._omen_denoise_gpu(scene, gpu_buffer, self.jepa_device_id)
            return self._copy_to_cpu(result)
        else:
            # CPU inference
            return self._omen_denoise_cpu(scene, noisy_rgba)
```

## Open Questions

1. **Q:** Which Mojo ML framework for neural ops?
   **A:** Nabla (nabla-ml) - has autograd, GPU, SPMD

2. **Q:** How to handle variable scene sizes in transformer?
   **A:** Pad to max size, use mask tokens (like BERT)

3. **Q:** Where to store trained JEPA models?
   **A:** `~/.cache/omen/models/` with project-local override

4. **Q:** Cornell box only for training?
   **A:** Start with Cornell (simple), extend to variety of scenes

5. **Q:** GPU memory budget for JEPA model?
   **A:** Target ~2GB for model + scene graph (fits on 8GB GPUs)

6. **Q:** How to share improvements across users?
   **A:** Opt-in anonymous upload of weight deltas (de-identified)

7. **Q:** Zero-copy buffer compatibility?
   **A:** Requires Mitsuba GPU variant + same GPU device, fallback to memcpy
