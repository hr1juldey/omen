## Context

Omen Mitsuba integrator skeleton exists (previous change). Current state:
- `OmenIntegrator` registered with Mitsuba plugin system
- Standard path tracing working via `mi.render()`
- Placeholder parameters: `jepa_model`, `use_gpu`

**What's missing:** Actual JEPA implementation with scene conditioning.

**Critical constraint:** Mitsuba's path tracer is C++. Python `mi.render()` is a binding — cannot inject into sampling loop. JEPA works via multi-pass rendering: render → extract → JEPA → render → merge.

**Why JEPA is different from 2D denoisers:**
- OptiX/OIDN: only see 2D pixels + normals
- Omen JEPA: sees exact 3D scene (geometry, materials, lights) from Mitsuba
- Self-training advantage: render YOUR scene at 4spp + 256spp → perfect pairs, unlimited data

**Infrastructure decision (revised after Nabla study):**
Both Mitsuba and Nabla are Python-callable libraries. Nabla has NO Mojo API — it's `import nabla as nb` in Python with Mojo/MAX backend for execution. This means:

1. **No C ABI bridge needed** — use DLPack zero-copy between Dr.Jit and Nabla tensors
2. **All ML code stays in Python** — Nabla Python API with `@nb.compile` for JIT to Mojo/MAX
3. **Custom Mojo GPU kernels** — for SIGReg loss, edge-aware merge via `call_custom_kernel()`
4. **Production path** — compile trained model to MAX format, load via MAX Engine C API for Python-free inference

**Test scene:** Cornell box (`mi.cornell_box()`) renders in 2s at 256x256. Target: same quality in <500ms.

## Goals / Non-Goals

**Goals:**
- Implement scene graph extraction from Mitsuba Python API
- Build JEPA model using Nabla Python API (`nb.nn.Module`, `nb.nn.optim.AdamW`)
- Implement 3 rendering modes (denoiser, adaptive, multires)
- Self-training protocol using Cornell box
- Zero-copy DLPack tensor passing between Mitsuba/Dr.Jit and Nabla
- Custom Mojo GPU kernels for SIGReg loss and merge operations

**Non-Goals:**
- Modifying Mitsuba C++ path tracer source
- Per-pixel adaptive sampling within single render (C++ limitation)
- Material node compilation (use Mitsuba's BSDFs)
- Pure Mojo model code (Nabla has no Mojo API — Python only)
- C ABI bridge (replaced by DLPack via Nabla Python API)

## Decisions

### Decision 1: 3D-Aware Scene Encoder (NOT ViT-Tiny)

**Choice:** Replace LeWM's 2D ViT-Tiny image encoder with a 3D-aware dual-encoder that leverages Mitsuba's exact scene data. Total model: ~8M params (down from ~20M).

**Rationale:**
- LeWM's ViT-Tiny treats renders as 2D images (patch embedding) — wastes Mitsuba's 3D data
- Mitsuba gives us EXACT geometry, materials, and light positions — use them directly
- A ViT-Tiny designed for ImageNet classification has no 3D understanding
- Smaller model = faster training, less GPU memory, faster inference

**New architecture: Scene-Aware Dual Encoder**

```
Component                  Params    Implementation              Nabla Ops
────────────────────────────────────────────────────────────────────────────────
Scene Graph Encoder        ~1M       scene_graph_encoder.py      nb.nn.Embedding
  (geometry/material/light embeddings, NOT image patches)    nb.nn.Linear, F.attention
  Encodes: vertices, face normals, material params, light positions/properties

Render Feature Encoder     ~1.5M     render_encoder.py           nb.nn.Conv2d
  (noisy RGBA + aux buffers from Mitsuba)                     nb.nn.Linear
  Input: noisy RGBA(H,W,4) + depth(H,W,1) + normal(H,W,3)
  Conv2d layers → flatten → Linear → latent(192)

Cross-Attention Fusion     ~0.5M     fusion.py                   F.scaled_dot_product_attention
  Scene features as keys/values, render features as queries
  Output: scene-aware latent (1, 192)

SceneDeltaEncoder          ~155K     scene_delta_encoder.py      nb.nn.Conv1d, nb.nn.Linear
  (from LeWM — Conv1d + MLP for per-frame scene changes)

ARPredictor                ~4M       arpredictor.py              nb.nn.TransformerEncoderLayer
  (simplified from LeWM's 10.8M — 4 layers instead of 6, dim=192, heads=8)
  Uses ConditionalBlock with AdaLN-zero conditioning

ConfidenceHead             ~30K      confidence_head.py          nb.nn.Linear, F.sigmoid
  Linear(192,96) → SiLU → Linear(96,48) → SiLU → Linear(48,1) → Sigmoid

Decoder                    ~1M       decoder.py                  nb.nn.Linear, nb.conv2d_transpose
  Latent(192) → upsample → RGBA(H,W,4)

SIGReg                     0         sigreg_kernel/              Custom Mojo GPU kernel
  knots=17, num_proj=1024, Epps-Pulley statistic                via call_custom_kernel()
────────────────────────────────────────────────────────────────────────────────
TOTAL                      ~8M
```

**Why this is better than ViT-Tiny:**

| Aspect | ViT-Tiny (LeWM) | Scene-Aware Encoder (Omen) |
|--------|-----------------|---------------------------|
| Input | 2D image patches | 3D scene graph + noisy render + aux buffers |
| 3D understanding | None ( learns from pixels) | Explicit (geometry, materials, lights) |
| Parameters | 5.5M (encoder alone) | ~3M (both encoders + fusion) |
| Scene changes | Must re-encode entire image | Only re-encode delta |
| Generalization | Scene-specific pixels | 3D structure transfers |

**Scene Graph Encoder detail:**
```python
class SceneGraphEncoder(nb.nn.Module):
    """Encode Mitsuba scene data into a fixed-size latent vector.

    NOT a ViT — uses structured embeddings for known scene elements.
    """
    def __init__(self, latent_dim=192):
        # Geometry: vertex positions + face normals → aggregate via attention
        self.geo_embed = nb.nn.Linear(6, 64)   # (pos_xyz, normal_xyz)
        self.geo_attn = nb.nn.MultiHeadAttention(64, num_heads=4)

        # Materials: type_id + parameters → per-face embedding
        self.mat_embed = nb.nn.Embedding(num_material_types, 64)
        self.mat_proj = nb.nn.Linear(64 + 8, 64)  # type_emb + 8 params

        # Lights: type_id + position + intensity + color
        self.light_embed = nb.nn.Linear(7, 64)    # type + pos + intensity + rgb

        # Aggregate all features via attention → single vector
        self.aggregate = nb.nn.MultiHeadAttention(64, num_heads=4)
        self.out_proj = nb.nn.Linear(64, latent_dim)

    def forward(self, scene_features):
        # scene_features: dict with geometry, materials, lights tensors
        geo = self.geo_attn(self.geo_embed(scene_features['geometry']),
                            self.geo_embed(scene_features['geometry']),
                            self.geo_embed(scene_features['geometry']))
        mat = self.mat_proj(cat(self.mat_embed(scene_features['mat_ids']),
                                scene_features['mat_params']))
        light = self.light_embed(scene_features['lights'])

        # Concatenate and aggregate via cross-attention
        all_features = cat([geo.mean(axis=1), mat.mean(axis=1), light.mean(axis=1)])
        return self.out_proj(all_features)
```

**Render Feature Encoder detail:**
```python
class RenderFeatureEncoder(nb.nn.Module):
    """Encode noisy render + auxiliary buffers into latent space.

    Uses Conv2d (not ViT patches) — designed for pixel data with spatial structure.
    """
    def __init__(self, latent_dim=192):
        # Input channels: RGBA(4) + depth(1) + normal(3) = 8 channels
        self.conv1 = ...  # Conv2d(8, 32, 3x3, stride=2)  → H/2, W/2
        self.conv2 = ...  # Conv2d(32, 64, 3x3, stride=2) → H/4, W/4
        self.conv3 = ...  # Conv2d(64, 128, 3x3, stride=2) → H/8, W/8
        self.pool = ...   # Global average pool → (128,)
        self.proj = nb.nn.Linear(128, latent_dim)

    def forward(self, noisy_rgba, depth=None, normal=None):
        # Stack channels: RGBA + optional aux buffers
        x = noisy_rgba  # (B, H, W, 4) — NHWC layout
        if depth is not None:
            x = nb.concat([x, depth], axis=-1)
        if normal is not None:
            x = nb.concat([x, normal], axis=-1)
        x = nb.relu(self.conv1(x))
        x = nb.relu(self.conv2(x))
        x = nb.relu(self.conv3(x))
        x = nb.mean(x, axis=(1, 2))  # Global pool
        return self.proj(x)  # → (B, 192)
```

### Decision 2: Nabla Python API — No C ABI Bridge

**Choice:** Use Nabla's Python API directly. Both Mitsuba and Nabla are Python libraries. Tensor interop via DLPack (`nb.Tensor.from_dlpack()`). No ctypes, no C header, no shared library compilation.

**Rationale:**
- Nabla has NO Mojo API — only `import nabla as nb` in Python
- Mitsuba has NO Mojo API — only `import mitsuba as mi` in Python
- Both in Python → no bridge needed, just function calls
- DLPack provides zero-copy GPU tensor transfer between Dr.Jit and Nabla
- `@nb.compile` JIT-compiles Python/Nabla code to MAX/Mojo for GPU execution
- Custom Mojo kernels via `call_custom_kernel()` for SIGReg, merge ops
- Eliminates: `C_ABI.mojo`, `omen_bridge.h`, `jepa_bridge.py` (ctypes), `libomen.so` build

**Old architecture (C ABI):**
```
Mitsuba → Python → numpy → ctypes → C ABI → Mojo .so → GPU
```

**New architecture (Nabla Python):**
```
Mitsuba → Python → DLPack → Nabla Python → @nb.compile → MAX Engine → GPU
```

**Production path (Python-free inference):**
```
Training: Nabla Python → compile model → export .max file
Runtime: Mitsuba → numpy → MAX Engine C API → GPU (no Python ML runtime)
```

**Tensor interop code:**
```python
import mitsuba as mi
import nabla as nb

# Mitsuba renders → Dr.Jit tensor → DLPack → Nabla tensor
noisy_dr = mi.render(scene, spp=4)        # Dr.Jit TensorXf
noisy_np = np.array(noisy_dr)             # to numpy (or direct DLPack if supported)
noisy_nb = nb.Tensor.from_dlpack(noisy_np) # to Nabla tensor (zero-copy on GPU)

# Or directly if Dr.Jit supports __dlpack__:
noisy_nb = nb.Tensor.from_dlpack(noisy_dr) # zero-copy, no intermediate numpy

# Run JEPA inference
clean = model.denoise(noisy_nb.cuda(), scene_features)

# Back to numpy for saving
clean_np = clean.cpu().to_numpy()
```

### Decision 3: Multi-pass rendering strategy

**Choice:** Full-image re-renders at different spp, not tile-based

**Rationale:**
- Mitsuba Python API doesn't support crop rendering
- `mi.render()` always renders full frame
- Merge happens after both renders complete (not per-tile)

**Mode 2 adaptive flow:**
```python
# PASS 1: Preview (4 spp)
preview = mi.render(scene, spp=4)
preview_nb = nb.Tensor.from_dlpack(np.array(preview)).cuda()
confidence, clean_preview = model.predict_confidence(scene_features, preview_nb)

# PASS 2: High-spp (128 spp)
high_spp = mi.render(scene, spp=128)
high_nb = nb.Tensor.from_dlpack(np.array(high_spp)).cuda()

# Merge: per-pixel based on confidence
output = confidence * clean_preview + (1 - confidence) * high_nb
```

### Decision 4: Scene graph representation

**Choice:** Python dicts/tensors, not C structs

**Rationale:**
- No C ABI → no need for C-compatible memory layout
- Nabla consumes Python tensors directly
- Variable-length scenes handled naturally with Python lists/dicts

**Scene features dict:**
```python
def extract_scene_features(scene):
    """Extract structured scene features from Mitsuba scene.

    Returns a dict of Nabla-ready tensors (no C packing needed).
    """
    shapes = scene.shapes()
    emitters = scene.emitters()
    sensors = scene.sensors()

    # Geometry: concatenate all mesh vertices + face normals
    vertices, normals, face_mat_ids = [], [], []
    for shape in shapes:
        params = mi.traverse(shape)
        verts = np.array(params['vertex_positions']).reshape(-1, 3)
        norms = np.array(params['vertex_normals']).reshape(-1, 3) if shape.has_vertex_normals() else np.zeros_like(verts)
        mat_id = shape.bsdf().__class__.__name__  # material type
        vertices.append(verts)
        normals.append(norms)

    # Materials: extract per-shape BSDF parameters
    materials = []
    for shape in shapes:
        bsdf = shape.bsdf()
        params = mi.traverse(bsdf)
        mat_type = material_type_id(bsdf)  # diffuse=0, glass=1, metal=2, etc.
        mat_params = extract_bsdf_params(params)  # 8 floats
        materials.append([mat_type] + mat_params)

    # Lights: position, intensity, color, type
    lights = []
    for emitter in emitters:
        if emitter.is_environment():
            lights.append([2, 0,0,0, 1,1,1])  # type + pos + intensity + color
        else:
            params = mi.traverse(emitter)
            lights.append(extract_light_params(params))

    return {
        'geometry': np.concatenate([np.concatenate([v, n], axis=1) for v, n in zip(vertices, normals)]),
        'materials': np.array(materials, dtype=np.float32),
        'lights': np.array(lights, dtype=np.float32),
        'camera': extract_camera(sensors[0]),
        'n_objects': len(shapes),
        'n_lights': len(emitters),
    }
```

### Decision 5: Self-training on Cornell box

**Choice:** Render same scene at multiple spp levels for training pairs. All training in Python using Nabla.

**Protocol:**
1. Render Cornell box at 4 spp → noisy input
2. Render Cornell box at 256 spp → ground truth
3. Extract scene features (geometry, materials, lights)
4. Train in Nabla: `model.train()` → `loss.backward()` → `optimizer.step()`
5. Repeat for 1000 iterations (different camera angles, light positions)

### Decision 6: Training with Nabla PyTorch-style API

**Choice:** Use Nabla's imperative (PyTorch-style) training API for development.

**Rationale:**
- `nb.nn.Module` with `forward()`, `parameters()`, `state_dict()`
- `loss.backward()` + `optimizer.step()` — familiar pattern
- Natural for variable-length scene graphs
- Can migrate to functional JAX-style + `@nb.compile` for production

**Training loop:**
```python
import nabla as nb
import nabla.nn.functional as F

model = OmenJEPA(latent_dim=192)
model.train()
optimizer = nb.nn.optim.AdamW(model, lr=5e-5, weight_decay=1e-3)

for iteration in range(500):
    model.zero_grad()

    # Generate training pair (Mitsuba renders)
    noisy = mi.render(scene, spp=4, seed=iteration)
    gt = mi.render(scene, spp=256, seed=iteration)
    scene_features = extract_scene_features(scene)

    # Convert to Nabla tensors
    noisy_nb = nb.Tensor.from_dlpack(np.array(noisy)).cuda()
    gt_nb = nb.Tensor.from_dlpack(np.array(gt)).cuda()

    # Forward pass
    predicted = model(noisy_nb, scene_features)

    # Loss: prediction + SIGReg
    pred_loss = nb.mean(nb.square(predicted - gt_nb))
    sigreg_loss = sigreg(model.get_embeddings())  # Custom Mojo kernel
    total_loss = pred_loss + 0.09 * sigreg_loss

    # Backward + step
    total_loss.backward()
    model = optimizer.step()

    # Checkpoint every 10 iterations
    if iteration % 10 == 0:
        save_checkpoint(model, optimizer, iteration)
```

### Decision 7: Custom Mojo kernels for SIGReg and merge

**Choice:** Write SIGReg loss and edge-aware merge as custom Mojo GPU kernels via Nabla's `call_custom_kernel()`.

**Rationale:**
- SIGReg's Epps-Pulley statistic with 17 knots and 1024 projections is compute-heavy
- Edge-aware merge needs per-pixel scene graph lookup — custom kernel is faster
- Nabla's `UnaryOperation` / `Operation` base classes handle autograd integration
- Custom ops compose with `nb.grad`, `nb.vmap`, `@nb.compile`

**SIGReg custom op:**
```python
from nabla.ops import UnaryOperation, call_custom_kernel
from pathlib import Path

class SIGRegOp(UnaryOperation):
    @property
    def name(self): return "sigreg"

    def kernel(self, args, kwargs):
        embeddings = args[0]
        result = call_custom_kernel(
            "sigreg_kernel",
            Path("kernels/"),
            embeddings,
            embeddings.type,
        )
        return [result]

    def vjp_rule(self, primals, output, ct):
        # SIGReg gradient: derivative of Epps-Pulley statistic
        ...
```

**SIGReg Mojo kernel (in `kernels/sigreg_kernel.mojo`):**
```mojo
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, foreach
from utils.index import IndexList

@compiler.register("sigreg_kernel")
struct SIGRegKernel:
    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor,
        embeddings: InputTensor[dtype=output.dtype, rank=output.rank],
        ctx: DeviceContextPtr,
    ):
        # SIGReg: Epps-Pulley statistic with 17 knots, 1024 projections
        # Compute on GPU — custom reduction kernel
        ...
```

**Merge custom op:**
```python
class MergeOp(Operation):
    @property
    def name(self): return "omen_merge"

    def kernel(self, args, kwargs):
        low_res, high_res, edge_map = args
        result = call_custom_kernel(
            "merge_kernel",
            Path("kernels/"),
            low_res, high_res, edge_map,
            low_res.type,
        )
        return [result]

    def vjp_rule(self, primals, output, ct):
        # Gradient flows through merge
        ...
```

### Decision 8: JEPA world model for animation (simplified from LeWM)

**Choice:** Autoregressive JEPA predictor using Nabla built-in TransformerEncoderLayer. Scene deltas replace robot actions.

**ARPredictor (using Nabla built-in layers):**
```python
class ARPredictor(nb.nn.Module):
    def __init__(self, dim=192, depth=4, heads=8, dim_head=64, mlp_dim=1024):
        # Use Nabla's built-in TransformerEncoderLayer for attention/FFN
        self.layers = [
            ConditionalBlock(dim, heads, dim_head, mlp_dim)
            for _ in range(depth)
        ]
        self.norm = nb.nn.LayerNorm(dim)

    def forward(self, history_emb, current_emb, delta_emb):
        # Concatenate history + current → transformer input
        x = nb.concat([history_emb, current_emb.unsqueeze(1)], axis=1)

        # Process through ConditionalBlocks with scene delta conditioning
        for layer in self.layers:
            x = layer(x, delta_emb)

        return self.norm(x[:, -1])  # Last token = prediction

class ConditionalBlock(nb.nn.Module):
    """AdaLN-zero conditioning block (from LeWM)."""
    def __init__(self, dim, heads, dim_head, mlp_dim):
        self.attn = nb.nn.MultiHeadAttention(dim, heads)
        self.mlp = FeedForward(dim, mlp_dim)
        self.norm1 = nb.nn.LayerNorm(dim)
        self.norm2 = nb.nn.LayerNorm(dim)
        self.adaLN = AdaLNModulation(dim)  # SiLU + Linear(dim, dim * 6)

    def forward(self, x, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(cond).chunk(6, axis=-1)

        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            modulate(self.norm1(x), shift_msa, scale_msa),
            modulate(self.norm1(x), shift_msa, scale_msa),
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```

### Decision 9: Inference compilation for production

**Choice:** Use `@nb.compile` for JIT compilation of inference paths. Optionally export to MAX format for C API deployment.

**Compiled inference:**
```python
@nb.compile
def omen_denoise_compiled(model_weights, noisy_render, scene_features):
    """JIT-compiled inference — Nabla traces and compiles to MAX graph."""
    latent = encode(noisy_render, scene_features)
    clean = decode(latent)
    return clean

# Production: export compiled model
# model.max can be loaded via MAX Engine C API without Python
```

## Data Flow Diagrams

### Mode 1: Denoiser

```
Mitsuba Scene
    │
    ├─> extract_scene_features(scene)
    │   └─> {geometry, materials, lights, camera}
    │
    ├─> mi.render(spp=4)
    │   └─> noisy_rgba [H, W, 4] (Dr.Jit tensor)
    │
    └─> model.denoise(features, noisy)
        ├─> nb.Tensor.from_dlpack(noisy)  # zero-copy to Nabla
        ├─> SceneEncoder(features)         # 3D scene → latent
        ├─> RenderEncoder(noisy)           # noisy image → latent
        ├─> CrossAttention(scene_latent, render_latent)
        ├─> Decoder(combined_latent)       # latent → clean RGBA
        └─> output.cpu().to_numpy()        # back to numpy
```

### Mode 2: Adaptive

```
Mitsuba Scene
    │
    ├─> PASS 1: mi.render(spp=4) → preview
    ├─> extract_scene_features(scene) → features
    ├─> model.predict_confidence(features, preview)
    │   └─> (clean_preview, confidence [H,W,1])
    ├─> PASS 2: mi.render(spp=128) → high_spp
    └─> merge: confidence * clean_preview + (1 - confidence) * high_spp
```

### Mode 3: Multi-Res

```
Mitsuba Scene
    │
    ├─> PASS 1: mi.render(spp=256, res=25%) → low_res_clean
    ├─> PASS 2: mi.render(spp=4, res=100%) → high_res_noisy
    ├─> extract_scene_features(scene) → features + edge_map
    └─> merge_kernel(low_res_clean, high_res_noisy, edge_map) → output
        └─> Custom Mojo GPU kernel via call_custom_kernel()
```

## Component Architecture

### Python Side — All code is Python (Nabla for ML, Mitsuba for rendering)

```
src/omen/
├── __init__.py
├── model/
│   ├── __init__.py
│   ├── omen_jepa.py              # Top-level OmenJEPA model (nb.nn.Module)
│   ├── scene_encoder.py           # SceneGraphEncoder (~1M params)
│   ├── render_encoder.py          # RenderFeatureEncoder (~1.5M params, Conv2d)
│   ├── fusion.py                  # Cross-attention fusion (~0.5M params)
│   ├── arpredictor.py             # ARPredictor + ConditionalBlock (~4M params)
│   ├── scene_delta_encoder.py     # SceneDeltaEncoder (~155K params)
│   ├── confidence_head.py         # ConfidenceHead (~30K params)
│   ├── decoder.py                 # Image decoder (~1M params, Conv2dTranspose)
│   └── layers.py                  # AdaLNModulation, modulate(), FeedForward helpers
│
├── kernels/                       # Custom Mojo GPU kernels
│   ├── __init__.mojo              # Empty init
│   ├── sigreg_kernel.mojo         # SIGReg Epps-Pulley statistic
│   ├── merge_kernel.mojo          # Edge-aware multires merge
│   └── sigreg_op.py               # Python wrapper: UnaryOperation subclass
│
├── scene/
│   ├── __init__.py
│   ├── extractor.py               # extract_scene_features(mi.Scene) → dict
│   └── delta.py                   # compute_delta(frame_A, frame_B) → SceneDelta
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                 # Training loop (Nabla AdamW + checkpoints)
│   ├── gym.py                     # Training data generation (Mitsuba renders)
│   └── cornell_schedule.py        # 4-phase Cornell box training schedule
│
├── modes/
│   ├── __init__.py
│   ├── denoiser.py                # Mode 1: 4spp → denoise → clean
│   ├── adaptive.py                # Mode 2: preview + confidence + high-spp → merge
│   ├── multires.py                # Mode 3: low-res clean + high-res noisy → merge
│   └── animation.py               # Mode 4: temporal prediction + surprise detection
│
├── checkpoint.py                  # Save/load model state_dict, LoRA adapters
├── inference.py                   # @nb.compile inference functions
└── config.py                      # Model config, hyperparameters from lewm.yaml
```

### No Mojo `.so` compilation needed

The only Mojo code is in `kernels/` — custom GPU ops that Nabla's `call_custom_kernel()` compiles on-demand. No separate build step, no `libomen.so`.

## Training Protocol

### 4-Phase Cornell Box Schedule (Nabla Python)

**Phase 1: Bootstrap denoiser (iterations 1-100)**
```python
model.train()
optimizer = nb.nn.optim.AdamW(model, lr=5e-5, weight_decay=1e-3)

for i in range(100):
    model.zero_grad()
    noisy = mi.render(scene, spp=4, seed=i)
    gt = mi.render(scene, spp=256, seed=i)
    features = extract_scene_features(scene)

    predicted = model(nb.Tensor.from_dlpack(np.array(noisy)).cuda(), features)
    gt_nb = nb.Tensor.from_dlpack(np.array(gt)).cuda()

    pred_loss = nb.mean(nb.square(predicted - gt_nb))
    sigreg_loss = sigreg_op(model.get_embeddings())
    total = pred_loss + 0.09 * sigreg_loss

    total.backward()
    model = optimizer.step()
```

**Phase 2: Confidence head (iterations 101-200)**
- Render 8× at 4spp → variance map → confidence labels
- Train ConfidenceHead: MSE(predicted_confidence, 1 - normalized_variance)

**Phase 3: Multi-res merge (iterations 201-300)**
- Train merge_kernel on (25% res 256spp, 100% res 4spp, 100% res 256spp GT) triplets

**Phase 4: Temporal prediction (iterations 301-500)**
- Train ARPredictor on consecutive animation frames + scene deltas
- Loss: `L_pred + 0.09 * L_sigreg`

### Checkpointing

```python
# Save
state = model.state_dict()  # OrderedDict[str, Tensor]
optimizer_state = optimizer.state_dict()
save_checkpoint(path, model_state=state, opt_state=optimizer_state, iteration=i)

# Load
state = load_checkpoint(path)
model.load_state_dict(state['model'])
# Resume training from saved iteration
```

### Scene-Specific Fine-Tuning (LoRA)

```python
from nabla.nn.finetune import init_lora_adapter, lora_linear, save_finetune_checkpoint

# Add LoRA adapters to encoder weights
for name, weight in model.named_parameters():
    if 'encoder' in name:
        adapters[name] = init_lora_adapter(weight, rank=8)

# Train only LoRA adapters (frozen base model)
save_finetune_checkpoint(scene_cache_path, lora_params=adapters, optimizer_state=opt_state)
```

## Model Lifecycle

```
~/.cache/omen/models/
├── base_v0.omen                    # Pre-trained on Cornell box variants
├── base_v0.omen.meta.json          # Architecture hash, version, metrics
└── scenes/
    └── <topology_hash>/            # Face connectivity + material types + light types
        ├── lora_adapter.omen       # Scene-specific LoRA weights
        └── meta.json               # SSIM, iterations, training config
```

**Scene model selection:**
1. Compute topology hash (geometry connectivity, material types, light types)
2. Check cache for matching hash → load LoRA adapter
3. No match → use base model, optionally fine-tune with LoRA (50 iterations)

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Nabla API instability | Pin nabla-ml version, wrap in Omen abstractions |
| DLPack zero-copy fails | Fallback: numpy copy (10-50ms overhead) |
| Custom Mojo kernel bugs | Provide numpy reference impl, validate against it |
| Nabla missing ops | Implement as custom ops via `call_custom_kernel()` |
| Training too slow | Use `@nb.compile` for training loop, LoRA for fine-tuning |
| Multi-pass is slow (3x renders) | JEPA speedup >3x makes it worth it |
| Scene extraction incomplete | Support Mitsuba primitives first, extend to custom BSDFs |

## Implementation Phases

### Phase 1: Scene Extraction + Model Skeleton (Week 1-2)
- [ ] `scene/extractor.py` from Mitsuba API
- [ ] `model/scene_encoder.py` (SceneGraphEncoder)
- [ ] `model/render_encoder.py` (RenderFeatureEncoder, Conv2d)
- [ ] `model/fusion.py` (Cross-attention)
- [ ] Test: extract Cornell box features, encode to latent

### Phase 2: Core Model + Training (Week 3-5)
- [ ] `model/arpredictor.py` (ConditionalBlock + AdaLN-zero)
- [ ] `model/decoder.py` (Conv2dTranspose upsample)
- [ ] `kernels/sigreg_kernel.mojo` + `sigreg_op.py`
- [ ] `training/trainer.py` (Nabla training loop)
- [ ] Test: train Phase 1 on Cornell box, verify SSIM > 0.95

### Phase 3: Modes (Week 6-8)
- [ ] `modes/denoiser.py` (Mode 1)
- [ ] `modes/adaptive.py` (Mode 2 + ConfidenceHead)
- [ ] `modes/multires.py` (Mode 3 + merge kernel)
- [ ] `checkpoint.py` (save/load state_dict)
- [ ] Test: all 3 modes on Cornell box

### Phase 4: Temporal + Animation (Week 9-12)
- [ ] `model/scene_delta_encoder.py`
- [ ] `modes/animation.py` (surprise detection, history buffer)
- [ ] `training/cornell_schedule.py` (4-phase training)
- [ ] `inference.py` (`@nb.compile` inference)
- [ ] Test: 100-frame animation with surprise detection
