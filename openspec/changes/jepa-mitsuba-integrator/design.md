## Context

Omen is a **render engine turbocharger** — it sits above ANY path tracer and makes it faster, smarter, and scene-aware. Today: Mitsuba 3. Tomorrow: Cycles, EEVEE, any engine.

Omen fights on three fronts:
1. **vs DLSS 4.0** (upscale + frame gen): NVIDIA-locked, pixel-level, scene-blind. Omen understands 3D scene data.
2. **vs OIDN/OptiX** (denoising): Scene-blind pixel denoisers. OIDN balanced = 460K params. Omen sees exact materials, lights, geometry.
3. **vs Diffusion models** (generation): Text-controlled, stochastic, non-reproducible. Omen takes EXACT 3D parameters — deterministic, artist-controlled.

**Architecture paradigm:**
```
JEPA = the BRAIN  (scene understanding + temporal prediction, 1024-dim latent)
U-Net = the HANDS (fast pixel-level denoising, conditioned by scene latent)
```

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
- Implement scene graph extraction from Mitsuba Python API (portable to Cycles/EEVEE later)
- Build 3-tier JEPA + U-Net model using Nabla Python API (4M / 16M / 64M params)
- Implement 4 rendering modes (denoiser, adaptive, multires, temporal) — all tiers support all modes
- Self-training protocol using Cornell box
- Energy conservation loss (physics-based, no photon creation during denoising)
- Zero-copy DLPack tensor passing between Mitsuba/Dr.Jit and Nabla
- Custom Mojo GPU kernels for SIGReg loss and merge operations

**Non-Goals:**
- Modifying Mitsuba C++ path tracer source
- Per-pixel adaptive sampling within single render (C++ limitation)
- Material node compilation (use Mitsuba's BSDFs)
- Pure Mojo model code (Nabla has no Mojo API — Python only)
- C ABI bridge (replaced by DLPack via Nabla Python API)

## Decisions

### Decision 1: Scene-Aware U-Net Denoiser with JEPA Conditioning (3 Tiers)

**Choice:** JEPA scene understanding system (1024-dim latent) + U-Net pixel denoiser conditioned via AdaLN modulation. NOT a latent-to-image generator — noisy pixels feed directly into U-Net, scene latent is conditioning.

**Three model tiers — same architecture, same capabilities, different size/quality/speed:**

| Tier | Params | Use Case | VRAM at 4K | U-Net Config |
|------|--------|----------|------------|-------------|
| Fast | 4M | Beat OIDN, simple scenes | ~1.2GB | C_base=48, 4 levels, bottleneck=384 |
| Medium | 16M | Kill OptiX with scene awareness | ~2.5GB | C_base=96, 5 levels, bottleneck=768 |
| High | 64M | Palace of mirrors, fog, 20K lights at 4K60 | ~4.5GB+ | C_base=192, 6 levels, bottleneck=1536 |

**All tiers support all 4 modes** (denoiser, adaptive, multires, temporal). Bigger tier = better quality, slower inference.

**Rationale:**
- LeWM's ViT-Tiny treats renders as 2D images (patch embedding) — wastes Mitsuba's 3D data
- Mitsuba gives us EXACT geometry, materials, and light positions — use them directly
- U-Net is proven for denoising (OIDN uses similar architecture) — JEPA adds scene awareness
- 3 tiers let artists trade quality vs speed based on scene complexity

**Component architecture:**

```
Component                  Params (Medium)    Implementation              Nabla Ops
────────────────────────────────────────────────────────────────────────────────
Scene Graph Encoder        ~1M                scene_graph_encoder.py      nb.nn.Embedding
  (geometry/material/light embeddings, NOT image patches)    nb.nn.Linear, F.attention
  Encodes: vertices, face normals, material params, light positions/properties
  Output: scene_latent (batch, 1024)
  Mamba usage: SSM layers for aggregating many scene graph tokens (thousands of
    vertices/faces in production scenes). O(n) scan over geometry tokens beats
    O(n²) attention when scene has 100K+ vertices.

U-Net Denoiser             ~13M               unet.py                     nb.conv2d, nb.conv2d_transpose
  (encoder-decoder with skip connections + Swin Transformer bottleneck)
  Input: cat([noisy_rgba(4), prev_clean(4), albedo(3), normal(3)]) = 14 ch
  Encoder: strided Conv2d → multi-scale features + skip connections
  Bottleneck: Swin Transformer blocks (windowed 8×8 attention) with AdaLN
    - Windowed attention at H/16×W/16 = trivial cost (~510 windows of 64 tokens)
    - Better global context for caustics/indirect lighting than Mamba (validated: MambaVision CVPR 2025)
    - Restormer-style transposed attention (channel-wise O(N)) as fallback for limited VRAM
  Decoder: Conv2dTranspose + skip concat from encoder
  Output: clean RGBA (H, W, 4) in linear HDR space
  Mamba usage: NOT at bottleneck (token count too small for O(n) to matter).
    Mamba is used in full-res encoder paths where 8.3M pixels make O(n²) impossible.

SceneDeltaEncoder          ~155K              scene_delta_encoder.py      nb.nn.Linear, nb.nn.Linear
  (Linear smoothing + MLP for per-frame scene changes)

ARPredictor                ~1.5M              arpredictor.py              nb.nn.TransformerEncoderLayer
  (ConditionalBlock layers with AdaLN-zero conditioning)
  Fast: 2 layers, 4 heads | Medium: 4 layers, 8 heads | High: 8 layers, 16 heads
  Mamba usage: Hybrid SSM+Attention for temporal sequences. At 4K 60fps 10min
    (36,000 frames), history window H=3 → sequences grow long. Mamba's O(n) scan
    over temporal tokens prevents quadratic blowup in autoregressive rollout.

ConfidenceHead             ~0.5M              confidence_head.py          nb.nn.Linear, F.sigmoid
  Linear(1024,512) → SiLU → Linear(512,256) → SiLU → Linear(256,1) → Sigmoid

SIGReg                     0                  sigreg_kernel/              Custom Mojo GPU kernel
  knots=17, num_proj=1024, Epps-Pulley statistic                via call_custom_kernel()
────────────────────────────────────────────────────────────────────────────────
TOTAL                      ~16M (Medium tier)
```

**Why this is better than ViT-Tiny:**

| Aspect | ViT-Tiny (LeWM) | Scene-Aware U-Net + JEPA (Omen) |
|--------|-----------------|---------------------------|
| Input | 2D image patches | 3D scene graph + noisy render + aux buffers |
| 3D understanding | None ( learns from pixels) | Explicit (geometry, materials, lights) |
| Parameters | 5.5M (encoder alone) | 4M/16M/64M (full model, 3 tiers) |
| Scene changes | Must re-encode entire image | Only re-encode delta |
| Generalization | Scene-specific pixels | 3D structure transfers |
| Denoising | Not designed for it | U-Net with scene-aware AdaLN conditioning |
| Temporal | No | ARPredictor with scene delta encoding |

**Scene Graph Encoder detail:**
```python
class SceneGraphEncoder(nb.nn.Module):
    """Encode Mitsuba scene data into a fixed-size latent vector.

    NOT a ViT — uses structured embeddings for known scene elements.
    """
    def __init__(self, latent_dim=1024):
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
    def __init__(self, latent_dim=1024):
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

### Decision 4: Hybrid Mamba-Swin architecture (Swin at bottleneck, Mamba where it matters)

**Choice:** Swin Transformer (windowed 8×8 attention) at the U-Net bottleneck. Mamba SSM in three areas where O(n²) attention is infeasible at production resolution.

**Rationale (from research survey, May 2026):**
- At H/16×W/16 bottleneck: 240×135 = 32,400 tokens. Windowed Swin = ~510 windows × 64 tokens = trivial cost. Mamba's O(n) advantage is negligible here.
- MambaVision (CVPR 2025) validates: Mamba in early stages (full res), Transformer in final stage (bottleneck) for global reasoning.
- MambaIRv2 (CVPR 2025) matches transformer quality but is "weaker at precise recall" — keep 1 transformer for exact detail.

**Three Mamba zones:**

```
┌─────────────────────────────────────────────────────────────────┐
│ ZONE 1: Full-resolution U-Net encoder path                      │
│ WHERE: Before downsampling, processing 8.3M pixels at 4K        │
│ WHY: Global attention O(n²) = 69 trillion ops at 3840×2160      │
│ WHAT: MambaIRv2-style Attentive SSM blocks for long-range       │
│       feature modeling at full resolution                        │
│ TIER: Medium and High only (Fast tier is too small)              │
├─────────────────────────────────────────────────────────────────┤
│ ZONE 2: Scene Graph Encoder                                      │
│ WHERE: Aggregating geometry/material/light tokens                │
│ WHY: Production scenes have 100K+ vertices, 50K+ faces,         │
│      20K lights. Attention over all tokens = O(n²) = billions.  │
│ WHAT: Mamba SSM scan over scene tokens → fixed-size latent       │
│ BENEFIT: O(n) regardless of scene complexity                    │
├─────────────────────────────────────────────────────────────────┤
│ ZONE 3: ARPredictor (temporal sequences)                         │
│ WHERE: Autoregressive rollout over frame history                 │
│ WHY: 4K 60fps 10min = 36,000 frames. History H=3 is short,     │
│      but prediction rollout can extend to hundreds of frames.    │
│ WHAT: Hybrid SSM + attention. SSM for sequence scan,             │
│      attention for precise frame-to-frame alignment.             │
│      Follows DriveMamba pattern (SSM + attention hybrid).       │
└─────────────────────────────────────────────────────────────────┘
```

**Bottleneck architecture per tier:**
```
Fast (4M):    Pure Swin Transformer, 2 blocks, window=8
              (bottleneck tiny ~16×16, Mamba adds nothing)

Medium (16M):  Swin Transformer, 4 blocks, window=8
              + Restormer transposed attention (channel-wise) for
                limited-VRAM fallback

High (64M):   Swin Transformer, 4 blocks, window=8
              + Restormer transposed attention
              + AdaLN conditioned by scene_latent (1024-dim)
              (windowed attention handles 240×135 bottleneck easily)
```

### Decision 5: Production scenarios — 4K film and multicam

**4K 60fps 10-minute render (Blender Studio / Disney / Pixar scale):**
- 3840×2160 × 60fps × 600s = 36,000 frames
- Full path trace: ~2s/frame × 36,000 = 20 hours (single GPU)
- Omen Mode 4 (temporal): ~10% frames path-traced = 3,600 at 2s + 32,400 predicted at <5ms
- Target: **36,000 frames in ~2.5 hours** (8× speedup)
- VRAM: High tier (64M) at 4K = ~4.5GB inference. Tiled processing for >4K.
- Training: Self-training on the actual production scene (LoRA fine-tune, 50 iterations)
- Key insight: The world model accumulates scene knowledge across all 36,000 frames — by frame 1000, it knows the scene better than by frame 3

**Multicam scene encoding (3+ cameras, shared world model):**
- Traditional: render 3 cameras × full path trace = 3× cost
- Omen: ONE shared scene_latent per scene, 3 cameras share the world model
- Architecture:
  ```
  Shared across all cameras:
  ┌──────────────────────────┐
  │ Scene Graph Encoder      │  ← Encodes scene ONCE (geometry, materials, lights)
  │ → scene_latent (1024)    │
  └──────────────────────────┘
            │
  ┌─────────┼─────────┐
  ▼         ▼         ▼
  Cam A    Cam B    Cam C           ← Per-camera: 1spp dirty render + camera transform
  U-Net    U-Net    U-Net           ← Same scene_latent, different noisy input
  denoise  denoise  denoise
  │         │         │
  ▼         ▼         ▼
  clean A  clean B  clean C        ← 3 clean outputs from 1 scene encoding
  ```
- Cost: 1 scene encode + 3 × 1spp render + 3 × U-Net denoise ≈ 1.3× single camera cost
- Savings: **3 cameras for 1.3× price instead of 3×** (2.3× free)
- Cross-camera consistency: Shared scene_latent ensures coherent lighting/materials across all angles
- Temporal: ARPredictor history from ALL cameras feeds into shared world model — more data = better predictions
- Surprise: If Cam A detects surprise (new object), Cam B and C immediately benefit from updated scene_latent

### Decision 6: Scene graph representation (was Decision 4)

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

### Decision 7: Self-training on Cornell box (was Decision 5)

**Choice:** Render same scene at multiple spp levels for training pairs. All training in Python using Nabla.

**Protocol:**
1. Render Cornell box at 4 spp → noisy input (via `mi.render(scene, spp=4)`)
2. Render Cornell box at 256 spp → ground truth (via `mi.render(scene, spp=256)`)
3. Extract scene features (geometry from `scene.shapes()`, materials from `shape.bsdf()`, lights from `scene.emitters()`)
4. Train in Nabla: `model.train()` → `loss.backward()` → `optimizer.step()`
5. Loss: `L_denoise + 0.1 * L_energy + 0.09 * L_sigreg` (energy conservation prevents photon creation)
6. Repeat for 1000 iterations (different camera angles, light positions)

### Decision 8: Training with Nabla PyTorch-style API (was Decision 6)

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

model = OmenJEPA(tier='medium')  # latent_dim=1024
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

    # Forward pass: U-Net denoiser conditioned by scene latent
    denoised = model.denoise(scene_features, noisy_nb, prev_clean=None)

    # Loss: denoising + energy conservation + SIGReg collapse prevention
    E_in = nb.sum(noisy_nb, axis=-1)
    E_out = nb.sum(denoised, axis=-1)
    pred_loss = nb.mean(nb.square(denoised - gt_nb))
    energy_loss = nb.mean(nb.relu(E_out - E_in - 0.01))  # no photon creation
    sigreg_loss = sigreg(model.get_embeddings())  # Custom Mojo kernel
    total_loss = pred_loss + 0.1 * energy_loss + 0.09 * sigreg_loss

    # Backward + step
    total_loss.backward()
    model = optimizer.step()

    # Checkpoint every 10 iterations
    if iteration % 10 == 0:
        save_checkpoint(model, optimizer, iteration)
```

### Decision 9: Custom Mojo kernels for SIGReg and merge (was Decision 7)

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

### Decision 10: JEPA world model for animation (simplified from LeWM) (was Decision 8)

**Choice:** Autoregressive JEPA predictor using Nabla built-in TransformerEncoderLayer. Scene deltas replace robot actions.

**ARPredictor (using Nabla built-in layers):**
```python
class ARPredictor(nb.nn.Module):
    def __init__(self, dim=1024, depth=4, heads=8, dim_head=64, mlp_dim=1024):
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

### Decision 11: Inference compilation for production (was Decision 9)

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
    │   └─> noisy_rgba [H, W, 4] + albedo [H,W,3] + normal [H,W,3]
    │
    └─> model.denoise(features, noisy, prev_clean=None)
        ├─> SceneEncoder(features)              # 3D scene → scene_latent (1024)
        ├─> Input: cat([noisy(4), zeros(4), albedo(3), normal(3)]) = 14 ch
        ├─> U-Net Encoder (strided Conv2d)      # multi-scale features + skips
        ├─> U-Net Bottleneck (Swin Transformer + AdaLN, conditioned by scene_latent)
        │   └─> Windowed 8×8 attention — trivial cost at H/16×W/16
        ├─> U-Net Decoder (Conv2dTranspose + skip concat)
        └─> output: clean RGBA (H, W, 4) linear HDR → numpy
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
│   ├── jepa.py                    # Top-level OmenJEPA model (compose all components)
│   ├── scene_encoder.py           # SceneGraphEncoder (~0.3-1M params depending on tier)
│   ├── unet.py                    # UNetDenoiser (3-55M params depending on tier)
│   ├── arpredictor.py             # ARPredictor + ConditionalBlock (0.5-6M depending on tier)
│   ├── decoder.py                 # Conv2dTranspose decoder (used by UNet internally)
│   ├── sigreg.py                  # SIGReg loss (custom kernel wrapper)
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
│   ├── extractor.py               # extract_scene_features(mi.Scene) → dict (Mitsuba today, portable)
│   └── delta.py                   # compute_delta(frame_A, frame_B) → SceneDelta
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                 # Training loop (Nabla AdamW + energy loss + checkpoints)
│   ├── data_gen.py                # Training data generation (Dr.Jit renders)
│   └── cornell_schedule.py        # 4-phase Cornell box training schedule
│
├── modes/
│   ├── __init__.py
│   ├── denoiser.py                # Mode 1: 4spp → U-Net denoise → clean
│   ├── adaptive.py                # Mode 2: preview + confidence + high-spp → merge
│   ├── multires.py                # Mode 3: low-res clean + high-res noisy → merge
│   └── animation.py               # Mode 4: temporal prediction + surprise detection
│
├── jepa_bridge.py                 # Load model, DLPack transfer, inference wrapper
├── checkpoint.py                  # Save/load state_dict, LoRA adapters
├── config.py                      # Tier configs (Fast/Medium/High), hyperparameters
└── inference.py                   # @nb.compile inference functions
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
    energy_violation = nb.relu(nb.sum(predicted, axis=-1) - nb.sum(noisy_nb, axis=-1) - 0.01)
    energy_loss = nb.mean(energy_violation)
    total = pred_loss + 0.1 * energy_loss + 0.09 * sigreg_loss

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
├── base_fast_v0.omen               # Pre-trained Fast tier (~16MB)
├── base_medium_v0.omen             # Pre-trained Medium tier (~64MB)
├── base_high_v0.omen               # Pre-trained High tier (~256MB)
├── *.meta.json                     # Architecture hash, version, metrics
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
| U-Net VRAM at 4K | Fast tier fits 1.2GB; High tier needs 8GB+ for 4K; tiled processing |
| Mamba SSM weaker at precise recall | Keep Swin Transformer at bottleneck for exact detail |
| Swin windowed attention misses cross-window features | Window size 8×8 is proven; shift windows between layers (SwinIR pattern) |
| Multicam scene changes invalidate shared latent | Surprise detection triggers re-encode; all cameras share updated latent |

## Implementation Phases

### Phase 1: Scene Extraction + Model Skeleton (Week 1-2)
- [ ] `scene/extractor.py` from Mitsuba API
- [ ] `model/scene_encoder.py` (SceneGraphEncoder with Mamba SSM for scene tokens)
- [ ] `model/render_encoder.py` (RenderFeatureEncoder, Conv2d)
- [ ] `model/fusion.py` (Cross-attention)
- [ ] Test: extract Cornell box features, encode to latent

### Phase 2: Core Model + Training (Week 3-5)
- [ ] `model/unet.py` (U-Net with Swin Transformer bottleneck + Mamba encoder blocks)
- [ ] `model/arpredictor.py` (ConditionalBlock + AdaLN-zero, hybrid SSM+Attention)
- [ ] `model/decoder.py` (Conv2dTranspose upsample)
- [ ] `kernels/sigreg_kernel.mojo` + `sigreg_op.py`
- [ ] `training/trainer.py` (Nabla training loop)
- [ ] Test: train Phase 1 on Cornell box, verify SSIM > 0.95

### Phase 3: Modes + Multicam (Week 6-8)
- [ ] `modes/denoiser.py` (Mode 1)
- [ ] `modes/adaptive.py` (Mode 2 + ConfidenceHead)
- [ ] `modes/multires.py` (Mode 3 + merge kernel)
- [ ] `modes/multicam.py` (shared scene_latent, multiple cameras)
- [ ] `checkpoint.py` (save/load state_dict)
- [ ] Test: all modes on Cornell box, test multicam with 3 angles

### Phase 4: Temporal + Animation (Week 9-12)
- [ ] `model/scene_delta_encoder.py`
- [ ] `modes/animation.py` (surprise detection, history buffer)
- [ ] `training/cornell_schedule.py` (4-phase training)
- [ ] `inference.py` (`@nb.compile` inference)
- [ ] Test: 100-frame animation with surprise detection
