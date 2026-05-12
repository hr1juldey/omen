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
  (encoder-decoder with MLA-compressed skip connections + Swin+MoE bottleneck)
  Input: cat([noisy_rgba(4), prev_clean(4), albedo(3), normal(3)]) = 14 ch
  Encoder: strided Conv2d → multi-scale features + skip connections
  Skip: MLA-compressed (16× reduction, ~6GB → ~375MB at 4K)
  Bottleneck: Swin Transformer blocks (windowed 8×8 attention) + MoE FFN
    - Swin attention: global context (~510 windows of 64 tokens, trivial cost)
    - MoE FFN: tile-based expert routing (8×8 windows with cryptomatte-style masks)
    - Shared expert (always active) + routed material/light/geo experts
    - Auxiliary-loss-free load balancing (DeepSeek-V3 bias adjustment)
    - Restormer-style transposed attention (channel-wise O(N)) as fallback for limited VRAM
  Decoder: Conv2dTranspose + MLA-reconstructed skip concat from encoder
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

### Decision 6: MLA-inspired skip connection compression (from DeepSeek-V2/V3)

**Choice:** Compress U-Net skip connections using MLA-style low-rank projections. Reduces 4K skip memory from ~6GB to ~375MB. Compresses temporal history for long rollouts using DeepSeek-V4 CSA/HCA-style tiered compression.

**Rationale:**
At 4K (3840×2160), skip connections dominate VRAM. DeepSeek-V2's MLA proved that projecting features into a low-rank latent space reduces cache by 93.3% with negligible quality loss. The same principle applies to U-Net skips:

```
Standard skip:  encoder(H,W,C) → store → decoder reads (H,W,C)
MLA skip:       encoder(H,W,C) → W_down → latent(H,W,d_c) → store
                decoder reads latent → W_up → reconstructed(H,W,C)
                where d_c = C/16  →  16× compression
```

**Compression targets per level (High tier):**

| Level | Resolution | Full C | Compressed d_c | Full | Compressed |
|-------|-----------|--------|----------------|------|------------|
| 0 | 3840×2160 | 192 | 12 | 3.17GB | ~200MB |
| 1 | 1920×1080 | 384 | 24 | 1.59GB | ~100MB |
| 2 | 960×540 | 768 | 48 | 0.80GB | ~50MB |
| 3 | 480×270 | 1536 | 96 | 0.40GB | ~25MB |
| **Total** | | | | **~6GB** | **~375MB** |

**Implementation:**
```python
class MLASkipConnection(nb.nn.Module):
    """MLA-inspired low-rank skip connection (from DeepSeek-V2)."""
    def __init__(self, channels, ratio=16):
        d_c = channels // ratio
        self.compress = nb.nn.Linear(channels, d_c)     # W_down
        self.reconstruct = nb.nn.Linear(d_c, channels)  # W_up

    def encode(self, encoder_feature):
        return self.compress(encoder_feature)    # (H, W, C/ratio) — store this

    def decode(self, compressed):
        return self.reconstruct(compressed)       # (H, W, C) — reconstruct at decoder
```

**DeepSeek-V4 CSA/HCA for temporal history compression:**
ARPredictor maintains H=3 frame latents. For 4K 60fps 10min (36,000 frames), DeepSeek-V4's tiered compression applies:
- **Recent frames** (last 4): full-resolution latents — CSA c4a style (4× compression)
- **Older frames** (5-128): heavily compressed — HCA c128a style (128× compression)
- At 1M-token equivalent (V4 achieves 2% KV cache), Omen can maintain long temporal context cheaply

### Decision 7: Material/Light/Geometry-aware MoE — Tile-based Routing with Cryptomatte Masks

**Choice:** MoE with experts specialized per material type, light type, and geometry type — NOT per scene type. Routing is TILE-BASED (8×8 Swin windows) using cryptomatte-style material/light/geo masks. A single pixel has no meaning — it needs spatial context. The same 8×8 window structure used by Swin attention is reused for MoE routing.

**Rationale:**
DeepSeekMoE proved fine-grained experts + shared expert isolation beats coarse experts. But two routing levels are WRONG:

1. **Per-scene routing** (interior, exterior, product) is wrong because scenes are always mixed:
   - Product shot = glass on metal table with diffuse walls → 3 material types in ONE frame
   - Interior = emissive screens + wood + chrome fixtures → mixed everywhere

2. **Per-pixel routing** is also wrong because 1 pixel has no meaning:
   - A pixel at (127,45) with material_id=2 tells you nothing — is it edge of metal? center? reflection?
   - No spatial context → expert sees a single value, not a pattern
   - Real denoising needs to see edges, gradients, transitions WITHIN a neighborhood
   - Production renderers already compute cryptomatte/object ID passes per-region

**Correct: tile-based routing.** The 8×8 Swin window already groups 64 pixels. Compute a tile fingerprint (material histogram + normal variance + depth edge density) and route the ENTIRE TILE. Expert sees spatial structure, not a meaningless dot.

**Expert taxonomy:**
```
MATERIAL EXPERTS (routed by tile material histogram from cryptomatte):
  Expert 0: Diffuse/Lambertian      — walls, floors, flat surfaces
  Expert 1: Glossy/Glass            — reflections, refraction, caustics
  Expert 2: Metal/Chrome            — conductor BSDF, mirrors, specular
  Expert 3: SSS/Skin                — subsurface scattering, wax, marble
  Expert 4: Volume/Smoke            — participating media, fog, fire
  Expert 5: Emissive                — area lights, LED panels, neon
  Expert 6: Hair/Fur                — curve primitives, anisotropic
  Expert 7: Cloth/Fabric            — microfiber, woven patterns

LIGHT EXPERTS (routed by tile light contribution histogram):
  Expert 0: Point/Spot light        — local, hard falloff
  Expert 1: Area light              — soft shadows, rectangular/disk
  Expert 2: Sun/Directional         — hard shadows, sky illumination
  Expert 3: Environment/HDRI        — ambient, indirect dome
  Expert 4: Emissive geometry       — mesh lights, emissive surfaces

GEOMETRY EXPERTS (routed by tile normal/depth statistics):
  Expert 0: Flat surfaces           — low normal variance, easy denoise
  Expert 1: Curved/organic          — smooth normal changes, SSS-friendly
  Expert 2: Edges/silhouettes       — high normal discontinuity, aliasing
  Expert 3: Fine detail/hair        — sub-pixel geometry, anisotropic noise
  Expert 4: Transparent/overlapping — depth discontinuity, refraction

SHARED EXPERT (always active — from DeepSeekMoE shared expert isolation):
  Base denoising — Gaussian noise removal, spatial filtering, universal patterns

MOTION EXPERTS (routed by tile velocity statistics — see Decision 14):
  Expert 0: Static            — low velocity variance, high temporal reuse
  Expert 1: Linear motion     — uniform velocity, warp + accumulate
  Expert 2: Fast motion/blur  — high velocity, shutter smear, deblur
  Expert 3: Occlusion boundary — velocity discontinuity, inpainting-style
```

**Tile-based routing with cryptomatte masks:**
```python
class TileMoERouter(nb.nn.Module):
    """Tile-based MoE routing using 8×8 window fingerprints.

    Routes ENTIRE 8×8 tiles (not individual pixels) to experts.
    A tile fingerprint = histogram of material/light/geo signals within the window.
    This preserves spatial context — experts see edges, gradients, transitions.
    """
    def __init__(self, n_material=8, n_light=5, n_geo=5, top_k=2, window_size=8):
        self.top_k = top_k
        self.window_size = window_size
        # Tile fingerprint dim: material_hist(8) + light_hist(5) + geo_stats(5) = 18
        fingerprint_dim = n_material + n_light + 5  # 5 = normal_var + depth_var + edge_density + mean_albedo + dominant_mat
        self.mat_route = nb.nn.Linear(fingerprint_dim, n_material)
        self.light_route = nb.nn.Linear(fingerprint_dim, n_light)
        self.geo_route = nb.nn.Linear(fingerprint_dim, n_geo)
        # Auxiliary-loss-free bias (DeepSeek-V3)
        self.mat_bias = nb.zeros(n_material)
        self.light_bias = nb.zeros(n_light)
        self.geo_bias = nb.zeros(n_geo)

    def compute_tile_fingerprint(self, aux_windows):
        """Compute routing fingerprint for each 8×8 tile.

        Args:
            aux_windows: (B, n_tiles, 64, C) — C = albedo(3) + normal(3) + depth(1) + material_id(1)
                         reshaped from (B, H, W, 8) into Swin windows

        Returns:
            fingerprint: (B, n_tiles, fingerprint_dim) — one routing vector per tile
        """
        material_ids = aux_windows[:, :, :, 7]           # (B, n_tiles, 64)
        # Cryptomatte-style histogram: count pixels per material type in this tile
        mat_hist = histogram(material_ids, bins=8) / 64.0  # (B, n_tiles, 8)
        # Normal variance within tile = edge/curvature indicator
        normals = aux_windows[:, :, :, 3:6]               # (B, n_tiles, 64, 3)
        normal_var = nb.var(normals, axis=2)              # (B, n_tiles, 3)
        # Depth variance = transparency/overlap indicator
        depth = aux_windows[:, :, :, 6]                   # (B, n_tiles, 64)
        depth_var = nb.var(depth, axis=2, keepdims=True)  # (B, n_tiles, 1)
        # Edge density: fraction of pixels where normal discontinuity > threshold
        edge_density = compute_edge_density(normals)       # (B, n_tiles, 1)
        # Dominant material and mean albedo
        dominant_mat = nb.argmax(mat_hist, axis=-1, keepdims=True)  # (B, n_tiles, 1)
        mean_albedo = nb.mean(aux_windows[:, :, :, :3], axis=(2,)) # (B, n_tiles, 3)
        return nb.concat([mat_hist, normal_var, depth_var, edge_density, dominant_mat, mean_albedo], axis=-1)

    def forward(self, aux_windows):
        # aux_windows: (B, n_tiles, 64, C) — reshaped from (B, H, W, 8) into Swin windows
        fp = self.compute_tile_fingerprint(aux_windows)    # (B, n_tiles, fingerprint_dim)
        mat_scores = self.mat_route(fp) + self.mat_bias
        light_scores = self.light_route(fp) + self.light_bias
        geo_scores = self.geo_route(fp) + self.geo_bias
        # Top-K per category — route ENTIRE TILE (all 64 tokens) to selected experts
        return route_topk(mat_scores, self.top_k), route_topk(light_scores, 1), route_topk(geo_scores, 1)
```

**Architecture integration (bottleneck MoE FFN replaces standard MLP):**
```
U-Net Bottleneck per Swin window:
┌───────────────────────────────────────────────────────────────┐
│  Swin Windowed Attention (8×8 = 64 tokens)                    │  ← spatial context
│  + AdaLN conditioned by scene_latent                          │
├───────────────────────────────────────────────────────────────┤
│  Tile Fingerprint Computation:                                 │
│  From 64 tokens, compute:                                     │
│    material histogram (cryptomatte-style)                     │
│    + normal variance (edge detector)                          │
│    + depth variance (transparency indicator)                  │
│    → one routing vector per 8×8 tile                          │
├───────────────────────────────────────────────────────────────┤
│  MoE FFN (replaces standard MLP):                             │
│  ALL 64 tokens in tile routed together:                       │
│                                                                │
│  Tile (glass-dominant)   → Material Expert 1 + Shared Expert  │
│  Tile (skin-dominant)    → Material Expert 3 + Shared Expert  │
│  Tile (mixed glass/metal)→ Expert 1 + Expert 2 + Shared       │
│  Tile (edge-heavy)       → Geo Expert 2 + Shared Expert       │
│  ...per-TILE, not per-pixel, preserving spatial structure      │
└───────────────────────────────────────────────────────────────┘
```

**Why tile-based beats per-pixel:**
```
PER-PIXEL (wrong):
  Pixel (127,45) material_id=2 → "metal expert"
  Pixel (127,46) material_id=2 → "metal expert"
  Pixel (127,47) material_id=1 → "glass expert" ← switched mid-edge!
  → No context. Expert can't see the edge between metal and glass.
  → Adjacent pixels may route to different experts → seam artifacts.

TILE-BASED (correct):
  8×8 tile around (127,45):
    material histogram = {metal: 45px, glass: 19px}
    normal variance = high (edge detected)
    → Route to Metal Expert + Glass Expert + Edge Geo Expert
    → Expert sees the FULL transition zone with spatial context
    → Smooth blending across material boundaries
```

**Expert config per tier:**

| Tier | Material | Light | Geo | Motion | Shared | Top-K | MoE? |
|------|----------|-------|-----|--------|--------|-------|------|
| Fast (4M) | — | — | — | — | 1 | — | No MoE (too small) |
| Medium (16M) | 8 | 5 | 5 | 4 | 1 | 2 | MoE in bottleneck only |
| High (64M) | 8 | 5 | 5 | 4 | 1 | 3 | MoE in bottleneck + encoder |

**Auxiliary-loss-free load balancing (DeepSeek-V3):**
Per-expert bias adjusted dynamically — zero gradient interference with denoising quality:
```
if expert overloaded:  bias[expert] -= 0.001   (discourage)
if expert underloaded: bias[expert] += 0.001   (encourage)
Updated every training step, does NOT participate in backward pass
```

### Decision 8: Scene graph representation (was Decision 4, previously Decision 6)

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

### Decision 9: Self-training on Cornell box (was Decision 5, previously Decision 7)

**Choice:** Render same scene at multiple spp levels for training pairs. All training in Python using Nabla.

**Protocol:**
1. Render Cornell box at 4 spp → noisy input (via `mi.render(scene, spp=4)`)
2. Render Cornell box at 256 spp → ground truth (via `mi.render(scene, spp=256)`)
3. Extract scene features (geometry from `scene.shapes()`, materials from `shape.bsdf()`, lights from `scene.emitters()`)
4. Train in Nabla: `model.train()` → `loss.backward()` → `optimizer.step()`
5. Loss: `L_denoise + 0.1 * L_energy + 0.09 * L_sigreg` (energy conservation prevents photon creation)
6. Repeat for 1000 iterations (different camera angles, light positions)

### Decision 10: Training with Nabla PyTorch-style API (was Decision 6, previously Decision 8)

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

### Decision 11: Custom Mojo kernels for SIGReg and merge (was Decision 7, previously Decision 9)

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

### Decision 12: JEPA world model for animation (simplified from LeWM) (was Decision 8, previously Decision 10)

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

### Decision 13: Inference compilation for production (was Decision 9, previously Decision 11)

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

### Decision 14: Motion Blur & Temporal Reprojection

**Choice:** Handle motion blur via motion vector AOV + previous-frame reprojection + motion-aware MoE experts. Motion blur is NOT just denoising — pixels at different shutter times mix materials, making standard auxiliary buffers ambiguous.

**The problem:**
```
Motion blur = samples at different TIME instants averaged together

Frame with object moving left to right:
  Pixel (100, 200) at t=0.0 → material_id = metal
  Pixel (100, 200) at t=0.5 → material_id = glass (object moved)
  Pixel (100, 200) at t=1.0 → material_id = background

  Rendered pixel = average of all three → blurry mix
  Auxiliary buffer captures ONE instant → material_id says "glass"
  But pixel is 33% metal + 33% glass + 33% bg
  → Tile fingerprint is WRONG → MoE routes to wrong experts
```

**Solution: Motion vector pass + temporal reprojection**

Motion vectors (2D screen-space velocity) are a standard AOV in both Blender and Mitsuba:
- Blender: `scene.render.use_pass_vector = True`
- Mitsuba: AOV integrator with motion time samples

**Temporal reprojection pipeline (how DLSS/FSR do it):**
```
Frame N-1 clean output  +  Frame N motion vectors
         │                          │
         └──── warp(prev_clean, motion_vectors) ────┐
                                                      ▼
                              reprojected_prev (aligned to frame N)
                                      │
                                      ▼
                  merge: α * reprojected_prev + (1-α) * current_noisy
                  where α = confidence(prev_frame) × motion_coherence
```

**Motion-aware MoE experts (4th routing dimension):**
```
MOTION EXPERTS (routed by tile velocity statistics):
  Expert 0: Static (low velocity variance) → standard denoise, high temporal reuse
  Expert 1: Linear motion (uniform velocity, object moving) → warp + accumulate
  Expert 2: Fast motion / blur (high velocity, shutter smear) → deblur expert
  Expert 3: Occlusion boundary (velocity discontinuity) → inpainting-style, low temporal reuse

Added to tile fingerprint:
  velocity_mean: mean(motion_vectors) in tile → 2-dim
  velocity_var: variance(motion_vectors) in tile → 2-dim
  velocity_max: max(||motion_vector||) in tile → 1-dim
  occlusion_fraction: pixels where velocity discontinuity > threshold → 1-dim
```

**Updated tile fingerprint (17 → 23 dim):**
```
Original:  [mat_hist(8) + normal_var(3) + depth_var(1) + edge_density(1) + dominant_mat(1) + mean_albedo(3)] = 17
Add motion: [velocity_mean(2) + velocity_var(2) + velocity_max(1) + occlusion_frac(1)] = 6
Total:     23-dim tile fingerprint
```

**Shutter-aware auxiliary buffers:**
When motion blur is enabled, capture auxiliary buffers at multiple time samples:
- albedo_t0, albedo_t1, albedo_t2 (start, mid, end of shutter)
- Tile fingerprint uses VARIANCE across time samples
- High temporal variance = motion blur zone → route to deblur expert

**Graceful degradation (no motion vectors):**
- Fill motion vector channels with zeros
- Motion experts never activated → static-only mode (Expert 0 handles everything)
- No temporal reprojection → single-frame denoise (current behavior)
- Log: "Motion vectors unavailable — static denoise mode (no temporal reprojection)"

**Updated MoE expert config per tier:**

| Tier | Material | Light | Geo | Motion | Shared | Top-K | MoE? |
|------|----------|-------|-----|--------|--------|-------|------|
| Fast (4M) | — | — | — | — | 1 | — | No MoE |
| Medium (16M) | 8 | 5 | 5 | 4 | 1 | 2 | MoE bottleneck |
| High (64M) | 8 | 5 | 5 | 4 | 1 | 3 | MoE bottleneck+encoder |

### Decision 15: Performance Optimizations

**Choice:** Six additional optimizations beyond the base architecture to maximize throughput. These are independent — each can be enabled incrementally.

**15a: Async pipeline (DualPipe from DeepSeek-V3)**
```
Sequential (current):   Mitsuba render → JEPA denoise → Mitsuba render → JEPA denoise
                       Frame N:  [======render======][======denoise======]
                       Frame N+1:                                        [======render======][======denoise======]

Async (DualPipe):       Frame N:  [======render======][======denoise======]
                       Frame N+1:          [======render======][======denoise======]
                                         ^ overlaps with N's denoise

Implementation:
- Thread 1: Mitsuba render loop (produces noisy frames)
- Thread 2: JEPA denoise loop (consumes noisy frames)
- Bounded queue (size 2) between them
- ~1.8× throughput for animation sequences
- Requires double-buffering GPU memory (2× VRAM for ping-pong)
```

**15b: Speculative multi-frame prediction (MTP from DeepSeek-V3)**
```
Current ARPredictor: predict frame N+1 only
MTP: predict N+1, N+2, N+3 simultaneously (shared trunk, separate heads)

Implementation:
  latent_N = encode(frame_N, scene_graph)
  frame_N1 = decode(predict_next(latent_N, delta_N1))     # predicted frame N+1
  frame_N2 = decode(predict_next(latent_N, delta_N2))     # predicted frame N+2
  frame_N3 = decode(predict_next(latent_N, delta_N3))     # predicted frame N+3

  Verify N+1 with cheap 1spp render:
    if SSIM(predicted_N1, render_1spp_N1) > 0.85: use prediction (skip render)
    else: fall back to normal render for N+2, N+3

  Speedup: 1.8× on animation (3 frames for price of ~1.7)
  Only active in Mode 4 (animation) with history buffer populated
```

**15c: Scene latent caching with smart invalidation**
```
Problem: Re-encoding scene graph every frame is wasteful if scene structure hasn't changed.
But: animated geometry changes vertex positions, lights change intensity, materials animate.

Smart cache — two-level hashing:
  Level 1: topology_hash (face connectivity + material TYPES + light TYPES)
    → Same across animation if no objects added/removed
    → Cheap to compute (integer comparison)
  Level 2: dynamic_hash (vertex positions + light intensities + material VALUES)
    → Changes every frame with animation
    → More expensive but still fast (hash of float arrays)

Cache strategy:
  Frame 0: full encode → cache (scene_latent, topology_hash, dynamic_hash)
  Frame N>0:
    compute topology_hash
    if topology_hash != cached:
      → Scene structure changed. Full re-encode. Update cache.
    elif topology_hash == cached:
      → Structure same, check dynamic changes
      compute delta = scene_delta(frame_N-1, frame_N)
      if delta.is_small():  # no births, no structural changes
        → cached_latent += SceneDeltaEncoder(delta)  # incremental update (~5ms)
        → Update dynamic_hash in cache
      else:
        → Large dynamic change. Full re-encode. Update cache.

What counts as "large dynamic change":
  - New object appeared (birth event)
  - Light added/removed
  - Material type changed (diffuse → glass)
  - Vertex count changed (subdivision level changed)

What's "small" (incremental update OK):
  - Object moved (vertex positions changed, same topology)
  - Light intensity/color changed
  - Material parameter values changed (roughness, color)
  - Camera moved

Savings: ~30ms per frame on static scenes, ~25ms on animated scenes (delta encode vs full)
```

**15d: Progressive adaptive (more aggressive than current Mode 2)**
```
Current Mode 2: PASS 1 (4spp everywhere) → PASS 2 (128spp everywhere) → merge
Progressive:    PASS 1 (2spp everywhere) → confidence → only add samples WHERE needed

Implementation:
  preview = mi.render(scene, spp=2)  # even cheaper preview
  clean_preview, confidence = model.predict_confidence(features, preview)

  # Build per-tile spp map based on confidence
  for tile in tiles:
    if confidence_tile > 0.8:    spp_map[tile] = 0    # done, use JEPA prediction
    elif confidence_tile > 0.5:  spp_map[tile] = 16   # moderate, add some samples
    elif confidence_tile > 0.2:  spp_map[tile] = 64   # low confidence, add more
    else:                        spp_map[tile] = 128  # very low, full path trace

  # Render with variable spp (Mitsuba doesn't support per-tile spp directly,
  # so render at max_spp and mask — or render multiple passes at different spp
  # and composite based on confidence regions)

Target: 10-16× sample reduction (vs current 4-8×) on scenes with >50% easy regions
```

**15e: Early exit in U-Net decoder**
```
U-Net decoder has multiple levels (4-6 depending on tier).
For high-confidence tiles (flat diffuse, no edges), skip deeper decoder levels:

  Level 0 output confidence per tile from bottleneck
  if tile_confidence > 0.9 AND tile is flat (low normal variance):
    → Use Level 0 decoder output directly (skip levels 1-3)
    → Saves ~40% of decoder compute on easy regions

  Implementation: decoder returns early for "done" tiles, continues for complex tiles
  Requires per-tile processing (not batched) — may hurt GPU utilization
  → Only enable for High tier at 4K where decoder is the bottleneck
```

**15f: FP8 mixed precision inference (from DeepSeek-V3)**
```
Current: BF16 for all tensors
FP8:     E4M3 for forward matmuls, per-tile dynamic scaling

  U-Net encoder Conv2d:   FP8 weights, BF16 accumulation
  Swin attention QKV:     BF16 (softmax needs precision)
  MoE expert FFN:         FP8 weights (experts are small, FP8 saves VRAM)
  U-Net decoder Conv2d:   FP8 weights, BF16 accumulation

  VRAM savings: 700MB → ~400MB at inference (MLA already compressed, FP8 halves rest)
  Speed: 1.5-2× faster matmuls on Ada Lovelace / Hopper GPUs
  Quality: validate PSNR drop < 0.5dB vs BF16 baseline

  Implementation: Nabla mixed precision via `nb.nn.Linear(..., dtype=nb.float8_e4m3fn)`
  Fallback: if GPU doesn't support FP8 (pre-Ada), use BF16 automatically
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
    │   └─> noisy_rgba [H, W, 4] + albedo [H,W,3] + normal [H,W,3] + motion_vectors [H,W,2]
    │
    └─> model.denoise(features, noisy, prev_clean=None)
        ├─> SceneEncoder(features)              # 3D scene → scene_latent (1024)
        ├─> Input: cat([noisy(4), zeros(4), albedo(3), normal(3)]) = 14 ch
        ├─> U-Net Encoder (strided Conv2d)      # multi-scale features
        ├─> MLA Skip Compression (16×)          # ~6GB → ~375MB at 4K
        ├─> U-Net Bottleneck (Swin Transformer + MoE FFN + AdaLN)
        │   ├─> Windowed 8×8 attention — global context
        │   └─> MoE FFN — tile-based material/light/geo expert routing (8×8 cryptomatte masks)
        ├─> MLA Skip Reconstruction (at decoder)
        ├─> U-Net Decoder (Conv2dTranspose + reconstructed skips)
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
│   ├── mla_skip.py                # MLASkipConnection — low-rank skip compression (DeepSeek-V2)
│   ├── moe.py                     # TileMoERouter (8×8 tile fingerprint + cryptomatte masks) + expert FFNs + motion experts
│   ├── arpredictor.py             # ARPredictor + ConditionalBlock (0.5-6M depending on tier)
│   ├── decoder.py                 # Conv2dTranspose decoder (used by UNet internally)
│   ├── sigreg.py                  # SIGReg loss (custom kernel wrapper)
│   └── layers.py                  # AdaLNModulation, modulate(), FeedForward helpers
│
├── kernels/                       # Custom Mojo GPU kernels
│   ├── __init__.mojo              # Empty init
│   ├── sigreg_kernel.mojo         # SIGReg Epps-Pulley statistic
│   ├── merge_kernel.mojo          # Edge-aware multires merge
│   ├── tile_fingerprint.mojo      # GPU 8×8 tile histogram + variance + velocity stats for MoE routing
│   └── sigreg_op.py               # Python wrapper: UnaryOperation subclass
│
├── scene/
│   ├── __init__.py
│   ├── extractor.py               # extract_scene_features(mi.Scene) → dict (Mitsuba today, portable)
│   ├── delta.py                   # compute_delta(frame_A, frame_B) → SceneDelta
│   └── latent_cache.py            # Two-level scene latent cache (topology_hash + dynamic_hash)
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
├── inference.py                   # @nb.compile inference functions
├── aov.py                         # AOV pass reader + graceful degradation for missing passes
└── motion.py                      # Motion vector processing + temporal reprojection
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
| MLA skip compression loses fine detail | Compression ratio 16× is aggressive; validate with PSNR on edges/caustics; reduce ratio if needed |
| MoE tile routing misclassifies region | Tile fingerprint (cryptomatte histogram) relies on auxiliary buffers; validate with ground-truth material ID maps |
| MoE load imbalance (some experts starved) | Auxiliary-loss-free bias adjustment from DeepSeek-V3; monitor expert utilization during training |
| MLA reconstruction introduces artifacts at edges | Add edge-aware loss term; apply MLA only on smooth regions, keep full-res at discontinuities |
| Motion blur makes auxiliary buffers ambiguous | Shutter-aware multi-time sampling + motion vector AOV + motion-aware MoE experts; fallback to static mode if no motion vectors |
| Temporal reprojection ghosting on fast motion | Clamp α by motion coherence; occlusion detection via velocity discontinuity; fall back to single-frame at boundaries |
| Scene latent cache stale after animation changes | Two-level hashing: topology hash (structure) + dynamic hash (values); invalidate on births, material type changes, vertex count changes |
| FP8 precision loss on HDR values | Per-tile dynamic scaling; validate PSNR drop < 0.5dB vs BF16; fallback to BF16 on pre-Ada GPUs |
| Async pipeline double-buffering doubles VRAM | Only enable async when VRAM > 2× inference budget; fall back to sequential otherwise |

## Implementation Phases

### Phase 1: Scene Extraction + Model Skeleton (Week 1-2)
- [ ] `scene/extractor.py` from Mitsuba API
- [ ] `model/scene_encoder.py` (SceneGraphEncoder with Mamba SSM for scene tokens)
- [ ] `model/render_encoder.py` (RenderFeatureEncoder, Conv2d)
- [ ] `model/fusion.py` (Cross-attention)
- [ ] Test: extract Cornell box features, encode to latent

### Phase 2: Core Model + Training (Week 3-5)
- [ ] `model/unet.py` (U-Net with Swin Transformer bottleneck + Mamba encoder blocks)
- [ ] `model/mla_skip.py` (MLA low-rank skip compression — 16× reduction)
- [ ] `model/moe.py` (TileMoERouter + tile fingerprint computation + per-type expert FFNs + shared expert)
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
