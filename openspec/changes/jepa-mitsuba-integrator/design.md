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
- Real-time viewport rendering
- Material node compilation (use Mitsuba's BSDFs)
- Temporal reuse (future work)

## Decisions

### Decision 1: JEPA architecture in Mojo

**Choice:** Scene-conditioned transformer with cross-attention

**Rationale:**
- Scene graph (structured data) → transformer encoder
- Image patches (spatial data) → CNN encoder
- Cross-attention: image queries scene context
- JEPA objective: predict latent patches from context

**Architecture (Mojo):**
```mojo
struct JEPAEncoder:
    var scene_encoder: TransformerEncoder     # Encodes geometry/materials/lights
    var image_encoder: ConvEncoder            # Encodes image patches
    var cross_attention: CrossAttention         # Image queries scene

struct JEPAModel:
    var encoder: JEPAEncoder
    var decoder: ConvDecoder
    var confidence_head: MLPLayer              # Mode 2 only
```

**Why this over 2D U-Net denoiser:**
- Scene conditioning enables "this is glass, needs more samples"
- Transformer handles variable-length scene (any number of meshes/lights)
- JEPA predicts in latent space (more stable than pixel prediction)

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

### Decision 5: Mojo C ABI interface

**Choice:** Register functions with `@register_function`, wrap pointers in structs

**Rationale:**
- Stable across Mojo versions
- Compatible with ctypes (C ABI standard)
- Zero-copy buffer passing with `UnsafePointer` and `owning=False`

**Function signatures:**
```mojo
@register_function
def omen_denoise(
    scene: SceneGraph,
    obs: RenderObservation,
    output_rgba: UnsafePointer[C_float],
    gpu_device_id: Int,
) -> Int:
    # Inference here
    return 0  # Success
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

### Mojo Side (`jepa_kernels/`)

```
C_ABI.mojo:
  @register_function def omen_denoise(...)
  @register_function def omen_predict_confidence(...)
  @register_function def omen_merge_multires(...)

scene_encoder.mojo:
  struct SceneGraph [CPU copy from Python]
  struct SceneEncoder:
    encode_geometry() -> Tensor
    encode_materials() -> Tensor
    encode_lights() -> Tensor
    encode_camera() -> Tensor

image_encoder.mojo:
  struct ImageEncoder:
    encode_patches() -> Tensor  # Strided convolutions

jepa.mojo:
  struct JEPAModel:
    var scene_encoder: SceneEncoder
    var image_encoder: ImageEncoder
    var cross_attn: CrossAttention
    var decoder: ImageDecoder

    def denoise(scene, noisy) -> clean
    def predict_confidence(scene, noisy) -> (clean, confidence)
    def merge_multires(scene, low_res, high_res) -> merged

confidence.mojo:
  struct ConfidenceHead:
    def predict(latent) -> confidence [0,1]

multires.mojo:
  struct MultiResMerge:
    def merge(low_res, high_res, scene) -> merged
```

## Training Protocol

### Self-Training Data Generation

```python
# Generate supervised training pairs
def generate_training_pair(scene, seed):
    # Low-spp (noisy)
    mi.set_seed(seed)
    noisy = mi.render(scene, spp=4)

    # High-spp (ground truth)
    mi.set_seed(seed)
    gt = mi.render(scene, spp=256)

    return noisy, gt
```

### Cornell Box Training Schedule

**Phase 1: Bootstrap (frames 1-100)**
- Render Cornell box at 4spp + 256spp
- Train JEPA denoiser: L1 loss vs ground truth
- Target: SSIM > 0.95 vs 256spp

**Phase 2: Confidence head (frames 101-200)**
- Render Cornell box 8× at 4spp → variance map
- Train confidence head: MSE vs variance
- Target: predict uncertainty (high variance = low confidence)

**Phase 3: Multi-res merge (frames 201-300)**
- Render at 25% res 256spp + 100% res 4spp
- Render at 100% res 256spp (ground truth)
- Train merge model: L1 loss vs ground truth
- Target: PSNR > 30dB vs ground truth

## File Structure

```
jepa_kernels/
├── C_ABI.mojo              # C interface, SceneGraph/RenderObservation structs
├── scene_encoder.mojo       # Transformer over scene tokens
├── image_encoder.mojo       # CNN encoder for image patches
├── jepa.mojo                # Main JEPA model (cross-attention, decoder)
├── confidence.mojo          # Confidence head (Mode 2)
├── multires.mojo            # Multi-resolution merge (Mode 3)
└── build.sh                 # Compile to libomen.so

src/omen_integrator/
├── __init__.py              # Updated: mode parameter
├── scene_extractor.py       # NEW: Mitsuba scene extraction
├── jepa_bridge.py           # NEW: Ctypes bridge to Mojo
├── modes/
│   ├── __init__.py
│   ├── denoiser.py          # NEW: Mode 1 orchestration
│   ├── adaptive.py          # NEW: Mode 2 orchestration
│   └── multires.py          # NEW: Mode 3 orchestration
└── core.py                   # Updated: dispatch to modes

tests/
├── test_scene_extractor.py  # Test Cornell box extraction
├── test_jepa_bridge.py      # Test C ABI loading
├── test_cornell_denoise.py  # Test Mode 1 on Cornell
├── test_cornell_adaptive.py # Test Mode 2 on Cornell
└── test_cornell_multires.py # Test Mode 3 on Cornell
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

## Open Questions

1. **Q:** Which Mojo ML framework for neural ops?
   **A:** Nabla (nabla-ml) - has autograd, GPU, SPMD

2. **Q:** How to handle variable scene sizes in transformer?
   **A:** Pad to max size, use mask tokens (like BERT)

3. **Q:** Where to store trained JEPA models?
   **A:** `~/.cache/omen/models/` or project-local `models/`

4. **Q:** Cornell box only for training?
   **A:** Start with Cornell (simple), extend to variety of scenes

5. **Q:** GPU memory budget for JEPA model?
   **A:** Target ~2GB for model + scene graph (fits on 8GB GPUs)
