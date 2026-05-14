## Context

Omen's JEPA model has fully wired training components (per-component optimizers, stratified replay buffer, surprise LR modulation, SIGReg loss) but zero training data. The model needs diverse scenes that exercise specific MoE expert routing paths to validate that the 23-expert architecture actually learns specialized representations.

Current state: `complex_scene.py` is a standalone script (not importable), `src/omen/scenes.py` has E-shaped and C-shaped room generators that use a custom `_make_rect`/`_make_wall` API that doesn't match Mitsuba's dict format. Neither produces scene-graph metadata.

## Goals / Non-Goals

**Goals:**
- 5 self-contained Mitsuba dict scene builders following the `complex_scene.py` pattern (mi.ScalarTransform4f, direct dict construction)
- Each scene exercises a distinct set of MoE expert routing paths
- Each scene returns a `scene_graph` metadata dict for scene-graph routing
- Multi-camera placement (4-5 cameras per scene) for view diversity
- 4-channel animation per scene (camera, mesh, material, light) for temporal training data
- Online training pipeline: render → encode → loss → backprop → discard (NO image saving by default)
- `TrainingDataGenerator` renders in-place at full HD, feeds model directly, frees memory
- `--save-images` debug toggle for inspecting renders
- CLI entry point for scene rendering and data generation

**Non-Goals:**
- Loading external .obj/.ply files (all scenes use Mitsuba primitives)
- Blender integration (this is pure Mitsuba training data)
- Saving training images to disk by default (online-only training, `--save-images` toggle for debug)
- Loading external HDRI envmap files (Studio Product uses constant + area lights instead)
- Multi-GPU rendering (single GPU or CPU only)

## Decisions

### D1: Scene builder API returns `(mi.Scene, dict)` tuple

Each `build_*()` function returns `(scene, scene_graph)` where `scene_graph` is a plain dict with geometry/material/light metadata structured for `SceneGraphEncoder.forward()`.

**Rationale**: The `SceneGraphEncoder` expects `scene_graph` with keys `geometry`, `materials`, `lights` containing tensor-like data. Building this alongside the Mitsuba scene ensures the encoder always gets valid input.

**Alternative considered**: Build scene_graph from the Mitsuba scene object post-hoc. Rejected because Mitsuba's Python API makes it hard to extract BSDF parameters programmatically.

### D2: Follow `complex_scene.py` pattern (not `scenes.py`)

Use `mi.ScalarTransform4f`, `_tf()` helper, and direct dict construction. Not the `_make_rect`/`_make_wall` abstraction from `scenes.py`.

**Rationale**: `complex_scene.py` is proven, renders correctly, and uses the canonical Mitsuba dict API. The `scenes.py` helpers use a custom abstraction that may not map 1:1 to Mitsuba's expectations.

### D3: Scene-graph metadata structure

```python
scene_graph = {
    "geometry": {
        "vertices": np.ndarray,  # (N, 3) vertex positions
        "faces": np.ndarray,     # face count
    },
    "materials": {
        "params": np.ndarray,    # (M, 5) material parameters
        "types": list[str],      # ["diffuse", "conductor", ...]
    },
    "lights": {
        "params": np.ndarray,    # (L, 7) light parameters
        "types": list[str],      # ["area", "point", "spot", ...]
    },
}
```

**Rationale**: Matches `SceneGraphEncoder` input format exactly. Each encoder head (geom_linear, mat_linear, light_linear) gets the right shape.

### D4: Online training pipeline (all components, no disk saves)

Training follows the diffusion model pattern but adapted for JEPA latent prediction with ALL model components trained together. The decoder is a **residual noise predictor** (not a full image reconstructor) — it predicts the path-tracing noise/residual, and the clean image is recovered as `clean = noisy - predicted_noise` (see D6).

```
For each training step:
  1. Render GT: 1 image at 256/512 SPP, FULL HD (1920x1080 or higher)
     → encode(scene_graph, gt) → target_latent (1024-dim)
  2. Render noisy: same view at 4^x SPP (4, 16, 64) at SAME full HD resolution
     → encode(scene_graph, noisy) → noisy_latent (1024-dim)
  3. JEPA loss (encoder brain):
     pred_loss = MSE(noisy_latent, target_latent)
  4. Decoder loss (noise/residual prediction):
     predicted_noise = decode(predicted_latent, noisy_image)
     residual = gt_pixels - noisy_pixels
     denoise_loss = MSE(predicted_noise, residual)   ← learn noise, not image
     clean = noisy_pixels - predicted_noise           ← inference formula
  5. SIGReg loss (prevent latent collapse):
     reg_loss = -log(std(predicted_latent) + eps)
  6. EpisodicCorrection (fast per-scene adaptation):
     corrected = main_output + episodic(scene_context)
  7. ARPredictor / LEWM (temporal, when animation frames available):
     temporal_loss = MSE(predicted_next_latent, actual_next_latent)
  8. Total = pred_loss + λ₁×denoise_loss + λ₂×reg_loss + λ₃×temporal_loss
  9. Backprop through ALL components with per-component optimizers
  10. Free both images from memory (no disk save)
```

Key differences from image diffusion:
- **Not predicting Gaussian noise** — predicting clean latent from noisy-render latent
- **Decoder predicts residual/noise** — not full image reconstruction. `clean = noisy - predicted_noise` (like diffusion model training but in render domain)
- **SIGReg** — prevents latent collapse (like VAE regularization but simpler)
- **EpisodicCorrection** — 400× higher LR for fast per-scene adaptation
- **ARPredictor (LEWM)** — temporal world model trained on animation frame sequences
- **Noise = path tracing variance** (low SPP), not Gaussian
- **Input is (scene_graph_3D, noisy_render)** — scene graph is conditioning (like CLIP text in SD)
- **GT is ONE big render** — 256/512 SPP at full HD
- **Everything is in-memory** — no saving to disk by default
- **`--save-images` toggle** for debugging only, off by default

**Rationale**: All components train together end-to-end. The encoder learns good latent representations via JEPA loss, the decoder learns to predict noise/residual guided by the JEPA latent, SIGReg keeps latents diverse, episodic correction adapts fast per-scene, and ARPredictor learns temporal coherence from animation sequences.

### D6: Decoder is a residual noise predictor with U-Net skip connections

The decoder does NOT reconstruct the full image from scratch. Instead, it takes the **JEPA latent + the noisy image** and predicts the **noise/residual map**. The clean image is recovered as:

```
clean = noisy - decode(jepa_latent, noisy_image)
```

This is architecturally superior to full-image reconstruction (Conv2dTranspose) because:
- The noisy image already contains 90%+ of the correct pixels — the decoder only needs to fix what's wrong
- Residual learning is provably easier for neural networks (He et al., ResNet)
- Memory efficient: skip connections carry noisy features directly, no need to decode from 1024-dim alone

**Architecture**: U-Net encoder-decoder with skip connections, following patterns from `docs/research/`:
- **Encoder path**: Conv blocks downsample noisy image, extract multi-scale features
- **Skip connections**: Encoder features concatenated to decoder at each resolution (U-Net pattern)
- **JEPA latent injection**: The 1024-dim JEPA latent is projected and injected at the bottleneck (like conditioning in diffusion models)
- **Decoder path**: Up-sample + skip concat → conv blocks, outputting a single-channel residual/noise map
- **Output**: Same spatial dimensions as input, 3-channel (RGB) residual. `clean = noisy - residual`

**Upsampling**: Use Pixel Shuffle (sub-pixel convolution) or DySample upsampling instead of Conv2dTranspose. Per `docs/research/latent_decoder_and_rendering_survey.md` §1.7, Conv2dTranspose causes checkerboard artifacts; Pixel Shuffle and DySample (CVPR 2024) avoid this entirely.

**Skip connection compression**: MLA-style low-rank compression for skip connections at higher resolutions, per `docs/research/deepseek-technical-survey.md`. At 1920×1080, a single 64-channel feature map at full resolution is ~500MB — MLA compression reduces this to ~50MB with <1% quality loss.

**Researched architectures** (from `docs/research/`):
- MambaIR (ECCV 2024): U-shaped encoder-decoder with Residual State Space Blocks — O(N) complexity
- Restormer: Multi-Dconv Head Transposed Attention — proven for image restoration
- All top denoising architectures use U-Net + residual learning (never full reconstruction)

**Alternative considered**: Conv2dTranspose full-image reconstruction (current `src/omen/model/decoder.py`). Rejected because:
1. Decoder has no access to the noisy image (only latent) — must hallucinate everything from 1024-dim
2. Conv2dTranspose causes checkerboard artifacts (proven in research)
3. Full reconstruction wastes capacity on pixels that are already correct

**Rationale**: The decoder's job is to predict the noise (path-tracing variance) given the JEPA latent as guidance. The JEPA encoder provides scene understanding — the decoder just needs to apply that understanding to fix the noisy pixels. U-Net with skip connections lets the noisy image flow through at full resolution while the JEPA latent conditions the bottleneck.

### D7: Per-pass denoising with scene-graph guided caustic/volumetric/SSS preservation

OIDN and OptiX destroy caustics, volumetrics, and SSS because they only receive `DENOISER_PASS_ALBEDO | DENOISER_PASS_NORMAL` — no scene knowledge, no light path separation. A caustic pixel 15x brighter than the floor albedo is statistically indistinguishable from a firefly, so they remove it.

Omen fixes this at three levels:

**1. Extended AOV passes** — denoise light paths separately, not one combined image:

**Verified Mitsuba 3 AOV support** (from official docs + source):
- Mitsuba 3 AOV integrator supports `<name>:<type>` pairs for surface properties only
- Available: `albedo`, `sh_normal` (shading normals), `dd.y` (depth), `position` (world-space)
- **NOT available**: per-bounce light path passes (`diffuse_direct`, `glossy_direct`, `transmission_direct`, `volume_direct`) — these are Cycles-specific
- `volpath` / `volpathmis` integrators exist for volumetric scenes (separate from AOV integrator)

This means the AOV spec must be renderer-aware (see D8):

```python
# Mitsuba adapter — surface properties only:
MITSUBA_AOV_SPEC = "albedo:albedo,normal:sh_normal,depth:dd.y,position:pos"

# Cycles adapter — full per-bounce light paths:
CYCLES_PASSES = {
    "MUST":  ["albedo", "normal", "depth"],
    "NICE-1": ["diffuse_direct", "glossy_direct"],
    "NICE-2": ["transmission_direct"],
    "NICE-3": ["volume_direct", "volume_scatter"],
}

# Unified output — adapter normalizes to this format:
UNIFIED_AOV = {
    "albedo": 3, "normal": 3, "depth": 1,          # MUST: always present
    "position": 3,                                    # NICE-1: surface property
    "diffuse_direct": 3, "glossy_direct": 3,          # NICE-2: Cycles only
    "transmission_direct": 3,                          # NICE-3: Cycles only
    "volume_direct": 3,                                # NICE-4: Cycles only
}  # Total: 22 channels (Mitsuba fills 13, zeros the rest)
```

**2. Scene graph as ground truth** — the model is TOLD what materials are present:
- `scene_graph["materials"]["types"]` contains `"dielectric"` → caustics are EXPECTED near this object
- `scene_graph["materials"]["types"]` contains `"null"` + volume params → volumetric scattering is EXPECTED
- The spatial latent encodes "add caustic at position (x,y)" not "maybe noise, remove"

**3. Expert routing** — tiles containing glass objects route to the dielectric expert who is trained to preserve caustics. Tiles with fog route to the volume expert. No single filter blindly removing outliers.

**Pass hierarchy (graceful degradation):**

| Tier | Passes | Always available? | Role |
|------|--------|-------------------|------|
| MUST | albedo, normal, depth, combined noisy | Yes — every renderer provides these | Baseline denoising. Model MUST work with only these. |
| NICE-1 | diffuse_direct, glossy_direct | Most renderers | Separate light path denoising |
| NICE-2 | transmission_direct | Cycles, Mitsuba, LuxCore | Caustic preservation |
| NICE-3 | volume_direct, volume_scatter | Cycles, Mitsuba volpath | Volumetric preservation |
| NICE-4 | SSS, emission, UV, motion | Cycles full AOV | Advanced features |

**Graceful degradation training**: The model trains with ALL passes (full knowledge). At inference, if a pass is disabled by the user in Blender, the model gracefully degrades:
- Missing NICE passes are zero-filled in the AOV buffer
- The scene_graph provides fallback — if `transmission_direct` is missing but scene_graph says `"dielectric"`, the model still knows caustics are expected
- The tile fingerprint dimension stays 23 (fixed) but unused pass slots are zeroed
- MoE routing adapts — the dielectric expert still activates based on scene_graph material types, not AOV data alone

**Training strategy**: Randomly drop NICE passes during training (like dropout) to simulate user-disabled passes. 20% of training steps drop NICE-2/NICE-3, 10% drop all NICE passes. This forces the model to learn scene_graph-based fallback behavior.

**Rationale**: OIDN/OptiX are blind — they see only pixels. Omen sees pixels + scene understanding. The scene_graph is the key differentiator that no standalone denoiser can have. Even with zero AOV passes beyond MUST, the scene_graph tells the model what physics to expect. Training with random pass dropout ensures the model never becomes dependent on any single NICE pass.

### D8: Renderer adapter interface (neural network is renderer-agnostic)

The neural network (encoder + decoder + MoE) is renderer-agnostic. It sees a unified AOV buffer format regardless of which renderer produced the data. A `RendererAdapter` translates renderer-specific data into this unified format.

```
┌─────────────────────────────────────────────────────────┐
│              Neural Network (Nabla)                      │
│  Always sees: UNIFIED_AOV (22 channels) + scene_graph   │
└─────────────┬───────────────────────────────────────────┘
              │
     ┌────────┴────────┐
     │ RendererAdapter  │  ← abstract base
     │ produce_aov()    │
     │ list_passes()    │
     │ integrator_type  │
     └────────┬────────┘
              │
    ┌─────────┼──────────┐
    │         │          │
    ▼         ▼          ▼
┌────────┐ ┌────────┐ ┌──────────┐
│Mitsuba │ │ Cycles │ │(Future:  │
│Adapter │ │Adapter │ │LuxCore)  │
└────────┘ └────────┘ └──────────┘
```

**MitsubaAdapter** (training scenes):
- MUST passes: albedo (3ch), normal (3ch), depth (1ch) — via AOV integrator
- NICE-1: position (3ch) — via AOV integrator (`position:pos`)
- NICE-2/3/4: zero-filled — Mitsuba has no per-bounce passes
- Integrator: `path` (default), `volpath` / `volpathmis` (volumetric scenes)
- Scene_graph provides the knowledge that Mitsuba AOVs lack (dielectric → expect caustics)

**CyclesAdapter** (production, Blender addon):
- MUST passes: albedo, normal, depth — always available
- NICE-1: diffuse_direct, glossy_direct — available via Cycles pass system
- NICE-2: transmission_direct — caustics from glass
- NICE-3: volume_direct, volume_scatter — volumetric scattering
- All passes mapped from `PASS_*` enum to unified format
- No zero-filling needed — full knowledge available

**Adapter interface:**
```python
class RendererAdapter(ABC):
    @abstractmethod
    def render(self, scene, spp, sensor, integrator_override=None) -> RenderResult: ...

    @abstractmethod
    def get_aov(self, render_result) -> dict[str, np.ndarray]:
        """Returns unified AOV dict. Missing passes are zero-filled."""
        ...

    @abstractmethod
    def list_available_passes(self) -> list[str]:
        """Returns which passes this renderer actually provides (non-zero)."""
        ...

    @abstractmethod
    def integrator_for_scene(self, scene_graph) -> dict:
        """Returns integrator config (path vs volpath) based on scene content."""
        ...
```

**Rationale**: The adapter pattern decouples the neural network from the renderer. This means:
1. Training on Mitsuba scenes works TODAY even though Mitsuba lacks per-bounce passes
2. When Cycles integration is live, the same model gets richer input and produces better results
3. The model trains with random NICE pass dropout anyway — Mitsuba's zero-filling is just permanent dropout
4. Adding a new renderer (LuxCore, Arnold) requires only a new adapter, no model changes

### D5: Scene progression for expert coverage

| Scene | Materials | Lights | Experts Exercised |
|-------|-----------|--------|-------------------|
| Cornell Box | diffuse (R/G/B walls) | area (top) | diffuse expert, geometry expert, area-light expert |
| Veach Ajar Door | dielectric, conductor, diffuse | point, spot, area | dielectric expert, conductor expert, all light experts |
| Shaderball | conductor, roughconductor, plastic, roughplastic, dielectric | area, env | ALL material experts |
| Studio Product | roughconductor (Au/Cu), roughplastic | area (key+fill+rim) | conductor expert, roughplastic expert |
| Foggy Corridor | diffuse, null BSDF | point, spot | null BSDF handling, volume, point-light expert |

## Risks / Trade-offs

**[Risk] Render time for high-SPP GT at full HD** → Mitigation: 256 SPP at 1920x1080 is ~5-15s on GPU. Training step dominates over render time since we backprop through the full encoder+decoder. Can use 128 SPP for early training, increase to 512 for fine-tuning.

**[Risk] Full HD images use significant GPU memory** → Mitigation: Render → downsample to 480x270 for encoder input (reduced resolution encoding), but GT latent is computed from full HD. Only the numpy array lives in memory briefly, freed after loss computation.

**[Risk] Scene-graph metadata may not perfectly match live Blender data** → Mitigation: The metadata schema matches `SceneGraphEncoder` input format. When Blender integration is live, the same schema will be used from the converter.

**[Risk] 5 scenes may not cover all 23 experts** → Mitigation: Each scene targets a subset. The Shaderball scene exercises ALL material experts. Full expert coverage is achieved across all 5 scenes combined. More scenes can be added later.

**[Risk] Animation frames multiply render cost** → Mitigation: Animation is optional (`--animate` flag). Per-frame cost is same as static. 170 animation frames at full HD is ~25 minutes total render time across all 5 scenes — acceptable for dataset generation.
