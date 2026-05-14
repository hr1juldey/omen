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

### D4: Online training pipeline (diffusion-like, no disk saves)

Training follows the diffusion model pattern but adapted for JEPA latent prediction:

```
For each training step:
  1. Render GT: 1 image at 256/512 SPP, FULL HD (1920x1080 or higher)
     → encode to latent via SceneGraphEncoder + RenderFeatureEncoder → target_latent
  2. Render noisy: same view at 4^x SPP (4, 16, 64) at SAME full HD resolution
     → encode with scene_graph → noisy_latent
  3. Model predicts: clean_latent = model(noisy_latent, scene_graph)
  4. Loss = JEPA_loss(predicted_latent, target_latent)
  5. Backprop, optimizer step
  6. Free both images from memory (no disk save)
```

Key differences from image diffusion:
- **Not predicting Gaussian noise** — predicting clean latent from noisy-render latent
- **Noise = path tracing variance** (low SPP), not Gaussian
- **Input is (scene_graph_3D, noisy_render)** — scene graph provides structure, noisy render provides appearance
- **GT is ONE big render** — 256/512 SPP at full HD, not multiple cheap renders
- **Everything is in-memory** — no saving to disk, no .npz files by default
- **`--save-images` toggle** for debugging only, off by default

**Rationale**: This matches how diffusion models train (generate pair in-place, train, discard) but uses JEPA's latent prediction instead of noise prediction. The scene_graph input is what makes this unique — it's conditioning information like CLIP text embeddings in Stable Diffusion, but for 3D scene structure.

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
