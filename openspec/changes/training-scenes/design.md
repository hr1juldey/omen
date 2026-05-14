## Context

Omen's JEPA model has fully wired training components (per-component optimizers, stratified replay buffer, surprise LR modulation, SIGReg loss) but zero training data. The model needs diverse scenes that exercise specific MoE expert routing paths to validate that the 23-expert architecture actually learns specialized representations.

Current state: `complex_scene.py` is a standalone script (not importable), `src/omen/scenes.py` has E-shaped and C-shaped room generators that use a custom `_make_rect`/`_make_wall` API that doesn't match Mitsuba's dict format. Neither produces scene-graph metadata.

## Goals / Non-Goals

**Goals:**
- 5 self-contained Mitsuba dict scene builders following the `complex_scene.py` pattern (mi.ScalarTransform4f, direct dict construction)
- Each scene exercises a distinct set of MoE expert routing paths
- Each scene returns a `scene_graph` metadata dict for scene-graph routing
- A `TrainingDataGenerator` that renders (noisy, clean) pairs at configurable SPP
- CLI entry point for quick scene rendering and data export

**Non-Goals:**
- Loading external .obj/.ply files (all scenes use Mitsuba primitives)
- Blender integration (this is pure Mitsuba training data)
- Scene animation/temporal sequences (separate change)
- HDRI envmap loading (Studio Product scene uses constant + area lights instead)
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

### D4: Training pair generation strategy

`TrainingDataGenerator` renders the same scene at two SPP levels:
- **Noisy** (low SPP): 4-16 spp — simulates real-time path tracing
- **Clean** (high SPP): 128-1024 spp — pseudo ground truth

Pairs are rendered with different seeds to avoid correlation. The generator can batch-render multiple pairs and store them as numpy arrays.

### D5: Scene progression for expert coverage

| Scene | Materials | Lights | Experts Exercised |
|-------|-----------|--------|-------------------|
| Cornell Box | diffuse (R/G/B walls) | area (top) | diffuse expert, geometry expert, area-light expert |
| Veach Ajar Door | dielectric, conductor, diffuse | point, spot, area | dielectric expert, conductor expert, all light experts |
| Shaderball | conductor, roughconductor, plastic, roughplastic, dielectric | area, env | ALL material experts |
| Studio Product | roughconductor (Au/Cu), roughplastic | area (key+fill+rim) | conductor expert, roughplastic expert |
| Foggy Corridor | diffuse, null BSDF | point, spot | null BSDF handling, volume, point-light expert |

## Risks / Trade-offs

**[Risk] Render time for high-SPP pairs** → Mitigation: Default to 256spp for clean pairs (~5s on GPU). CLI flag to increase. GPU-batched rendering.

**[Risk] Scene-graph metadata may not perfectly match live Blender data** → Mitigation: The metadata schema matches `SceneGraphEncoder` input format. When Blender integration is live, the same schema will be used from the converter.

**[Risk] 5 scenes may not cover all 23 experts** → Mitigation: Each scene targets a subset. The Shaderball scene exercises ALL material experts. Full expert coverage is achieved across all 5 scenes combined. More scenes can be added later.
