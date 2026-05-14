## Why

The JEPA model has no training data. The fix-training-wiring change wired all the optimizer/loss/replay components, but the model has zero scenes to train on. We need self-contained Mitsuba scenes (no external assets, no Blender) that exercise specific MoE expert groups, so we can start self-supervised training immediately and validate that each expert routing path works end-to-end.

## What Changes

- Add 5 classic benchmark scenes as pure Mitsuba dict builders in `src/omen/scenes.py`:
  1. **Cornell Box** — GI/color bleeding, diffuse walls + area light (exercises: diffuse expert, geometry expert, area-light expert)
  2. **Veach Ajar Door** — MIS stress test, dielectric+point+spot lights (exercises: dielectric expert, all light experts, bidirectional routing)
  3. **Shaderball** — material validation sphere on checker plane, all BSDF types (exercises: material experts — roughconductor, plastic, roughplastic, conductor, dielectric)
  4. **Studio Product** — HDRI-lit product scene, conductor+roughplastic objects (exercises: conductor expert, roughplastic expert, envmap expert)
  5. **Foggy Corridor** — volumetric scene with homogeneous medium (exercises: null BSDF, volume expert, volpath integrator)
- Each scene returns a `(mi.Scene, scene_graph_dict)` tuple where `scene_graph_dict` contains geometry/material/light metadata for scene-graph routing
- Add a `TrainingDataGenerator` class that renders multi-SPP pairs (low-SPP noisy / high-SPP clean) for JEPA self-supervised learning
- Add a CLI entry point `python -m omen.scenes --scene cornell --spp-pair 4,256` for quick scene rendering and data generation

## Capabilities

### New Capabilities
- `training-scenes`: 5 benchmark scene builders with scene-graph metadata, training data generator for JEPA self-supervised learning pairs

### Modified Capabilities
- `online-training`: Training data generator integrates with StratifiedReplayBuffer for per-scene stratified sampling

## Impact

- **Files modified**: `src/omen/scenes.py` (major expansion), `src/omen/training/data_gen.py` (training data generator)
- **Files created**: None new (all goes into existing modules)
- **Dependencies**: Mitsuba 3 (already required), numpy (already required)
- **No breaking changes** — purely additive
