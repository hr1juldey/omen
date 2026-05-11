# Omen — Scene-Aware JEPA Render Accelerator for Blender

> **Omen** is a Blender addon that uses a JEPA (Joint Embedding Predictive Architecture) world model to denoise and accelerate Cycles path-traced rendering by understanding the 3D scene graph.

## What Omen Does

| Mode | What It Does | Impact | Complexity |
|------|-------------|--------|------------|
| **Mode 1: Denoiser** | Post-process denoising with 3D scene knowledge | Replace OptiX/OIDN | Baseline |
| **Mode 2: Accelerator** | Predict "obvious" pixels, only path-trace the "confusing" ones | 4-8x fewer samples needed | Medium |
| **Mode 3: Multi-Resolution** | High spp at low res + low spp at high res, JEPA fills the gap | 8-16x effective speedup | Hard |

## The Fundamental Advantage

Existing denoisers (OptiX, OIDN) operate on 2D image data only. They have no concept of:
- What objects are in the scene
- Where lights are positioned
- What materials are being rendered
- The underlying 3D geometry

**Omen knows the scene**. We extract the exact scene graph from Blender:
- Evaluated geometry from the BVH (modifiers, armatures, subdivision applied)
- Exact material parameters from Principled BSDF inputs
- All light sources: light objects, emissive materials, **and geometry node instances**
- Ground truth on demand — render at any spp for self-training

This allows JEPA to learn "what this scene *should* look like" rather than just removing noise from pixels.

## Installation

### Prerequisites

- Blender 5.1+
- Mojo SDK (Q1 2026 or later)
- CUDA-capable GPU (NVIDIA) or HIP-capable GPU (AMD)

### Build

```bash
cd omen/lib
./build.sh  # Builds libomen_core.so from Mojo source
```

### Install as Blender Addon

1. In Blender, go to **Edit → Preferences → Add-ons → Install...**
2. Navigate to `omen/` and select the folder
3. Enable the "Omen: JEPA Render Accelerator" addon

## Usage

### Quick Start

```python
import bpy

# Enable Omen
scene = bpy.context.scene
scene.omen.use_denoiser = True
scene.omen.mode = 'DENOISER'  # or 'ACCELERATOR' or 'MULTIRES'

# Render as normal
bpy.ops.render.render(animation=True)
```

### Python API

```python
import bpy

scene = bpy.context.scene
omen = scene.omen

# Mode selection
omen.mode = 'DENOISER'       # Basic denoising
omen.mode = 'ACCELERATOR'    # Predict obvious pixels
omen.mode = 'MULTIRES'       # Multi-resolution upscaling

# Training settings
omen.self_training = True
omen.training_interval = 8   # Train every 8 frames
omen.high_spp_multiplier = 4  # Ground truth = 4x samples

# Performance tuning
omen.patch_size = 8          # JEPA patch size (pixels)
omen.tile_size = 128         # Render tile size (patches)
omen.gpu_device_id = 0       # GPU to use

# Memory management
omen.texture_cache_mb = 2048
omen.geometry_cache_mb = 1024
```

### Panel Location

Find Omen settings in the **Render Properties** tab under:
- **Omen Denoiser** section (for Mode 1)
- **Omen Accelerator** section (for Mode 2)
- **Omen Training** section (self-training controls)

## Architecture

For detailed architecture documentation, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

### Quick Architecture Overview

```
Blender Scene → Python Addon → Mojo Core (.so) → GPU
     │               │              │            │
     ├─ bpy.data    ├─ Scene      ├─ JEPA     ├─ TileTensor
     ├─ Depsgraph  │ Extractor   │ Model     │ Shared Memory
     └─ Render     ├─ Material   │ Scene      │
       Engine      │ Reader      │ Graph      │
                   └─ FFI        └─ GPU       │
                   Bridge       Memory       │
```

## Development

### Project Structure

```
omen/
├── ARCHITECTURE.md           # Detailed HLD/LLD
├── README.md                 # This file
├── pyproject.toml
├── main.py                   # Addon entry point
│
├── python/                   # Python addon code
│   ├── scene_extractor.py    # BVH/geometry extraction
│   ├── material_reader.py    # Material parameters
│   ├── light_reader.py       # Light detection
│   └── ...
│
├── mojo/                     # Mojo core engine
│   ├── omen_core.mojo
│   ├── jepa_model.mojo
│   └── ...
│
└── tests/                    # Unit tests
```

### Key Design Decisions

1. **Mojo Core, Python Shell**: All heavy computation in Mojo. Python only orchestrates.
2. **Zero-Copy GPU**: CUDA/HIP pointers wrapped via `DeviceBuffer(raw_ptr, owning=False)`.
3. **Evaluated Geometry Only**: Always use `depsgraph.evaluated_get()` for final geometry.
4. **Emissive Instance Detection**: Critical for geometry node instanced lights.

## Contributing

This is a research project. Contributions welcome in:
- JEPA model architecture improvements
- Additional render pass integration
- Support for more material types
- Performance optimizations

## License

GPL-3.0-or-later (required for Blender addon distribution)

## Acknowledgments

- **Facebook AI Research** — JEPA architecture (arxiv 2504.14151, CC-BY-NC license)
- **Blender Foundation** — Cycles render engine
- **Modular** — Mojo programming language
- **Nabla-ML** — Autograd infrastructure

## Contact

For questions or discussions:
- Project discussions: [GitHub Issues](https://github.com/yourusername/omen/issues)
- Blender dev chat: #module-render-cycles on chat.blender.org

## Status

[![Phase](https://img.shields.io/badge/Phase-1%20Research-red)]()
[![Mojo](https://img.shields.io/badge/Mojo-Q1%202026-blue)]()
[![Blender](https://img.shields.io/badge/Blender-5.1%2B-orange)]()

**Current Phase**: Architecture research complete. Implementation pending.
