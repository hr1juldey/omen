# Omen — Scene-Aware JEPA Render Accelerator

> **Omen** is a research rendering engine that uses JEPA (Joint Embedding Predictive Architecture) for scene-aware path tracing acceleration.

## Components

| Component | Description | Status |
|-----------|-------------|--------|
| **Mitsuba Integrator** | JEPA-accelerated path tracing plugin for Mitsuba 3 | ✅ Implemented |
| **Blender Addon** | Blender integration via Mitsuba-Blender bridge | 🚧 In Development |
| **JEPA Acceleration** | World model for adaptive sampling | 📋 Planned |
| **Mojo GPU Kernels** | High-performance rendering kernels | 📋 Planned |

## Quick Start (Mitsuba Integrator)

### Setup

```bash
# One-command setup
./setup.sh
```

Or manually:
```bash
pixi install
pixi run pip install mitsuba drjit
```

### Basic Usage

```python
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

# Register Omen
from omen_integrator import register
register()

# Render with Omen
scene = mi.load_dict(mi.cornell_box())
result = mi.render(scene, integrator=mi.load_dict({'type': 'omen'}))
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | int | -1 (infinite) | Maximum path bounces |
| `rr_depth` | int | 5 | Russian roulette start depth |
| `jepa_model` | string | "" | Path to JEPA model (future) |
| `use_gpu` | boolean | true | Enable GPU acceleration (future) |

## Installation

### Prerequisites

- Python 3.14
- Pixi package manager
- CUDA-capable GPU (optional, for future GPU acceleration)

### Install as Blender Addon (Future)

1. In Blender, go to **Edit → Preferences → Add-ons → Install...**
2. Navigate to `omen/` and select the folder
3. Enable the "Omen: JEPA Render Accelerator" addon

## Architecture

```
Blender Scene → Mitsuba-Blender → Mitsuba Engine → Omen Integrator
     │               │                  │                │
     ├─ bpy.data    ├─ Export          ├─ Scene        ├─ Path Tracer
     ├─ Depsgraph  │ to XML           │ Graph         │ (current)
     └─ Render     │                  │ Sensors       │
       Engine      └─ Integrator      │ Film          │
                   Selection          └─ BSDFs        │
                                       │                │
                                       └────────────────┘
                                              │
                                        JEPA Model
                                        (future)
```

## Development

### Project Structure

```
omen/
├── README.md                 # This file
├── SETUP.md                  # Setup instructions
├── setup.sh                  # One-command setup script
├── pixi.toml                 # Pixi environment configuration
│
├── src/                      # Mitsuba integrator plugin
│   └── omen_integrator/
│       ├── __init__.py       # Plugin registration
│       ├── core.py           # Main render loop
│       ├── path.py           # Path tracing logic
│       ├── direct.py         # Direct illumination
│       ├── jepa.py           # JEPA integration (future)
│       └── gpu.py            # GPU kernels (future)
│
├── openspec/                 # Change management
│   └── changes/
│       └── omen-mitsuba-integrator-plugin/
│           ├── proposal.md
│           ├── design.md
│           ├── specs/
│           └── tasks.md
│
└── docs/                     # Documentation
    ├── ARCHITECTURE.md
    ├── BPY_MITSUBA_FINDINGS.md
    └── ...
```

### Key Design Decisions

1. **Mitsuba-First Development**: Start with Mitsuba 3 plugin, proven renderer for research
2. **Python Plugin**: Pure Python implementation for rapid prototyping
3. **Blender-Free**: No Blender dependency for core rendering logic
4. **Incremental JEPA Integration**: Add JEPA acceleration in phases

## Status

[![Mitsuba](https://img.shields.io/badge/Mitsuba-3.8.0-green)]()
[![Python](https://img.shields.io/badge/Python-3.14-blue)]()
[![Pixi](https://img.shields.io/badge/Pixi-Ready-orange)]()

**Current Phase**: Mitsuba integrator implemented. JEPA acceleration pending.

## Contributing

This is a research project. Contributions welcome in:
- JEPA model architecture
- Mitsuba integration improvements
- Blender addon development
- GPU kernel optimization

## License

GPL-3.0-or-later (required for Blender addon distribution)

## Acknowledgments

- **Mitsuba 3** — Research rendering system
- **Dr.Jit** — Just-in-time compilation for rendering
- **Facebook AI Research** — JEPA architecture (arxiv 2504.14151, CC-BY-NC license)
- **Modular** — Mojo programming language


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
