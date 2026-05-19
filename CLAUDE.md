# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Setup (uv is the primary package manager, NOT pixi)
./setup.sh

# Run Python tests
uv run pytest tests/test_denoiser.py -x
uv run pytest tests/ -k "pattern" -x

# Run specific GPU training scripts
uv run tests/test_gpu_omen_scale.py --phase 0
uv run tests/test_gpu_tiled_aov_denoiser.py --steps 100

# Run Mojo GPU programs directly
mojo run tests/test_mojo_tiled_denoiser.mojo
mojo run tests/gpu_stress.mojo

# Lint (mandatory after every code change)
ruff check --fix .
ruff format .
```

## System Requirements

- **GPU**: NVIDIA RTX 3060 12GB VRAM (or equivalent)
- **RAM**: 32GB (nabla JIT compilation peaks at 14-27GB)
- **Python**: 3.13+
- **Mojo/MAX**: installed via modular (nightly channel)
- **Package manager**: `uv` (NOT pixi)

## Architecture

### Dual-Runtime: Python (orchestration) + Mojo (GPU compute)

Python orchestrates rendering and training. Mojo handles GPU kernels. The two interop via:
- **Nabla custom kernels**: `call_custom_kernel("name", dir, tensor, type, device=tensor.device)` — the `device=` param is MANDATORY or you get SIGSEGV
- **Mojo Python extension modules**: `@export` functions + `PythonModuleBuilder` for Python-callable Mojo
- **Python from Mojo**: `Python.import_module()`, `Python.evaluate()`, `PythonObject`

### Package Layout (`src/`)

```
src/
├── omen/                          # Core library
│   ├── model/                     # JEPA model architecture
│   │   ├── jepa.py               # Top-level JEPA model
│   │   ├── functional/           # Functional building blocks
│   │   │   ├── scene_encoder.py  # Deep MLP, scene features → latent
│   │   │   ├── render_encoder.py # Conv2d-based, AOV tiles → latent
│   │   │   ├── cross_attn.py     # FiLM cross-attention fusion
│   │   │   ├── decoder.py        # U-Net decoder
│   │   │   ├── ar_predictor.py   # Autoregressive predictor
│   │   │   ├── confidence.py     # Confidence head
│   │   │   ├── episodic.py       # Episodic correction
│   │   │   └── sigreg.py         # Sigma regularization loss
│   │   ├── moe/                  # Mixture of experts
│   │   ├── mla_skip.py           # Multi-head latent attention with skip
│   │   └── tier_config.py        # Tiered model configurations
│   ├── kernels/                   # Mojo GPU kernels + Python bridges
│   │   ├── activations_gpu.mojo/.py  # sigmoid, silu (Padé approx)
│   │   ├── conv2d_im2col.mojo     # im2col for conv2d
│   │   ├── aov_pack.mojo/.py      # AOV channel packing
│   │   ├── tile_fingerprint.mojo  # Tile hashing
│   │   ├── moe_dispatch.mojo/.py  # MoE routing
│   │   ├── mla_compress.mojo/.py  # MLA compression
│   │   └── ssim_kernel.mojo/.py   # SSIM computation
│   ├── training/                  # Training infrastructure
│   │   ├── trainer/              # Compiled trainer, optimizers, loss
│   │   ├── online_gen.py         # Online data generation (Mitsuba)
│   │   ├── tile.py               # Tiling utilities
│   │   └── anim_gen.py           # Animation data generation
│   ├── scenes.py                  # 5 scene builders (cornell, veach, shaderball, studio, foggy)
│   └── modes/                     # Runtime modes (denoiser, animation, adaptive, etc.)
├── omen_engine/                   # Blender/Mitsuba rendering backend
├── omen_integrator/               # Mitsuba 3 path tracer plugin
└── omen_blender/                  # Blender addon integration
```

### Pure Mojo GPU Denoiser (`tests/test_mojo_tiled_denoiser.mojo`)

Self-contained Mojo GPU denoiser with forward+backward+AdamW — no nabla, no Python at runtime. Architecture:
- **Scene encoder**: Linear(18,128) + [depth-2 ResBlocks with SiLU] + Linear(128,128)
- **Tile encoder**: im2col→Conv1(15→128)→FiLM→SiLU→Conv2(128→128)→FiLM→SiLU→Pool→Linear(128,128)
- **FiLM conditioning**: gamma/beta = Linear(scene_latent) — modulates tile features
- **Cross-attention fusion**: gate = sigmoid(tile_lat @ W + b), fused = tile_lat + gate * scene_lat
- **Loss**: MSE + SIGReg + energy

## Critical Gotchas

### Nabla
- `nb.zeros(32)` WRONG → `nb.zeros((32,))` (must be tuple shape)
- nn.Linear weight layout: (in_features, out_features) — `x @ weight + bias`, NO transpose
- No `nn.Conv2d` → use `nb.conv2d` functional API, HWIO filter layout
- `nb.realize_all(dict)` BUG: doesn't recurse into dict values — realize each tensor explicitly
- Native `nb.conv2d` SIGABRT on backward with 2+ layers — use `conv2d_safe` (im2col+matmul) instead
- JIT compilation is **persisted to disk** at `~/.cache/modular/` — first compile is 14-27GB RAM / 4-10 min; cached runs are 2-14GB / 1-5 min
- Maximum 8 `conv2d_safe` layers on 32GB RAM (14.4GB peak compile)
- Nabla VJP bugs: `nb.sigmoid/silu/tanh/sqrt/log` create CPU scalar constants on GPU → use GPU-safe replacements in `activations.py`

### Mojo GPU
- Kernels are plain `def` (no `__global__` decorator)
- Launch: `ctx.enqueue_function[kernel](args, grid_dim=G, block_dim=B)`
- **Must bind comptime params first**: `comptime kernel = my_kernel[type_of(layout)]`
- `comptime assert tensor.flat_rank == N` required before any subscript on TileTensor
- Use `rebind[Scalar[dtype]](tensor[idx])` for arithmetic between different-layout tensors
- Shared memory: `stack_allocation[dtype, address_space=AddressSpace.SHARED](layout)`
- **Matmul kernel threading**: `matmul_1d_kernel` needs M*N threads — use `grid_dim=ceildiv(M*N, BLOCK_SIZE), block_dim=BLOCK_SIZE` (NEVER `grid_dim=1, block_dim=(1,16)`)
- `global_idx.x` returns `Int` — compare directly

### Mojo Syntax (latest)
- `def` only (no `fn`), add `raises` explicitly when needed
- `comptime` replaces `alias` and `@parameter`
- `out self` for constructors, `mut self` for mutable methods
- `PythonObject` → Mojo: MUST use `py=` keyword (`Int(py=obj)`, NOT `Int(obj)`)
- No `let` keyword — use `var` always
- `from std.python import Python, PythonObject` (NOT `from python`)

### Mitsuba
- Set variant before any import: `mi.set_variant("scalar_rgb")` or `"cuda_ad_rgb"`
- AOV integrator: `"aovs": "albedo:albedo,normal:sh_normal,depth:depth,position:position,uv:uv,material:shape_index"`
- `scalar_rgb` returns TensorXf (not dict) — extract channels by offset
- `cuda_ad_rgb` returns dict — check with `isinstance(result, dict)`
- AOV channel "normal" is WRONG → use "sh_normal"
- drjit/mitsuba crashes on process exit — use `os._exit(0)` to skip C++ destructor cleanup

## Code Policy (from CLAUDE_POLICY.md)

- **Absolute imports only** — no `from .` or `from ..`
- **100 lines executable code per file**, 50 lines overhead (test files exempt)
- **Ruff compliance mandatory** — `ruff check --fix` + `ruff format` after every change
- SOLID principles — single responsibility, no god objects
- No magic numbers, no circular imports, no deeply nested conditionals
- Priority: imports > ruff > file size > architecture boundaries > design principles

## GPU Training Benchmarks (64x64, AdamW lr=1e-3)

| Layers | Conv type | Convergence | Steady-state RAM |
|--------|-----------|-------------|------------------|
| 2 | conv2d_safe | loss→0 at ~step 200 | 1.7GB cached |
| 4 | conv2d_safe | loss→0 at ~step 500 | 2.0GB cached |
| 8 | conv2d_safe | loss→0 at ~step 700 | 2.2GB cached |
| 16 | conv2d_safe | **OOM** (27GB+ compile) | N/A |

GPU utilization: 1-8% at 64x64, 39% at 256x256, 84% at 512x512.
