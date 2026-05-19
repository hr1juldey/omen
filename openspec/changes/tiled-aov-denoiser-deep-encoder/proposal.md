## Why

The current AOV conv2d test processes full images at once, causing OOM at 1024x1024+ (9.9GB VRAM at 512x512). More critically, the scene encoder is a shallow 18→32→128 MLP (~3K params) that mean-pools all scene entities into a single vector — losing light-material-geometry relationships essential for understanding non-local noise (caustics, indirect illumination, off-screen reflections). Research from LeWM (LeCun 2026), D-JEPA (ICLR 2025), and FiLM (AAAI 2018) validates: (1) deep conditioning at every layer via FiLM/AdaLN, (2) multi-term losses including physics regularization, (3) latent-space prediction not pixel reconstruction.

## What Changes

- New test file `tests/test_gpu_tiled_aov_denoiser.py` — single self-contained file
- **Tiled processing**: 256x256 tiles from arbitrary-resolution renders (fits in ~2GB VRAM per tile)
- **Deep scene encoder**: 12-layer residual MLP (18→128→...→128, ~250K params) replacing the current 2-layer shallow encoder
- **FiLM conditioning**: scene_latent injected at every conv layer via γ*scale + β*shift (validated by LeWM AdaLN pattern)
- **Tile position encoding**: 2 sin/cos channels appended to AOV input (10→12ch)
- **Multi-term loss**: MSE latent prediction + SIGReg variance regularization + energy conservation physics loss
- **Decode visualization**: optional `--show` flag renders GT / noisy / denoised side-by-side via matplotlib
- Uses `src/omen/kernels/` (conv2d_safe, silu_mojo, sigmoid_mojo, square) — no modification to src/
- GPU rendering via cuda_ad_rgb Mitsuba variant, 5 random scenes with camera variation
- Kill switch: 24GB WARN / 28GB KILL with sys.exit(99)

## Capabilities

### New Capabilities
- `tiled-aov-encoder`: Tile-based 256x256 AOV encoding with FiLM scene conditioning, 12-layer residual scene encoder, tile position encoding, and multi-term loss (MSE + SIGReg + energy conservation)
- `decode-visualization`: Side-by-side GT / noisy / denoised image output via matplotlib, opt-in via CLI

### Modified Capabilities
<!-- No existing specs modified — this is a new test file that doesn't touch src/omen/ -->

## Impact

- **New file**: `tests/test_gpu_tiled_aov_denoiser.py` (~800 lines)
- **Reads from**: `src/omen/kernels/conv2d.py`, `src/omen/kernels/activations.py`, `src/omen/kernels/activations_gpu.py`, `src/omen/scenes.py`, `src/omen/model/functional/sigreg.py`
- **No modifications** to any src/omen/ files
- **Dependencies**: nabla, mitsuba, numpy, matplotlib (for --show), max.driver Accelerator
- **Hardware**: Requires GPU (RTX 3060 12GB VRAM target), 32GB system RAM
- **Runtime**: First JIT compile ~5-10min (cached), steady-state ~150ms/step at 256x256 tile
