## 1. Scaffolding & Infrastructure

- [ ] 1.1 Create `tests/test_gpu_tiled_aov_denoiser.py` with docstring, imports (nabla, mitsuba, numpy, max.driver, omen kernels), logging setup, and CLI argument parser
- [ ] 1.2 Implement system helpers: `_rss()`, `_vram_mb()`, `_gpu_util()`, `guard()` with WARN=24GB/KILL=28GB thresholds, `dev()`, `to_dev()`, `to_cpu()`, `cleanup()`
- [ ] 1.3 Set Mitsuba variant priority: cuda_ad_rgb > llvm_ad_rgb > scalar_rgb, with sys.setrecursionlimit(50000)

## 2. AOV Data Pipeline

- [ ] 2.1 Implement `_render_aov()` — Mitsuba AOV integrator with `"aovs": "albedo:albedo,normal:sh_normal,depth:depth"`, handle both dict (scalar_rgb) and tensor (cuda_ad_rgb) returns
- [ ] 2.2 Implement `_pack_aov()` — pack albedo(3)+normal(3)+depth(1)+material_id(1)+motion(2) into (H,W,10), zero-fill missing channels
- [ ] 2.3 Implement `_render_pair_with_aov()` — render GT (spp=64) + noisy (spp=2) + AOV, return (aov, gt_rgb, scene_feat) with random scene/camera selection from 5 builders
- [ ] 2.4 Implement `_generate_synthetic_aov()` fallback for when Mitsuba AOV fails
- [ ] 2.5 Implement tile position encoding: `_add_tile_position(aov, tile_row, tile_col, grid_h, grid_w)` appending 2 sin/cos channels (10→12ch)

## 3. Deep Scene Encoder

- [ ] 3.1 Implement `init_scene_encoder_params()` — Linear(18,128) + 10× ResBlock params (W_residual, b_residual each 128→128) + Linear(128,128) = ~250K params with He init
- [ ] 3.2 Implement `scene_encoder_fn(p, scene_features)` — 12-layer residual MLP: input → Linear(18,128) → 10× [silu(x @ W + b) + x] → Linear(128,128) → scene_latent (1, 128)

## 4. FiLM-Conditioned Tile Encoder

- [ ] 4.1 Implement `init_tile_encoder_params(channels=128, aov_ch=12, latent=128)` — Conv1(3,3,12,ch), Conv2(3,3,ch,ch), bias for each, FiLM generator params (W_gamma/b_gamma/W_beta/b_beta per conv layer), pool+linear(ch→latent), cross-attention params (gate_w, gate_b, norm_w, norm_b)
- [ ] 4.2 Implement `_film_modulate(conv_out, scene_latent, W_gamma, b_gamma, W_beta, b_beta)` — compute γ = scene_latent @ W_gamma + b_gamma, β = scene_latent @ W_beta + b_beta, return γ * conv_out + β (reshape to broadcast over spatial dims)
- [ ] 4.3 Implement `tile_encoder_fn(p, aov_tile, scene_latent)` — Conv1(stride=2) → FiLM → silu → Conv2(stride=2) → FiLM → silu → GlobalAvgPool → Linear(128→128) → render_latent
- [ ] 4.4 Implement `cross_attn_fn(p, render_latent, scene_latent)` — gate = sigmoid(render @ gate_w + gate_b), fused = LayerNorm(render + gate * scene)

## 5. Multi-Term Loss

- [ ] 5.1 Implement `loss_fn(p, aov_tile, scene_feat, gt_latent, scene_latent_params)` — full forward: scene_encode → tile_encode → cross_attn → MSE + SIGReg + energy
- [ ] 5.2 Implement SIGReg loss inline: `sigreg = -mean(log(std(fused, axis=0) + 1e-6))`
- [ ] 5.3 Implement energy conservation loss inline: `energy = mean(square(sum(abs(render_lat)) - sum(abs(gt_lat))))`
- [ ] 5.4 Implement `make_loss()` closure with configurable λ_sigreg=0.09 and λ_energy=0.01

## 6. Training Loop

- [ ] 6.1 Implement `init_all_params(latent=128, channels=128)` — merge scene_encoder + tile_encoder + cross_attn params into single dict, log param count
- [ ] 6.2 Implement `train_loop()` — per-step: to_dev all params/data → nb.value_and_grad → realize_all → to_cpu grads → AdamW update (numpy) → guard() → cleanup. Track compile time, steady time, losses
- [ ] 6.3 Implement `run_phase()` — render pair, init params, run train_loop, log results, check all finite

## 7. Tiling Pipeline

- [ ] 7.1 Implement `tile_image(full_aov, tile_size=256, overlap=16)` — split (H,W,C) into list of (tile_size+2*overlap, tile_size+2*overlap, C) tiles with position metadata
- [ ] 7.2 Implement `untile_image(tiles, full_h, full_w, overlap=16)` — stitch tiles back with linear blend in overlap regions
- [ ] 7.3 Implement tiled training step: encode scene once, loop over tiles, aggregate loss across tiles, single gradient update

## 8. Decode & Visualization

- [ ] 8.1 Implement `decode_tile(p, aov_tile, scene_latent)` — run encoder forward, return render_latent as proxy for denoised tile (simple linear decode to RGB: latent @ W_decode → reshape)
- [ ] 8.2 Implement `--show` flag handler: after training, run decode on all tiles, stitch, save matplotlib figure with GT / Noisy / Denoised side-by-side to `logs/tiled_denoise_VISUAL_{timestamp}.png`

## 9. CLI & Sustained Mode

- [ ] 9.1 Implement CLI: `--steps`, `--sustain` (minutes), `--channels` (128 default), `--latent` (128), `--show` (visualization flag), `--resolution` (256 default)
- [ ] 9.2 Implement `_run_sustained()` — sustained training with cosine LR decay, scene re-rendering every N steps, RSS/VRAM monitoring
- [ ] 9.3 Implement `main()` — guard at start, dispatch to phase/sustain/show modes, final guard

## 10. Verification

- [ ] 10.1 Run smoke test: `uv run tests/test_gpu_tiled_aov_denoiser.py --steps 10` at 256x256 single tile, verify loss converges and all values finite
- [ ] 10.2 Run 1000-step convergence test: verify loss reaches ~0.000
- [ ] 10.3 Run multi-tile test: `--resolution 512` (4 tiles), verify stitching produces correct 512x512 output
- [ ] 10.4 Run `--show` visualization: verify GT/Noisy/Denoised figure saved to logs/
