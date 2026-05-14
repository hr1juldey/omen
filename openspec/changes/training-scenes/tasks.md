## 1. Scene Infrastructure

- [ ] 1.1 Add `_tf()` helper and `_build_sensor_multi()` to `src/omen/scenes.py` — multi-camera sensor builder that places N cameras at configurable positions around the scene, returning a list of (camera_name, sensor) pairs
- [ ] 1.2 Add `_build_scene_graph()` helper — extracts geometry vertices, material params, light params from scene definition dicts into numpy arrays matching `SceneGraphEncoder` input format
- [ ] 1.3 Add `SCENE_REGISTRY` dict mapping scene names to their builder functions
- [ ] 1.4 Add `SceneAnimation` class — takes a base scene dict + a list of animation channels (camera, mesh, material, light), generates per-frame scene dicts by interpolating parameters

## 2. Cornell Box

- [ ] 2.1 Build `build_cornell_box()` — 6-wall box room: red left wall (diffuse, 0.63,0.065,0.05), green right wall (diffuse, 0.14,0.45,0.091), white floor/ceiling/back (diffuse, 0.725,0.71,0.68), tall box and short box on floor (white diffuse)
- [ ] 2.2 Place area light on ceiling — rectangle emitter (0.8,0.8,0.8) at y=1.325
- [ ] 2.3 Place 5 cameras: front, left-45, right-45, top-down, close-up on tall box
- [ ] 2.4 Build scene_graph metadata — geometry (12 vertices from 6 walls + 2 boxes), materials (3 types), lights (1 area)
- [ ] 2.5 Camera animation: 12-frame orbit around scene center at fixed radius and height
- [ ] 2.6 Mesh animation: 8-frame sequence rotating the tall box 90°, translating the short box across the floor
- [ ] 2.7 Material animation: 6-frame sequence shifting the red wall color toward orange (0.63→0.8), green wall toward teal (0.14→0.3)
- [ ] 2.8 Light animation: 8-frame sequence dimming the area light from full (1.0) to half (0.5) and shifting color temperature from neutral to warm

## 3. Veach Ajar Door

- [ ] 3.1 Build `build_veach_ajar()` — dark room (diffuse black walls) with slightly open door gap on one wall
- [ ] 3.2 Place glass sphere (dielectric, ior=1.5), metal sphere (conductor, Au), matte sphere (diffuse, white) on floor
- [ ] 3.3 Place 3 lights: point light (warm, through door gap), spot light (from above), area light (behind camera)
- [ ] 3.4 Place 5 cameras: through door gap, side view, above, close on glass sphere, close on metal sphere
- [ ] 3.5 Build scene_graph metadata — 3 material types (dielectric, conductor, diffuse), 3 light types (point, spot, area)
- [ ] 3.6 Camera animation: 10-frame dolly moving from far corridor through the door gap into the room
- [ ] 3.7 Mesh animation: 8-frame sequence opening the door from 5° to 45°, moving glass sphere 0.3 units to the right
- [ ] 3.8 Material animation: 6-frame sequence changing glass IOR from 1.3 to 1.8 (visible refraction shift), metal roughness from 0.02 to 0.3
- [ ] 3.9 Light animation: 8-frame sequence moving point light from behind door to room center, dimming spot light from 80 to 20

## 4. Shaderball

- [ ] 4.1 Build `build_shaderball()` — ground plane (checkerboard roughplastic), 5 spheres in a row: conductor (mirror), roughconductor (Cu, α=0.15), plastic (SSS-like, skin tone), roughplastic (clay, α=0.2), dielectric (glass, ior=1.5)
- [ ] 4.2 Place area light above + constant environment emitter (0.4,0.4,0.45)
- [ ] 4.3 Place 4 cameras: front, 45-degree angle, top-down, close-up on material row
- [ ] 4.4 Build scene_graph metadata — 5 material types, 1 area light + 1 env light
- [ ] 4.5 Camera animation: 12-frame circular orbit around the material row at 45° elevation
- [ ] 4.6 Mesh animation: 8-frame sequence scaling each sphere from 0.5x to 1.5x size (one at a time), plus a bounce on the central sphere
- [ ] 4.7 Material animation: 10-frame sequence sweeping roughness on roughconductor from 0.0 to 0.5, roughplastic from 0.1 to 0.4, then cycling plastic IOR from 1.3 to 1.7
- [ ] 4.8 Light animation: 8-frame sequence rotating area light position 180° around the spheres (simulating studio light sweep)

## 5. Studio Product

- [ ] 5.1 Build `build_studio_product()` — 3 objects: gold sphere (roughconductor Au, α=0.05), copper cylinder (roughconductor Cu, α=0.12), matte vase (roughplastic, α=0.25) on a dark ground plane
- [ ] 5.2 Place 3-point studio lighting: key area light (warm, 45° right-above), fill area light (cool, 45° left), rim area light (behind/above)
- [ ] 5.3 Place 4 cameras: product front, product 3/4 view, overhead, low-angle hero
- [ ] 5.4 Build scene_graph metadata — roughconductor + roughplastic materials, 3 area lights
- [ ] 5.5 Camera animation: 12-frame turntable orbit at 3/4 view height (full 360° sweep)
- [ ] 5.6 Mesh animation: 8-frame sequence lifting gold sphere up 0.5 units, rotating copper cylinder 180°, scaling vase from 0.8x to 1.2x
- [ ] 5.7 Material animation: 6-frame sequence changing gold roughness 0.02→0.2, copper roughness 0.1→0.3, vase color shift from matte gray to warm terracotta
- [ ] 5.8 Light animation: 8-frame sequence dimming key light from 100% to 30%, brightening fill to compensate, shifting rim light color from neutral to blue

## 6. Foggy Corridor

- [ ] 6.1 Build `build_foggy_corridor()` — L-shaped corridor (diffuse gray walls), null-BSDF volume boundary (homogeneous medium, σ_t=0.2, albedo=0.9, Henyey-Greenstein g=0.3)
- [ ] 6.2 Place point light at corridor junction, spot light at one end
- [ ] 6.3 Place 4 cameras: corridor entrance, junction looking both ways, down the long arm
- [ ] 6.4 Build scene_graph metadata — diffuse + null BSDF, volume params (σ_t, albedo, g), 2 lights
- [ ] 6.5 Camera animation: 10-frame walkthrough from corridor entrance to junction to end of long arm
- [ ] 6.6 Mesh animation: 6-frame sequence moving a diffuse box obstacle along the corridor (blocking/unblocking light path)
- [ ] 6.7 Material animation: 8-frame sequence changing wall color from gray to warm white, floor from dark to checkerboard
- [ ] 6.8 Light animation: 8-frame sequence varying fog density (σ_t 0.05→0.5), moving point light along corridor, spot light cone narrowing from 45° to 15°

## 7. Decoder Redesign (Residual Noise Predictor)

- [ ] 7.1 Rewrite `src/omen/model/decoder.py` — replace Conv2dTranspose full-image reconstruction with U-Net residual noise predictor. Decoder takes (jepa_latent, noisy_image) and outputs a noise/residual map. `clean = noisy - predicted_noise`
- [ ] 7.2 Implement U-Net encoder path — Conv blocks that downsample noisy image, extract multi-scale features (64→128→256 channels, 4 downsample stages)
- [ ] 7.3 Implement JEPA latent injection at bottleneck — project 1024-dim JEPA latent to spatial feature map and inject at U-Net bottleneck (conditioning, like diffusion model cross-attention)
- [ ] 7.4 Implement U-Net decoder path with skip connections — upsample + concat encoder features at each resolution, conv blocks, output 3-channel residual map
- [ ] 7.5 Use Pixel Shuffle (or DySample) upsampling — NOT Conv2dTranspose. Per `docs/research/latent_decoder_and_rendering_survey.md` §1.7, Conv2dTranspose causes checkerboard artifacts
- [ ] 7.6 Add MLA-style compression for skip connections at higher resolutions — per `docs/research/deepseek-technical-survey.md`, compress 64-channel full-res features from ~500MB to ~50MB
- [ ] 7.7 Update `src/omen/jepa_inference.py` — change denoise flow from `encode → decode(latent) → return full RGBA` to `encode → predict_noise(latent, noisy) → return noisy - noise`
- [ ] 7.8 Update `src/omen/model/jepa.py` — change `decode()` to pass noisy image to decoder: `self.decoder(latent, noisy_image)` instead of `self.decoder(latent, height, width)`
- [ ] 7.9 Wire decoder loss into training — in `train_step_online()`, compute `residual = gt - noisy`, then `denoise_loss = MSE(predicted_noise, residual)` (not MSE against full image)

## 8. Renderer Adapter + Extended AOV + Graceful Degradation

- [ ] 8.1 Create `src/omen/render_adapter.py` with `RendererAdapter` abstract base class — methods: `render()`, `get_aov()`, `list_available_passes()`, `integrator_for_scene()`
- [ ] 8.2 Implement `MitsubaAdapter(RendererAdapter)` — wraps Mitsuba render + AOV integrator. MUST passes (albedo, normal, depth) via `aov:sh_normal,dd.y` spec. NICE-1 (position) via `position:pos`. NICE-2/3/4 zero-filled. Detects volumetric scenes → switches to `volpath`/`volpathmis`
- [ ] 8.3 Implement `CyclesAdapter(RendererAdapter)` — wraps Cycles pass system. Maps `PASS_DIFFUSE_DIRECT`, `PASS_GLOSSY_DIRECT`, `PASS_TRANSMISSION_DIRECT`, `PASS_VOLUME_DIRECT` to unified AOV format. Full pass coverage, no zero-filling
- [ ] 8.4 Define `UNIFIED_AOV_CHANNELS` dict in config — maps pass names to channel counts. Total: 22 channels (albedo:3, normal:3, depth:1, position:3, diffuse_direct:3, glossy_direct:3, transmission_direct:3, volume_direct:3)
- [ ] 8.5 Update `src/omen/aov.py` `pack_aux_buffer()` — accept unified AOV dict (22ch) and pack into (H, W, 22) tensor. Backward-compatible with old 10ch format via shape detection
- [ ] 8.6 Update `tile_fingerprint.mojo` — change `AUX_CH = 10` to `AUX_CH = 22`. Add per-pass variance features: transmission_var for caustic detection, volume_var for volumetric detection. May increase `FP_DIM` beyond 23 (update routing.py accordingly)
- [ ] 8.7 Rebuild Mojo kernel after AUX_CH/FP_DIM change — `mojo build --emit shared-lib` for tile_fingerprint.so. Update `src/omen/kernels/__init__.py` to load new .so
- [ ] 8.8 Update `_render_with_aov()` in `src/omen/modes/denoiser.py` — use `MitsubaAdapter` instead of direct Mitsuba calls. Add `integrator_type` parameter for volpath scenes
- [ ] 8.9 Add `AOV_PASS_TIER` enum in config — MUST (albedo, normal, depth), NICE-1 (position), NICE-2 (diffuse_direct, glossy_direct), NICE-3 (transmission_direct), NICE-4 (volume_direct, volume_scatter)
- [ ] 8.10 Wire scene_graph material types into caustic preservation — when `scene_graph["materials"]["types"]` contains `"dielectric"`, flag tiles near object's screen projection as "caustic expected" even if transmission_direct is zero-filled
- [ ] 8.11 Add random AOV pass dropout to training loop — 20% of steps drop NICE-2/3/4, 10% drop all NICE. Mitsuba permanently drops NICE-2/3/4 (zero-filled by adapter). Model learns scene_graph fallback

### Nabla Verified API (gaps resolved)

- [ ] 8.12 **Pixel Shuffle**: NOT in Nabla — implement manually via `nb.reshape` + `nb.transpose`. Example: (B,H,W,C*4) → (B,H,2,W,2,C) → transpose → (B,H*2,W*2,C). Verify autograd flows through reshape+transpose
- [ ] 8.13 **nb.concatenate / nb.stack**: CONFIRMED working. `nb.concatenate([a, b], axis=0)` and `nb.stack([a, b], axis=0)` both available. U-Net skip connections will use `nb.concatenate([enc_feat, dec_feat], axis=-1)`
- [ ] 8.14 **F.interpolate()**: Used in current decoder. CONFIRMED in Nabla nn.functional. Verify supports size parameter for upsample
- [ ] 8.15 **nb.conv2d filter layout HWIO**: CONFIRMED. Filter init: `F.he_normal((K_h, K_w, C_out, C_in))`. Same for all U-Net conv blocks

- [ ] 8.1 Expand `_AOV_SPEC` in `src/omen/modes/denoiser.py` from 3 passes (albedo, normal, depth) to 7+ passes: add `diffuse_direct`, `glossy_direct`, `transmission_direct`, `volume_direct`
- [ ] 8.2 Update `src/omen/aov.py` `pack_aux_buffer()` — expand from (H,W,10) to handle 7+ AOV channels. Keep backward compatible with old 3-pass format via shape detection
- [ ] 8.3 Update `tile_fingerprint.mojo` — expand 23-dim fingerprint to include per-pass variance features (diffuse vs glossy vs transmission variance). Dielectric expert sees high transmission variance → preserves caustics
- [ ] 8.4 Update `src/omen/kernels/aov_pack.py` and `aov_pack.mojo` — pack extended AOV buffer with pass-level separation. Each pass gets its own variance estimate
- [ ] 8.5 Add `AOV_PASS_TIER` enum in config — MUST (albedo, normal, depth), NICE-1 (diffuse_direct, glossy_direct), NICE-2 (transmission_direct), NICE-3 (volume_direct, volume_scatter), NICE-4 (SSS, emission, UV, motion)
- [ ] 8.6 Implement graceful degradation in `src/omen/aov.py` — `read_all_aov()` checks which passes are available, zero-fills missing NICE passes, logs warning. MUST passes missing → raise error
- [ ] 8.7 Update `src/omen/modes/denoiser.py` `_render_with_aov()` — accept optional `aov_passes` parameter. Default: all passes. User can disable NICE passes via config. MUST passes always rendered
- [ ] 8.8 Wire scene_graph material types into caustic preservation — when `scene_graph["materials"]["types"]` contains `"dielectric"`, flag tiles near that object's screen projection as "caustic expected" even if transmission_direct pass is missing
- [ ] 8.11 Add random AOV pass dropout to training loop — 20% of steps drop NICE-2/3/4, 10% drop all NICE. Mitsuba permanently drops NICE-2/3/4 (zero-filled by adapter). Model learns scene_graph fallback

## 9. Online Training Data Generator

- [ ] 9.1 Implement `TrainingDataGenerator.__init__()` — accepts resolution (default 1920x1080), gt_spp (default 256), noisy_spp (default 4), gpu flag, save_images toggle (default False)
- [ ] 9.2 Implement `train_step_online()` — core diffusion-like training loop: render GT at full HD + high SPP → encode to target_latent, render noisy at same resolution + low SPP → encode with scene_graph to noisy_latent, decoder predicts noise/residual, compute all losses (JEPA + denoise + SIGReg), backprop, free both images from memory. NO disk saves unless save_images=True
- [ ] 9.3 Implement multi-camera training — `train_step_online()` iterates all camera positions, running one train_step per camera. Each camera gets independent noisy/GT renders
- [ ] 9.4 Implement `train_animation_sequence()` — renders temporal frames from all 4 animation channels (camera, mesh, material, light), feeds consecutive frame pairs to ARPredictor for temporal loss. Each frame: render GT + noisy → encode → predict next latent → loss → backprop → free
- [ ] 9.5 Implement `--save-images` toggle — when enabled, saves rendered pairs to output_dir as .exr/.png for debugging. Default OFF (online-only, images freed after loss computation)
- [ ] 9.6 Integrate with `StratifiedReplayBuffer` — scene_graph hash keys per-scene sub-buffers; `train_step_online()` adds (noisy_latent, target_latent) to buffer, optionally samples replay pairs for mixed training

## 10. CLI Entry Point

- [ ] 10.1 Add `__main__.py` or update `scenes.py` with argparse CLI: `--scene`, `--spp`, `--gt-spp`, `--noisy-spp`, `--resolution`, `--count`, `--output`, `--list`, `--camera`, `--animate`, `--animate-type` (camera|mesh|material|light|all), `--save-images`
- [ ] 10.2 Implement `--list` — print SCENE_REGISTRY names + descriptions
- [ ] 10.3 Implement `--save-images` toggle — when set, saves renders to output_dir as .exr/.png for inspection. Default: images are NOT saved (online training only)
- [ ] 10.4 Implement `--animate` flag — render animation frames for temporal training
- [ ] 10.5 Implement `--animate-type` flag — select which animation channels to render (camera, mesh, material, light, or all)
- [ ] 10.6 Implement `--camera all` flag — render/train from all camera positions

## 11. Validation

- [ ] 11.1 Verify all 5 scenes render at 64spp without errors (CPU + GPU)
- [ ] 11.2 Verify scene_graph metadata has correct shapes for SceneGraphEncoder
- [ ] 11.3 Verify decoder outputs residual/noise map (same spatial dims as input, 3-channel)
- [ ] 11.4 Verify `clean = noisy - predicted_noise` produces correct denoised output
- [ ] 11.5 Verify TrainingDataGenerator produces valid (noisy, clean) pairs
- [ ] 11.6 Verify all 4 animation types produce valid temporal sequences (camera, mesh, material, light)
- [ ] 11.7 Verify animation frame-to-frame coherence (no sudden jumps, smooth parameter interpolation)
- [ ] 11.8 Verify caustic preservation — render Veach scene (glass sphere), denoise, compare caustic region PSNR against GT. Caustics MUST NOT be removed
- [ ] 11.9 Verify volumetric preservation — render Foggy Corridor, denoise, compare fog scattering region against GT. Volumetric scattering MUST NOT be flattened
- [ ] 11.10 Verify graceful degradation — render with only MUST passes (albedo, normal, depth), denoise, verify output quality degrades gracefully (no crashes, no NaN, reasonable quality loss < 2dB PSNR)
- [ ] 11.11 Verify AOV pass dropout training — model trained with random NICE pass dropout still produces acceptable denoised output when all passes are available at inference
- [ ] 11.12 Run full test suite to ensure no regressions

## 12. Implementation Gaps Checklist

> Hard blockers discovered by reading actual source code. Each gap MUST be resolved before or during the related section. Failing to address these causes runtime crashes, wrong shapes, or silent data loss.

### Nabla API Gaps (blocks 7, 8, 9)

- [ ] 12.1 **Nabla has no Pixel Shuffle / depth_to_space** — D6 specifies Pixel Shuffle upsampling. Nabla nn module has: Module, Linear, LayerNorm, Embedding, MultiHeadAttention, Sequential. No Pixel Shuffle. Must implement manually: reshape (B,H,W,C*r*r) to (B,H*r,W*r,C) via nb.reshape + nb.transpose. Verify works with autograd before building decoder.
- [ ] 12.2 **Nabla nb.conv2d filter layout is HWIO** — Confirmed in RenderFeatureEncoder (3,3,4,32). Decoder U-Net conv blocks must follow same layout. Filter init uses F.he_normal((K_h, K_w, C_out, C_in)) NOT PyTorch (C_out, C_in, K_h, K_w).
- [ ] 12.3 **Nabla nb.conv2d_transpose exists** — Confirmed in current Decoder. Filter layout (K_h, K_w, C_out, C_in). But we are replacing Conv2dTranspose with Pixel Shuffle + Conv, so less critical.
- [ ] 12.4 **Nabla F.interpolate() exists** — Used in current decoder for resize-to-target. Verify it supports upsampling sizes needed for U-Net.
- [ ] 12.5 **Nabla tensor concatenation** — U-Net skip connections need nb.concat([encoder_feat, decoder_feat], axis=-1). Verify nb.concat or nb.concatenate exists and works with autograd.
- [ ] 12.6 **Nabla nb.pad** — Used in SceneGraphEncoder. Confirmed working. No gap.
- [ ] 12.7 **Nabla nb.topk** — Used in ExpertGroup for top-k routing. Confirmed working. No gap.

### Mitsuba AOV Gaps (blocks 2-6, 8)

- [ ] 12.8 **VERIFIED: Mitsuba 3 has NO per-bounce light path passes** — Mitsuba AOV integrator only provides surface properties: albedo, sh_normal, dd.y, position. Per-bounce passes (diffuse_direct, glossy_direct, transmission_direct, volume_direct) are Cycles-specific. RESOLVED by D8 renderer adapter: MitsubaAdapter zero-fills per-bounce passes, scene_graph provides fallback knowledge. No separate renders needed.
- [ ] 12.9 **VERIFIED: Mitsuba volpath/volpathmis exist** — Both documented. volpathmis provides spectral MIS for spectrally varying extinction. MitsubaAdapter.integrator_for_scene() detects volumetric scenes and switches to volpath. Null BSDF + thin dielectric get special handling. RESOLVED by D8 adapter.
- [ ] 12.10 **_channel_offset() only has 3 mappings** — albedo:0, normal:3, depth:6. Adding new passes requires new offsets. But if Mitsuba does not support per-bounce AOVs, this gap may be moot.
- [ ] 12.11 **pack_aux_buffer() hardcodes (H, W, 10)** — albedo(3)+normal(3)+depth(1)+material_id(1)+motion(2)=10. Adding 4 new passes changes total channel count and pack order.

### Mojo Kernel Gaps (blocks 8)

- [ ] 12.12 **tile_fingerprint.mojo hardcoded AUX_CH = 10** — Expanding AOV passes changes input channel count. Mojo kernel must be recompiled with new AUX_CH. Fingerprint computation only uses channels 0-6 currently. New pass channels need fingerprint features (transmission_variance for caustic detection).
- [ ] 12.13 **tile_fingerprint.mojo hardcoded FP_DIM = 23** — Adding per-pass variance features changes fingerprint dimension. Cascades to routing.py FINGERPRINT_DIM = 23 and MoE router input. ALL downstream consumers must update.
- [ ] 12.14 **Mojo kernel recompilation** — Any change to AUX_CH or FP_DIM requires mojo build --emit shared-lib. The .so must be rebuilt and kernels/__init__.py must load the new version. Plan for rebuild step.

### Signature / API Breaking Changes (blocks 7, 9)

- [ ] 12.15 **OmenJEPA.decode(latent, height, width) must become decode(latent, noisy_image)** — Current signature at jepa.py:76-78 passes (latent, height, width). New decoder needs (latent, noisy_image). Breaking change to public API. All callers must update: jepa_inference.py, trainer/core.py, test files.
- [ ] 12.16 **Decoder.forward(latent, height, width) must become forward(latent, noisy_image)** — Same breaking change. Decoder must accept noisy image tensor, not H/W integers.
- [ ] 12.17 **ConfidenceHead.forward(latent, height, width) same signature issue** — If confidence is computed AFTER denoising, it needs denoised output, not height/width. Check if needs updating.
- [ ] 12.18 **jepa_inference.py denoise flow** — Current: encode then decode(latent,H,W) then return rgba. New: encode then predict_noise(latent,noisy) then return noisy minus noise. denoise() method signature and return type both change.

### SceneGraphEncoder Shape Gaps (blocks 2-6)

- [ ] 12.19 **geom_linear = nn.Linear(6, 64) expects exactly 6 features** — Scene graph geometry must produce 6 features (face center xyz + normal xyz). Scene builders must compute face centers and normals from vertex data.
- [ ] 12.20 **mat_linear = nn.Linear(5, 64) expects exactly 5 material params** — Each material must produce 5-element vector (e.g., diffuse RGB + roughness + IOR). Different material types need different encoding schemes.
- [ ] 12.21 **light_linear = nn.Linear(7, 64) expects exactly 7 light params** — Each light must produce 7-element vector (position xyz + color rgb + intensity).
- [ ] 12.22 **No volume encoder head** — Foggy Corridor adds volume params (sigma_t, albedo, g). Current SceneGraphEncoder has no volume_linear head. Either add new head or pack volume params into existing features.

### Training Pipeline Gaps (blocks 9)

- [ ] 12.23 **Current trainer has NO decoder loss** — trainer/core.py only computes latent MSE + SIGReg. Adding decoder noise prediction loss (MSE(predicted_noise, gt - noisy)) is entirely new code. Must wire GT pixel data into training step.
- [ ] 12.24 **GT and noisy images must be numpy to Nabla tensor** — Renders produce numpy arrays. Decoder expects Nabla tensors. Must convert with nb.array() or similar. Verify Nabla creates tensors from numpy without copy overhead.
- [ ] 12.25 **Full HD images are 25M values (1920x1080x3)** — Backpropping through U-Net decoder at full HD uses significant GPU memory. May need gradient checkpointing or tiled processing. Nabla may not support gradient checkpointing natively — verify.
- [ ] 12.26 **Random AOV pass dropout mechanism** — Must randomly zero-fill NICE passes per training step. Implementation goes in training loop, not AOV reader. Need a per-step mask applied to the AOV buffer channels.

### Existing Code Reuse (confirmed working, no gaps)

- [ ] 12.27 **MLASkipCompress / MLASkipReconstruct** — Already exists at src/omen/model/mla_skip.py. Reuse directly for decoder U-Net skip compression.
- [ ] 12.28 **EpisodicCorrection module** — Already exists at src/omen/model/episodic.py. Already wired into OmenJEPA.
- [ ] 12.29 **ARPredictor module** — Already exists at src/omen/model/arpredictor.py. Already wired into OmenJEPA.
- [ ] 12.30 **nb.conv2d functional API** — Confirmed working in RenderFeatureEncoder. HWIO filter layout. Reuse for U-Net conv blocks.
