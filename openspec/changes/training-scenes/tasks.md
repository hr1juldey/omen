## 1. Scene Infrastructure

> Spec: Cornell Box scene builder, Veach Ajar Door scene builder, Shaderball scene builder, Studio Product scene builder, Foggy Corridor scene builder

- [ ] 1.1 Add `_tf()` helper and `_build_sensor_multi()` to `src/omen/scenes.py` — multi-camera sensor builder that places N cameras at configurable positions around the scene, returning a list of (camera_name, sensor) pairs
- [ ] 1.2 Add `_build_scene_graph()` helper — extracts geometry vertices, material params, light params from scene definition dicts into numpy arrays matching `SceneGraphEncoder` input format (geom_linear expects 6 features, mat_linear expects 5, light_linear expects 7 — see §12.19-12.22)
- [ ] 1.3 Add `SCENE_REGISTRY` dict mapping scene names to their builder functions
- [ ] 1.4 Add `SceneAnimation` class — takes a base scene dict + a list of animation channels (camera, mesh, material, light), generates per-frame scene dicts by interpolating parameters

## 2. Cornell Box

> Spec: Cornell Box scene builder — scenarios "renders without error", "scene_graph has correct structure"

- [ ] 2.1 Build `build_cornell_box()` — 6-wall box room: red left wall (diffuse, 0.63,0.065,0.05), green right wall (diffuse, 0.14,0.45,0.091), white floor/ceiling/back (diffuse, 0.725,0.71,0.68), tall box and short box on floor (white diffuse)
- [ ] 2.2 Place area light on ceiling — rectangle emitter (0.8,0.8,0.8) at y=1.325
- [ ] 2.3 Place 5 cameras: front, left-45, right-45, top-down, close-up on tall box
- [ ] 2.4 Build scene_graph metadata — geometry (12 vertices from 6 walls + 2 boxes), materials (3 types), lights (1 area). Verify `scene_graph["materials"]["params"]` has at least 3 rows and `scene_graph["lights"]["params"]` has exactly 1 row (spec: Cornell Box scene_graph has correct structure)
- [ ] 2.5 Camera animation: 12-frame orbit around scene center at fixed radius and height
- [ ] 2.6 Mesh animation: 8-frame sequence rotating the tall box 90°, translating the short box across the floor
- [ ] 2.7 Material animation: 6-frame sequence shifting the red wall color toward orange (0.63→0.8), green wall toward teal (0.14→0.3)
- [ ] 2.8 Light animation: 8-frame sequence dimming the area light from full (1.0) to half (0.5) and shifting color temperature from neutral to warm

## 3. Veach Ajar Door

> Spec: Veach Ajar Door scene builder — scenarios "renders with multiple light types", "scene_graph lists all BSDF types"

- [ ] 3.1 Build `build_veach_ajar()` — dark room (diffuse black walls) with slightly open door gap on one wall
- [ ] 3.2 Place glass sphere (dielectric, ior=1.5), metal sphere (conductor, Au), matte sphere (diffuse, white) on floor
- [ ] 3.3 Place 3 lights: point light (warm, through door gap), spot light (from above), area light (behind camera)
- [ ] 3.4 Place 5 cameras: through door gap, side view, above, close on glass sphere, close on metal sphere
- [ ] 3.5 Build scene_graph metadata — 3 material types (dielectric, conductor, diffuse), 3 light types (point, spot, area). Verify `scene_graph["materials"]["types"]` contains "dielectric", "conductor", "diffuse" and `scene_graph["lights"]["types"]` contains "point", "spot", "area" (spec: Veach scene_graph lists all BSDF types)
- [ ] 3.6 Camera animation: 10-frame dolly moving from far corridor through the door gap into the room
- [ ] 3.7 Mesh animation: 8-frame sequence opening the door from 5° to 45°, moving glass sphere 0.3 units to the right
- [ ] 3.8 Material animation: 6-frame sequence changing glass IOR from 1.3 to 1.8 (visible refraction shift), metal roughness from 0.02 to 0.3
- [ ] 3.9 Light animation: 8-frame sequence moving point light from behind door to room center, dimming spot light from 80 to 20

## 4. Shaderball

> Spec: Shaderball scene builder — scenarios "renders all material types", "scene_graph has comprehensive material coverage"

- [ ] 4.1 Build `build_shaderball()` — ground plane (checkerboard roughplastic), 5 spheres in a row: conductor (mirror), roughconductor (Cu, α=0.15), plastic (SSS-like, skin tone), roughplastic (clay, α=0.2), dielectric (glass, ior=1.5)
- [ ] 4.2 Place area light above + constant environment emitter (0.4,0.4,0.45)
- [ ] 4.3 Place 4 cameras: front, 45-degree angle, top-down, close-up on material row
- [ ] 4.4 Build scene_graph metadata — 5 material types, 1 area light + 1 env light. Verify `scene_graph["materials"]["types"]` contains at least 5 distinct BSDF types including "conductor", "roughconductor", "plastic", "roughplastic", "dielectric" (spec: Shaderball scene_graph has comprehensive material coverage)
- [ ] 4.5 Camera animation: 12-frame circular orbit around the material row at 45° elevation
- [ ] 4.6 Mesh animation: 8-frame sequence scaling each sphere from 0.5x to 1.5x size (one at a time), plus a bounce on the central sphere
- [ ] 4.7 Material animation: 10-frame sequence sweeping roughness on roughconductor from 0.0 to 0.5, roughplastic from 0.1 to 0.4, then cycling plastic IOR from 1.3 to 1.7
- [ ] 4.8 Light animation: 8-frame sequence rotating area light position 180° around the spheres (simulating studio light sweep)

## 5. Studio Product

> Spec: Studio Product scene builder — scenarios "renders with 3-point lighting", "scene_graph lists conductor and roughplastic"

- [ ] 5.1 Build `build_studio_product()` — 3 objects: gold sphere (roughconductor Au, α=0.05), copper cylinder (roughconductor Cu, α=0.12), matte vase (roughplastic, α=0.25) on a dark ground plane
- [ ] 5.2 Place 3-point studio lighting: key area light (warm, 45° right-above), fill area light (cool, 45° left), rim area light (behind/above)
- [ ] 5.3 Place 4 cameras: product front, product 3/4 view, overhead, low-angle hero
- [ ] 5.4 Build scene_graph metadata — roughconductor + roughplastic materials, 3 area lights. Verify `scene_graph["materials"]["types"]` contains "roughconductor" and "roughplastic", `scene_graph["lights"]["types"]` contains "area" at least 3 times (spec: Studio scene_graph lists conductor and roughplastic)
- [ ] 5.5 Camera animation: 12-frame turntable orbit at 3/4 view height (full 360° sweep)
- [ ] 5.6 Mesh animation: 8-frame sequence lifting gold sphere up 0.5 units, rotating copper cylinder 180°, scaling vase from 0.8x to 1.2x
- [ ] 5.7 Material animation: 6-frame sequence changing gold roughness 0.02→0.2, copper roughness 0.1→0.3, vase color shift from matte gray to warm terracotta
- [ ] 5.8 Light animation: 8-frame sequence dimming key light from 100% to 30%, brightening fill to compensate, shifting rim light color from neutral to blue

## 6. Foggy Corridor

> Spec: Foggy Corridor scene builder — scenarios "renders with volumetric scattering", "scene_graph includes volume metadata"

- [ ] 6.1 Build `build_foggy_corridor()` — L-shaped corridor (diffuse gray walls), null-BSDF volume boundary (homogeneous medium, σ_t=0.2, albedo=0.9, Henyey-Greenstein g=0.3)
- [ ] 6.2 Place point light at corridor junction, spot light at one end
- [ ] 6.3 Place 4 cameras: corridor entrance, junction looking both ways, down the long arm
- [ ] 6.4 Build scene_graph metadata — diffuse + null BSDF, volume params (σ_t, albedo, g), 2 lights. Verify `scene_graph["materials"]["types"]` contains "null" and scene_graph includes "volume" key with sigma_t and albedo (spec: Foggy corridor scene_graph includes volume metadata)
- [ ] 6.5 Camera animation: 10-frame walkthrough from corridor entrance to junction to end of long arm
- [ ] 6.6 Mesh animation: 6-frame sequence moving a diffuse box obstacle along the corridor (blocking/unblocking light path)
- [ ] 6.7 Material animation: 8-frame sequence changing wall color from gray to warm white, floor from dark to checkerboard
- [ ] 6.8 Light animation: 8-frame sequence varying fog density (σ_t 0.05→0.5), moving point light along corridor, spot light cone narrowing from 45° to 15°

## 7. Decoder Redesign (Residual Noise Predictor)

> Spec: Decoder is a residual noise predictor — scenarios "outputs residual not full image", "uses U-Net with skip connections", "no checkerboard artifacts"

- [ ] 7.1 Rewrite `src/omen/model/decoder.py` — replace Conv2dTranspose full-image reconstruction with U-Net residual noise predictor. Decoder takes (jepa_latent, noisy_image) and outputs a noise/residual map. `clean = noisy - predicted_noise` (spec: Decoder outputs residual, not full image)
- [ ] 7.2 Implement U-Net encoder path — Conv blocks that downsample noisy image, extract multi-scale features (64→128→256 channels, 4 downsample stages) (spec: Decoder uses U-Net with skip connections)
- [ ] 7.3 Implement JEPA latent injection at bottleneck — project 1024-dim JEPA latent to spatial feature map and inject at U-Net bottleneck (conditioning, like diffusion model cross-attention) (spec: Decoder uses U-Net with skip connections — "1024-dim JEPA latent SHALL be injected at the bottleneck")
- [ ] 7.4 Implement U-Net decoder path with skip connections — upsample + concat encoder features at each resolution, conv blocks, output 3-channel residual map (spec: Decoder uses U-Net with skip connections)
- [ ] 7.5 Use Pixel Shuffle (or DySample) upsampling — NOT Conv2dTranspose. Per `docs/research/latent_decoder_and_rendering_survey.md` §1.7, Conv2dTranspose causes checkerboard artifacts (spec: No checkerboard artifacts from upsampling)
- [ ] 7.6 Add MLA-style compression for skip connections at higher resolutions — per `docs/research/deepseek-technical-survey.md`, compress 64-channel full-res features from ~500MB to ~50MB
- [ ] 7.7 Update `src/omen/jepa_inference.py` — change denoise flow from `encode → decode(latent) → return full RGBA` to `encode → predict_noise(latent, noisy) → return noisy - noise`
- [ ] 7.8 Update `src/omen/model/jepa.py` — change `decode()` to pass noisy image to decoder: `self.decoder(latent, noisy_image)` instead of `self.decoder(latent, height, width)`
- [ ] 7.9 Wire decoder loss into training — in `train_step_online()`, compute `residual = gt - noisy`, then `denoise_loss = MSE(predicted_noise, residual)` (spec: Online training step — "compute decoder noise prediction loss: MSE(predicted_noise, gt - noisy)")

### Nabla API Smoke Tests (MUST pass before building decoder)

> Spec: Nabla API verification for U-Net decoder — scenarios "Pixel Shuffle autograd verification", "skip connection concatenation with autograd", "compiled training step uses functional API", "optimizer initialization order"

- [ ] 7.10 **Verify Pixel Shuffle autograd** — write standalone test: `nb.reshape(B,H,W,C*4)` → `nb.reshape(B,H,2,W,2,C)` → `nb.transpose` → `nb.reshape(B,H*2,W*2,C)`. Verify output shape and that `nb.value_and_grad` computes gradients through the operation. (spec: Pixel Shuffle autograd verification)
- [ ] 7.11 **Verify skip connection concatenation** — write standalone test: `nb.concatenate([encoder_feat, decoder_feat], axis=-1)` with both tensors requiring grad. Verify output channels = enc_ch + dec_ch, autograd flows to both branches. (spec: Skip connection concatenation with autograd)
- [ ] 7.12 **Verify conv2d stride=2 downsampling** — write standalone test: `nb.conv2d(x, w, stride=(2,2))` produces output at half spatial resolution. Verify autograd flows. Filter layout HWIO: `(K_h, K_w, C_in, C_out)`. (spec: Decoder uses U-Net with skip connections — encoder path downsamples)
- [ ] 7.13 **Verify nb.avg_pool2d / nb.max_pool2d autograd** — write standalone test confirming pooling operations work with gradient computation. U-Net encoder uses pooling. (spec: Decoder uses U-Net with skip connections — encoder path downsamples)
- [ ] 7.14 **Verify nb.value_and_grad with multi-argument function** — write standalone test: `nb.value_and_grad(loss_fn, argnums=0)(model, noisy_img, gt_img)` where loss_fn takes 3 arguments. Verify gradients computed for argnums=0 only. (spec: Compiled training step uses functional API)
- [ ] 7.15 **Verify @nb.compile with functional optimizer** — write standalone test: `@nb.compile` decorated function using `nb.value_and_grad` + `nb.nn.optim.adamw_update()`. Confirm this works and that `loss.backward()` does NOT work inside compiled functions. (spec: Compiled training step uses functional API)
- [ ] 7.16 **Verify optimizer init order** — write standalone test: create `nb.nn.optim.AdamW(model)` while model is in train mode. Verify no pytree mismatch. Then create `adamw_init` after `model.eval()`. (spec: Optimizer initialization order)
- [ ] 7.17 **Verify nb.Tensor.from_dlpack zero-copy from numpy** — write standalone test: `nb.Tensor.from_dlpack(numpy_array)` produces tensor with correct shape/dtype. Verify memory layout compatibility for rendered image arrays (H,W,3) float32. (spec: Online training step — renders produce numpy, decoder expects tensors)

## 8. Renderer Adapter + Extended AOV + Graceful Degradation

> Spec: Renderer adapter interface, Extended AOV passes, Graceful degradation for missing AOV passes

### Renderer Adapter (D8)

- [ ] 8.1 Create `src/omen/render_adapter.py` with `RendererAdapter` abstract base class — methods: `render()`, `get_aov()`, `list_available_passes()`, `integrator_for_scene()`. (spec: Renderer adapter interface — adapter SHALL provide render, get_aov, list_available_passes, integrator_for_scene)
- [ ] 8.2 Implement `MitsubaAdapter(RendererAdapter)` — wraps Mitsuba render + AOV integrator. MUST passes (albedo, normal, depth) via `aov:sh_normal,dd.y` spec. NICE-1 (position) via `position:pos`. NICE-2/3/4 zero-filled. Detects volumetric scenes (scene_graph volume key) → switches to `volpath`/`volpathmis`. (spec: MitsubaAdapter zero-fills per-bounce passes)
- [ ] 8.3 Implement `CyclesAdapter(RendererAdapter)` — wraps Cycles pass system. Maps `PASS_DIFFUSE_DIRECT`, `PASS_GLOSSY_DIRECT`, `PASS_TRANSMISSION_DIRECT`, `PASS_VOLUME_DIRECT` to unified AOV format. Full pass coverage, no zero-filling. (spec: CyclesAdapter provides full pass coverage)
- [ ] 8.4 Update `src/omen/modes/denoiser.py` `_render_with_aov()` — use `MitsubaAdapter` instead of direct Mitsuba calls. Accept `RendererAdapter` parameter for renderer selection. Pass `integrator_type` from adapter for volpath scenes. (spec: Adapter returns unified AOV regardless of renderer)

### Unified AOV Format

- [ ] 8.5 Define `UNIFIED_AOV_CHANNELS` dict in config — maps pass names to channel counts: `albedo:3, normal:3, depth:1, position:3, diffuse_direct:3, glossy_direct:3, transmission_direct:3, volume_direct:3`. Total: 22 channels. (spec: Extended AOV capture — returned AOV dict SHALL contain all 7+ pass keys)
- [ ] 8.6 Add `AOV_PASS_TIER` enum in config — `MUST` (albedo, normal, depth — missing → error), `NICE_1` (position), `NICE_2` (diffuse_direct, glossy_direct, transmission_direct), `NICE_3` (volume_direct, volume_scatter). (spec: Graceful degradation — MUST vs NICE tier classification)
- [ ] 8.7 Update `src/omen/aov.py` `pack_aux_buffer()` — accept unified AOV dict (22ch) and pack into (H, W, 22) tensor. Backward-compatible with old 10ch format via shape detection. (spec: Extended AOV capture — each pass SHALL be numpy array with shape (H,W,3))

### Graceful Degradation

- [ ] 8.8 Implement graceful degradation in `src/omen/aov.py` — `read_all_aov()` checks which passes are available: missing NICE passes → zero-fill + log warning; missing MUST passes → raise `ValueError` with pass name. (spec: Must passes missing → error; Nice passes missing → graceful degradation)
- [ ] 8.9 Wire scene_graph material types into caustic preservation — when `scene_graph["materials"]["types"]` contains `"dielectric"`, flag tiles near object's screen projection as "caustic expected" even if `transmission_direct` is zero-filled. (spec: Scene graph fallback for caustics without transmission pass)
- [ ] 8.10 Add random AOV pass dropout to training loop — 20% of steps drop NICE-2/NICE-3, 10% drop all NICE, 0% drop MUST. Implementation: per-step mask applied to AOV buffer channels before encoding. (spec: Random pass dropout during training)

### Mojo Kernel Updates

- [ ] 8.11 Update `tile_fingerprint.mojo` — change `AUX_CH = 10` to `AUX_CH = 22`. Add per-pass variance features: transmission_var for caustic detection, volume_var for volumetric detection. May increase `FP_DIM` beyond 23 (update `routing.py` `FINGERPRINT_DIM` accordingly).
- [ ] 8.12 Update `src/omen/kernels/aov_pack.py` and `aov_pack.mojo` — pack extended 22-channel AOV buffer with pass-level separation. Each pass gets its own variance estimate.
- [ ] 8.13 Rebuild Mojo kernel after AUX_CH/FP_DIM change — `mojo build --emit shared-lib` for tile_fingerprint.so. Update `src/omen/kernels/__init__.py` to load new .so.

### Nabla Verified API (gaps resolved)

- [ ] 8.14 **Pixel Shuffle**: NOT in Nabla — implement manually via `nb.reshape` + `nb.transpose`. Example: `(B,H,W,C*4)` → `(B,H,2,W,2,C)` → transpose → `(B,H*2,W*2,C)`. Verified autograd in task 7.10.
- [ ] 8.15 **nb.concatenate**: CONFIRMED working. `nb.concatenate([a, b], axis=-1)` for U-Net skip connections. Verified in task 7.11.
- [ ] 8.16 **F.interpolate()**: Used in current decoder. CONFIRMED in Nabla nn.functional. Verify supports size parameter for arbitrary upscale in U-Net decoder path.
- [ ] 8.17 **nb.conv2d filter layout HWIO**: CONFIRMED. Filter init: `F.he_normal((K_h, K_w, C_in, C_out))`. Same layout for all U-Net conv blocks. Verified stride=2 in task 7.12.

## 9. Online Training Data Generator

> Spec: Training data generator — scenarios "Online training step (no disk I/O)", "Full HD ground truth render", "Debug save toggle", "Multi-camera training step"

- [ ] 9.1 Implement `TrainingDataGenerator.__init__()` — accepts resolution (default 1920x1080), gt_spp (default 256), noisy_spp (default 4), gpu flag, save_images toggle (default False). Create optimizer while model is in `train()` mode (spec: Optimizer initialization order). (spec: Full HD ground truth render)
- [ ] 9.2 Implement `train_step_online()` — core diffusion-like training loop: render GT at full HD + high SPP → encode to target_latent, render noisy at same resolution + low SPP → encode with scene_graph to noisy_latent, decoder predicts noise/residual, compute all losses (JEPA + denoise + SIGReg), backprop, free both images from memory. NO disk saves unless save_images=True. Convert numpy renders to Nabla tensors via `nb.Tensor.from_dlpack()`. (spec: Online training step — "render GT at 256 SPP at full HD, render noisy at 4 SPP, compute losses, backprop, free images, NOT write files")
- [ ] 9.3 Implement multi-camera training — `train_step_online()` iterates all camera positions, running one train_step per camera. Each camera gets independent random seeds for noisy renders. (spec: Multi-camera training step — "run one train_step per camera position, each step uses independent random seeds")
- [ ] 9.4 Implement `train_animation_sequence()` — renders temporal frames from all 4 animation channels (camera, mesh, material, light), feeds consecutive frame pairs to ARPredictor for temporal loss. Each frame: render GT + noisy → encode → predict next latent → loss → backprop → free
- [ ] 9.5 Implement `--save-images` toggle — when enabled, saves rendered pairs to output_dir as .exr/.png for debugging. Default OFF (online-only, images freed after loss computation). (spec: Debug save toggle — "each rendered pair SHALL be saved as .exr files to output_dir")
- [ ] 9.6 Integrate with `StratifiedReplayBuffer` — scene_graph hash keys per-scene sub-buffers; `train_step_online()` adds (noisy_latent, target_latent) to buffer, optionally samples replay pairs for mixed training

## 10. CLI Entry Point

> Spec: CLI entry point for scene rendering — scenarios "Render Cornell Box via CLI", "Run online training", "List available scenes"

- [ ] 10.1 Add `__main__.py` or update `scenes.py` with argparse CLI: `--scene`, `--spp`, `--gt-spp`, `--noisy-spp`, `--resolution`, `--count`, `--output`, `--list`, `--camera`, `--animate`, `--animate-type` (camera|mesh|material|light|all), `--save-images`
- [ ] 10.2 Implement `--list` — print SCENE_REGISTRY names + descriptions. (spec: List available scenes — "print all 5 scene names with brief descriptions")
- [ ] 10.3 Implement `--save-images` toggle — when set, saves renders to output_dir as .exr/.png for inspection. Default: images are NOT saved (online training only). (spec: Render Cornell Box via CLI — "render at 64spp and save to cornell.exr, print render time and output path")
- [ ] 10.4 Implement `--animate` flag — render animation frames for temporal training
- [ ] 10.5 Implement `--animate-type` flag — select which animation channels to render (camera, mesh, material, light, or all)
- [ ] 10.6 Implement `--camera all` flag — render/train from all camera positions

## 11. Validation

> Spec: All requirement scenarios — validate against spec acceptance criteria

- [ ] 11.1 Verify all 5 scenes render at 64spp without errors (CPU + GPU). (spec: Cornell Box "renders without error" — "rendering at 64spp SHALL produce an image with visible color bleeding"; Veach "renders with multiple light types"; Shaderball "renders all material types" — "no NaN or inf"; Studio "renders with 3-point lighting"; Foggy "renders with volumetric scattering" — "no NaN or black pixels")
- [ ] 11.2 Verify scene_graph metadata has correct shapes for SceneGraphEncoder. (spec: Cornell Box "scene_graph has correct structure" — "vertices shape (N,3) where N>0, materials at least 3 rows, lights exactly 1 row")
- [ ] 11.3 Verify decoder outputs residual/noise map (same spatial dims as input, 3-channel). (spec: Decoder outputs residual — "output SHALL have same spatial dimensions as noisy_image, 3-channel RGB residual")
- [ ] 11.4 Verify `clean = noisy - predicted_noise` produces correct denoised output. (spec: Decoder outputs residual — "noisy_image - output SHALL produce the denoised image")
- [ ] 11.5 Verify TrainingDataGenerator produces valid (noisy, clean) pairs. (spec: Online training step — "render GT at 256 SPP, render noisy at 4 SPP, encode both, compute loss, backprop, free images, NOT write files")
- [ ] 11.6 Verify all 4 animation types produce valid temporal sequences (camera, mesh, material, light)
- [ ] 11.7 Verify animation frame-to-frame coherence (no sudden jumps, smooth parameter interpolation)
- [ ] 11.8 Verify caustic preservation — render Veach scene (glass sphere), denoise, compare caustic region PSNR against GT. (spec: Caustic preservation via transmission pass — "caustic patterns SHALL be preserved, caustic region PSNR within 2dB of non-caustic region")
- [ ] 11.9 Verify volumetric preservation — render Foggy Corridor, denoise, compare fog scattering region against GT. (spec: Volumetric preservation via volume pass — "volumetric scattering SHALL be preserved, fog density SHALL NOT be flattened")
- [ ] 11.10 Verify graceful degradation — render with only MUST passes (albedo, normal, depth), denoise, verify output quality degrades gracefully (no crashes, no NaN, quality loss < 2dB PSNR). (spec: Nice passes missing → graceful degradation — "output quality SHALL degrade by less than 2dB PSNR")
- [ ] 11.11 Verify AOV pass dropout training — model trained with random NICE pass dropout still produces acceptable denoised output when all passes are available at inference. (spec: Random pass dropout during training)
- [ ] 11.12 Run full test suite to ensure no regressions

## 12. Implementation Gaps Checklist

> Hard blockers discovered by reading actual source code. Each gap MUST be resolved before or during the related section.

### Nabla API Gaps (blocks 7, 8, 9)

- [ ] 12.1 **Nabla has no Pixel Shuffle / depth_to_space** — D6 specifies Pixel Shuffle upsampling. Must implement manually: reshape `(B,H,W,C*r*r)` to `(B,H*r,W*r,C)` via nb.reshape + nb.transpose. Verified in task 7.10 before building decoder.
- [ ] 12.2 **Nabla nb.conv2d filter layout is HWIO** — Confirmed in RenderFeatureEncoder (3,3,4,32). Decoder U-Net conv blocks must follow same layout. Filter init uses `F.he_normal((K_h, K_w, C_in, C_out))` NOT PyTorch (C_out, C_in, K_h, K_w).
- [ ] 12.3 **Nabla nb.conv2d_transpose exists** — Confirmed in current Decoder. Filter layout `(K_h, K_w, C_out, C_in)`. But we are replacing Conv2dTranspose with Pixel Shuffle + Conv, so less critical.
- [ ] 12.4 **Nabla F.interpolate() exists** — Used in current decoder for resize-to-target. Verify it supports upsampling sizes needed for U-Net.
- [ ] 12.5 **Nabla tensor concatenation** — U-Net skip connections need `nb.concatenate([encoder_feat, decoder_feat], axis=-1)`. Verified in task 7.11.
- [ ] 12.6 **Nabla nb.pad** — Used in SceneGraphEncoder. Confirmed working. No gap.
- [ ] 12.7 **Nabla nb.topk** — Used in ExpertGroup for top-k routing. Confirmed working. No gap.

### Mitsuba AOV Gaps (blocks 2-6, 8)

- [ ] 12.8 **VERIFIED: Mitsuba 3 has NO per-bounce light path passes** — Mitsuba AOV integrator (`aov.cpp` AOVType enum) only provides 12 surface property types: Albedo, Depth, Position, UV, GeometricNormal, ShadingNormal, dPdU, dPdV, dUVdx, dUVdy, PrimIndex, ShapeIndex. Per-bounce passes (diffuse_direct, glossy_direct, transmission_direct, volume_direct) are Cycles-specific. RESOLVED by D8 renderer adapter: MitsubaAdapter zero-fills per-bounce passes, scene_graph provides fallback knowledge.
- [ ] 12.9 **VERIFIED: Mitsuba volpath/volpathmis exist** — Both documented in `src/integrators/`. volpath: standard volumetric path tracer. volpathmis: MIS for spectrally varying extinction (Miller et al. 2019). Null BSDF + thin dielectric get special handling. RESOLVED by D8 adapter `integrator_for_scene()`.
- [ ] 12.10 **_channel_offset() only has 3 mappings** — albedo:0, normal:3, depth:6. Adding new passes requires new offsets. Resolved by task 8.7 (rewrite pack_aux_buffer for unified 22ch).
- [ ] 12.11 **pack_aux_buffer() hardcodes (H, W, 10)** — albedo(3)+normal(3)+depth(1)+material_id(1)+motion(2)=10. Resolved by task 8.7 (unified 22ch format).

### Mojo Kernel Gaps (blocks 8)

- [ ] 12.12 **tile_fingerprint.mojo hardcoded AUX_CH = 10** — Expanding AOV passes changes input channel count to 22. Mojo kernel must be recompiled. Resolved by task 8.11.
- [ ] 12.13 **tile_fingerprint.mojo hardcoded FP_DIM = 23** — Adding per-pass variance features changes fingerprint dimension. Cascades to `routing.py` `FINGERPRINT_DIM = 23` and MoE router input. ALL downstream consumers must update. Resolved by task 8.11.
- [ ] 12.14 **Mojo kernel recompilation** — Any change to AUX_CH or FP_DIM requires `mojo build --emit shared-lib`. Resolved by task 8.13.

### Signature / API Breaking Changes (blocks 7, 9)

- [ ] 12.15 **OmenJEPA.decode(latent, height, width) must become decode(latent, noisy_image)** — Current signature at jepa.py:76-78 passes (latent, height, width). New decoder needs (latent, noisy_image). Breaking change to public API. All callers must update: jepa_inference.py, trainer/core.py, test files. Resolved by task 7.8.
- [ ] 12.16 **Decoder.forward(latent, height, width) must become forward(latent, noisy_image)** — Same breaking change. Decoder must accept noisy image tensor, not H/W integers. Resolved by task 7.1.
- [ ] 12.17 **ConfidenceHead.forward(latent, height, width) same signature issue** — If confidence is computed AFTER denoising, it needs denoised output, not height/width. Check if needs updating alongside task 7.8.
- [ ] 12.18 **jepa_inference.py denoise flow** — Current: encode then decode(latent,H,W) then return rgba. New: encode then predict_noise(latent,noisy) then return noisy minus noise. Resolved by task 7.7.

### SceneGraphEncoder Shape Gaps (blocks 2-6)

- [ ] 12.19 **geom_linear = nn.Linear(6, 64) expects exactly 6 features** — Scene graph geometry must produce 6 features (face center xyz + normal xyz). Scene builders must compute face centers and normals from vertex data. Addressed in task 1.2.
- [ ] 12.20 **mat_linear = nn.Linear(5, 64) expects exactly 5 material params** — Each material must produce 5-element vector (e.g., diffuse RGB + roughness + IOR). Different material types need different encoding schemes. Addressed in task 1.2.
- [ ] 12.21 **light_linear = nn.Linear(7, 64) expects exactly 7 light params** — Each light must produce 7-element vector (position xyz + color rgb + intensity). Addressed in task 1.2.
- [ ] 12.22 **No volume encoder head** — Foggy Corridor adds volume params (sigma_t, albedo, g). Current SceneGraphEncoder has no volume_linear head. Either add new head or pack volume params into existing features. Addressed in task 1.2.

### Training Pipeline Gaps (blocks 9)

- [ ] 12.23 **Current trainer has NO decoder loss** — trainer/core.py only computes latent MSE + SIGReg. Adding decoder noise prediction loss `MSE(predicted_noise, gt - noisy)` is entirely new code. Must wire GT pixel data into training step. Resolved by task 7.9.
- [ ] 12.24 **GT and noisy images must be numpy to Nabla tensor** — Renders produce numpy arrays. Decoder expects Nabla tensors. Convert with `nb.Tensor.from_dlpack()`. Verified in task 7.17.
- [ ] 12.25 **Full HD images are 25M values (1920x1080x3)** — Backpropping through U-Net decoder at full HD uses significant GPU memory. May need gradient checkpointing or tiled processing. Nabla may not support gradient checkpointing natively — verify.
- [ ] 12.26 **Random AOV pass dropout mechanism** — Must randomly zero-fill NICE passes per training step. Implementation goes in training loop, not AOV reader. Need a per-step mask applied to the AOV buffer channels. Resolved by task 8.10.

### Nabla Training API Gaps — from official examples (blocks 7, 9)

> Discovered from Nabla examples #6a (Transformer PyTorch-style), #11 (LoRA/QLoRA), #13 (CNN Training). These are NOT in the original gap list.

- [ ] 12.27 **@nb.compile requires nb.value_and_grad, NOT loss.backward()** — Inside compiled functions, the imperative `.backward()` / `.grad` path does NOT work. Must use functional `nb.value_and_grad(loss_fn, argnums=0)`. Omen's compiled training path MUST use functional API. Verified in task 7.15. (source: Nabla Transformer example #6a §6)
- [ ] 12.28 **Optimizer init order: train mode required** — `nb.nn.optim.AdamW(model)` must be created while `model.train()`. The optimizer snapshots pytree metadata including `_training`. Creating optimizer after `model.eval()` causes pytree mismatch on first `model.train()` call. Verified in task 7.16. (source: Nabla Transformer example #6a §4)
- [ ] 12.29 **model = optimizer.step() — reassignment required** — Nabla's lazy execution cannot mutate tensor data in-place. The updated model is returned from `optimizer.step()` and must be assigned: `model = optimizer.step()`. All training loops must use this pattern. (source: Nabla Transformer example #6a §5)
- [ ] 12.30 **nb.realize_all() for lazy execution batching** — Nabla is lazy by default. In training loops, batch-realize all tensors (loss, updated params, optimizer state) with `nb.realize_all(*tensors)` to force computation. Without this, operations queue indefinitely. (source: Nabla LoRA example #11 §2)
- [ ] 12.31 **nb.nn.optim.adamw_init() + adamw_update() for functional path** — Alternative to stateful `AdamW(model)`. `adamw_init(model)` returns optimizer state, `adamw_update(model, grads, state, lr)` returns (new_model, new_state). Required for `@nb.compile` training. (source: Nabla LoRA example #11 §2, Transformer #6a §6)
- [ ] 12.32 **Nabla has built-in LoRA/QLoRA via nb.nn.finetune** — `init_lora_adapter()`, `lora_linear()`, `quantize_nf4()`, `save_finetune_checkpoint()`, `load_finetune_checkpoint()`. Evaluate whether to replace manual LoRAManager with built-in for EpisodicCorrection. (source: Nabla LoRA example #11 §2-4)
- [ ] 12.33 **nb.nn.functional.cross_entropy_loss() exists** — Nabla has built-in loss functions. Check if `mse_loss()` or `l1_loss()` also exist. If not, keep manual `nb.mean(diff * diff)` for MSE. (source: Nabla Transformer example #6a §5)
- [ ] 12.34 **Nb.Tensor.from_dlpack(numpy_array) for data conversion** — Verified pattern: create numpy array → `nb.Tensor.from_dlpack(arr)`. Works for any numpy dtype. This is how rendered images enter Nabla. No copy overhead with compatible layouts. Verified in task 7.17. (source: Nabla CNN example #13 §2, LoRA #11 §1)

### Existing Code Reuse (confirmed working, no gaps)

- [ ] 12.35 **MLASkipCompress / MLASkipReconstruct** — Already exists at `src/omen/model/mla_skip.py`. Reuse directly for decoder U-Net skip compression (task 7.6).
- [ ] 12.36 **EpisodicCorrection module** — Already exists at `src/omen/model/episodic.py`. Already wired into OmenJEPA.
- [ ] 12.37 **ARPredictor module** — Already exists at `src/omen/model/arpredictor.py`. Already wired into OmenJEPA.
- [ ] 12.38 **nb.conv2d functional API** — Confirmed working in RenderFeatureEncoder. HWIO filter layout. Reuse for U-Net conv blocks.
