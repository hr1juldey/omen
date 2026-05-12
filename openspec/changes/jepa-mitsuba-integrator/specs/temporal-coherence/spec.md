## ADDED Requirements

### Background: JEPA as World Model (from LeWorldModel)

LeWorldModel (LeWM, Maes et al. 2026, co-authored by Yann LeCun) proves JEPA works as a world model with:
- **Two losses only**: next-embedding prediction + SIGReg (Gaussian regularizer, λ=0.09)
- **~18M parameters** (ViT-Tiny 5.5M + ARPredictor 10.8M + projections 1.6M), trains in hours on single GPU
- **48× faster planning** than foundation-model world models
- **Surprise detection**: reliably detects physically implausible events
- **Autoregressive rollout**: predicts N future frames from history window (H=3)

Omen replaces LeWM's "robot actions" with "scene deltas" (camera moves, object transforms, new emitters, light changes). The world model predicts what the next frame looks like WITHOUT path tracing. Only path-trace on surprise (unexpected scene change).

### Requirement: Autoregressive frame prediction using JEPA world model

Omen SHALL implement an autoregressive JEPA predictor that predicts clean frame output from 1spp dirty render + scene graph + temporal history. Prediction SHALL NEVER generate frames from nothing — always conditioned on actual render data + scene graph.

#### Scenario: Predict clean frame from 1spp dirty render + history

- **WHEN** rendering animation frame N (N > 0)
- **AND** 1spp dirty render available: `dirty = mi.render(scene, spp=1)` → `(H, W, 3)` (~30ms at 256×256)
- **AND** scene graph extracted: `sg = extract_scene_graph(scene)` (geometry, materials, lights)
- **AND** history buffer has H=3 previous frame latents: `history = [latent_{N-3}, latent_{N-2}, latent_{N-1}]`
- **AND** scene delta computed: `delta = compute_delta(frame_{N-1}, frame_N)`
- **THEN** encode dirty frame + scene graph: `current_latent = ViTEncoder.encode(dirty, sg)` → shape `(1, 192)`
- **AND** encode scene delta: `delta_emb = SceneDeltaEncoder(delta)` → shape `(1, 192)`
- **AND** predict: `predicted_latent = ARPredictor(history, current_latent, delta_emb)` → shape `(1, 192)`
  - ARPredictor is 6-layer ConditionalBlock transformer with AdaLN-zero conditioning
  - Input: concatenation `[history[-3:], current_latent]` → `(1, 4, 192)`
  - Conditioning: `delta_emb` modulates each layer via SiLU + Linear(192, 1152) → 6 modulation params
- **AND** decode: `clean_frame = Decoder(predicted_latent)` → `(H, W, 4)` RGBA
- **AND** total pipeline: 1spp render (~30ms) + encode (~5ms) + predict (~5ms) + decode (~5ms) = <50ms target

#### Scenario: History window management

- **WHEN** rendering animation frame N
- **THEN** maintain `CircularBuffer[Tensor]` of size `history_size=3` (configurable)
- **AND** on sequence start (frame 0):
  - Render 1spp → encode → denoise via JEPA → store latent as anchor
  - `history.push(latent_0)`
- **AND** on frame N > 0:
  - After prediction: `history.push(predicted_latent_N)`
  - Truncate to last H=3: `history = history[-3:]` (like LeWM's `emb[:, -HS:]`)
- **AND** on camera jump cut: `history.clear()` → render 4spp → denoise → new anchor

### Requirement: SceneDeltaEncoder for animation changes

Omen SHALL encode per-frame scene changes as structured "scene deltas" replacing LeWM's action encoder. Architecture: `Conv1d(delta_dim, smoothed, k=1)` → `MLP(smoothed → 4*192 → 192)` (155K params).

#### Scenario: Encode camera movement delta

- **WHEN** camera transforms between frames are available
- **THEN** compute: `T_delta = T_frame_N × inverse(T_frame_N-1)`
- **AND** extract translation delta (3 floats) + rotation quaternion delta (4 floats) = 7 floats
- **AND** flatten all deltas into `delta_tensor`:
  ```
  [camera(7), objects(N×8), lights(M×7), births(K×8), materials(P×5)]
  ```
- **AND** pass through Conv1d (smoothing) then MLP: `Linear(smoothed, 768) → SiLU → Linear(768, 192)`
- **AND** output: `delta_embedding` shape `(1, 192)`

#### Scenario: Encode fluid/smoke introduction (birth event)

- **WHEN** new volume emitter (fluid, smoke, fire) appears (not in previous frame)
- **THEN** detect via scene graph diff: new emitter in `scene.emitters()` list
- **AND** encode: `[type=volume(1), position(3), size(3), density(1)]` = 8 floats
- **AND** include in `delta_tensor` with birth flag
- **AND** flag as **high-surprise** (confidence=0, trigger full 4spp path trace)

#### Scenario: Encode light change delta

- **WHEN** light parameters change between frames
- **THEN** compute per-light: `[light_id, intensity_delta, color_delta(3), position_delta(3)]` = 7 floats
- **AND** if intensity change > 50%: flag medium-surprise (reduce prediction confidence)
- **AND** if new light appears: flag high-surprise (trigger full path trace)

#### Scenario: Encode material animation delta

- **WHEN** material parameters are animated
- **THEN** compute per-material: `[mat_id, param_deltas(4)]` = 5 floats
- **AND** include in delta_tensor

### Requirement: Surprise detection for unexpected scene changes

Omen SHALL implement surprise detection (based on LeWM) to identify frames where JEPA prediction is unreliable.

#### Scenario: Detect surprise via latent comparison

- **WHEN** JEPA predicts frame N latent AND actual render latent is available (periodic validation)
- **THEN** compute: `surprise = MSE(predicted_latent, actual_render_latent)`
- **AND** normalize relative to running average: `z_score = (surprise - mean) / std`
- **AND** if `z_score > 2.0` (configurable threshold):
  - Log: "Surprise detected at frame {N} (score: {surprise:.4f}, z: {z:.1f})"
  - Re-render at 4spp → JEPA denoise → replace predicted latent in history with actual
  - Mark subsequent predictions as lower confidence

#### Scenario: Auto-surprise for new scene elements

- **WHEN** scene graph diff shows structural changes:
  - New emitter (fluid, smoke, fire, light)
  - Deleted object
  - Material type change (diffuse → glass)
- **THEN** classify as high-surprise WITHOUT computing prediction
- **AND** trigger full 4spp path trace for this frame
- **AND** clear history buffer (old context invalid)
- **AND** restart prediction from this frame as new anchor

#### Scenario: Periodic validation (every 5 predicted frames)

- **WHEN** JEPA has predicted 5 consecutive frames without path tracing
- **THEN** render validation frame: `mi.render(scene, spp=1)` → encode → actual latent
- **AND** compare: `surprise = MSE(predicted_latent, actual_latent)`
- **AND** if surprise < threshold: prediction reliable, continue
- **AND** if surprise > threshold: prediction drifted → re-anchor with actual render, clear old history

#### Scenario: Camera jump cut detection

- **WHEN** camera transform delta exceeds threshold
- **THEN** check: translation > 1.0 unit OR rotation > 45° (0.785 rad)
- **AND** detect as jump cut
- **AND** clear history buffer
- **AND** render 4spp → JEPA denoise → new anchor
- **AND** log: "Jump cut detected at frame {N}, re-anchoring"

### Requirement: SIGReg loss for stable latent space (λ=0.09)

Omen SHALL implement SIGReg from LeWM to prevent representation collapse. Lambda=0.09 from lewm.yaml (NOT 1.0).

#### Scenario: Apply SIGReg during temporal training

- **WHEN** JEPA temporal training is running
- **THEN** compute prediction loss: `L_pred = MSE(predicted_latent, target_latent)`
- **AND** compute SIGReg loss: `L_sigreg = SIGReg(embeddings)` — Epps-Pulley statistic with 17 knots, 1024 projections on [0,3]
- **AND** total loss: `L = L_pred + 0.09 * L_sigreg` (λ=0.09 from lewm.yaml)
- **AND** SIGReg has ZERO learnable parameters
- **AND** optimizer: `NablaAdamW(lr=5e-5, weight_decay=1e-3)`, gradient clip=1.0, BF16 precision

### Requirement: Animation render pipeline with prediction

Omen SHALL implement animation pipeline that uses JEPA prediction for most frames, path tracing only on surprise. Target: 10-50× speedup.

#### Scenario: Render animation sequence

- **WHEN** rendering frames 0 through N
- **THEN** **Frame 0** (anchor):
  - `mi.render(scene, spp=1)` → encode → JEPA denoise → `latent_0` → `history.push(latent_0)`
- **AND** **Frames 1..N**:
  - `mi.render(scene, spp=1)` → dirty frame (ALWAYS render 1spp for exact geometry/occlusion)
  - Extract scene graph + compute scene delta
  - Encode dirty + scene graph → `current_latent`
  - `predicted_latent = ARPredictor(history[-3:], current_latent, delta_emb)`
  - Decode → clean frame output
  - If surprise: re-render 4spp → denoise → update history anchor
  - Else: `history.push(predicted_latent)`

#### Scenario: Speed benchmark (256×256, 100 frames)

- **WHEN** benchmarking Mode 4 animation
- **THEN** measure: full path trace ~2s/frame × 100 = 200s
- **AND** measure: predicted frames <5ms/frame
- **AND** target: ~10% frames path-traced (anchors + surprises), 90% predicted
- **AND** target: 10-50× speedup over full path trace
- **AND** verify: SSIM > 0.90 for predicted frames vs full path trace
- **AND** verify: no flickering between consecutive predicted frames (temporal coherence)
- **AND** log: "100 frames in {total}s ({fps} fps, {predicted_pct}% predicted)"

### Requirement: Scene delta training data generation

Omen SHALL generate temporal training pairs by rendering consecutive animation frames.

#### Scenario: Generate temporal pairs

- **WHEN** training mode for temporal prediction is enabled
- **THEN** render consecutive frames:
  - Frame T: `mi.render(scene, spp=4)` → denoise → `latent_T`
  - Frame T+1: `mi.render(scene, spp=4)` → denoise → `latent_T1`
  - Ground truth T+1: `mi.render(scene, spp=256)` → `gt_T1`
  - Scene delta T→T+1: extract from animation timeline
- **AND** training sample: `(latent_T, delta_T→T+1) → latent_T1` with GT `gt_T1`
- **AND** generate sequences with: smooth orbits (low surprise), jump cuts (high surprise), light ramps (medium), new lights (high), object motion (low)

#### Scenario: Train temporal predictor

- **WHEN** temporal pairs available
- **THEN** train ARPredictor: `L = L_pred + 0.09 * L_sigreg` with NablaAdamW(lr=5e-5)
- **AND** 500 iterations on Cornell box animation data
- **AND** validate: predicted frame SSIM > 0.85 vs 256spp GT
- **AND** validate: surprise detection catches >90% of actual surprises, false positive <10%
