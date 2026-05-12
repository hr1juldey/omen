## ADDED Requirements

### Background: JEPA as World Model (from LeWorldModel)

LeWorldModel (LeWM, Maes et al. 2026, co-authored by Yann LeCun) proves JEPA works as a world model with:
- **Two losses only**: next-embedding prediction + SIGReg (Gaussian regularizer)
- **~15M parameters**, trains in hours on single GPU
- **48x faster planning** than foundation-model world models
- **Surprise detection**: reliably detects physically implausible events
- **Autoregressive rollout**: predicts N future frames from history window

Omen adapts LeWM's architecture for **rendering acceleration**: replace "robot actions" with "scene deltas" (camera moves, object transforms, new emitters, light changes). The world model predicts what the next frame looks like WITHOUT path tracing. Only path-trace on surprise (unexpected scene change).

### Requirement: Autoregressive frame prediction using JEPA world model

Omen SHALL implement an autoregressive JEPA predictor (based on LeWM architecture) that predicts clean frame output from 1spp dirty render + scene graph + temporal history. Every frame SHALL be rendered at 1spp (providing exact geometry/occlusion ground truth) and then cleaned by the JEPA world model using temporal context. Prediction SHALL NEVER generate frames from nothing - always conditioned on actual render data + exact scene graph (geometry, shaders, lights).

#### Scenario: Predict clean frame from 1spp dirty render + history

- **WHEN** rendering an animation sequence (frame N, where N > 0)
- **AND** 1spp dirty render for frame N is available (`mi.render(scene, spp=1)`)
- **AND** scene graph for frame N is extracted (exact geometry, materials, lights)
- **AND** previous frame latents are available (frame N-1, N-2, ... up to history_size)
- **AND** scene delta between frame N-1 and frame N is computed
- **THEN** encode 1spp dirty render + scene graph into current frame latent
- **AND** encode scene delta: camera transform delta, object transform deltas, light parameter deltas, material parameter deltas
- **AND** feed to autoregressive predictor: `predict(history_latents, current_latent, delta_embeddings)`
- **AND** receive predicted clean latent for frame N
- **AND** decode predicted latent to RGB: `decoder(predicted_latent) → frame_N_clean_rgba`
- **AND** 1spp render + prediction pipeline SHALL complete in <50ms at 256x256

#### Scenario: History window management

- **WHEN** rendering animation frame N
- **THEN** maintain circular buffer of previous `history_size` latents (default: 3)
- **AND** maintain corresponding scene deltas and clean frame latents
- **AND** truncate to most recent `history_size` frames (like LeWM's `emb[:, -HS:]`)
- **AND** on sequence start (frame 0): render 1spp + JEPA denoise → store as first latent
- **AND** on camera jump cut: clear buffer, render 1spp + JEPA denoise → new anchor

### Requirement: Scene delta encoder for animation changes

Omen SHALL encode per-frame scene changes as structured "scene deltas" that replace LeWM's action encoder. Scene deltas SHALL capture all animation data from the timeline: camera movement, object transforms, light changes, material animation, and new/deleted scene elements.

#### Scenario: Encode camera movement delta

- **WHEN** camera transforms between frames are available
- **THEN** compute transform delta: `T_delta = T_frame_N × inverse(T_frame_N-1)`
- **AND** extract translation delta (3 floats) and rotation delta (quaternion, 4 floats)
- **AND** encode via scene delta encoder MLP: `delta_emb = encoder([trans_delta, rot_delta])`
- **AND** pass delta embedding to autoregressive predictor

#### Scenario: Encode object animation delta

- **WHEN** animated objects have per-frame transforms
- **THEN** for each animated object: compute transform delta from previous frame
- **AND** include object type hint (mesh, volume, instance)
- **AND** encode per-object deltas: `object_delta_emb = encoder([obj_id, trans_delta, rot_delta, scale_delta])`
- **AND** aggregate object deltas into scene delta vector (sum or attention)

#### Scenario: Encode fluid/smoke introduction (new scene element)

- **WHEN** a new volume emitter (fluid, smoke, fire) appears in the scene
- **AND** it did not exist in the previous frame
- **THEN** detect new scene element via scene graph diff (new emitter in list)
- **AND** encode as "birth" delta: `birth_delta = encoder([type=volume, position, size, density])`
- **AND** flag this as a **high-surprise event** (confidence = 0)
- **AND** trigger full path trace for this frame (prediction unreliable)
- **AND** add new element to scene graph for subsequent frames

#### Scenario: Encode light change delta

- **WHEN** light parameters change between frames (intensity, color, position)
- **THEN** compute per-light parameter deltas
- **AND** encode: `light_delta = encoder([light_id, intensity_delta, color_delta, pos_delta])`
- **AND** if intensity change > 50%: flag as medium-surprise (reduce prediction confidence)
- **AND** if new light appears: flag as high-surprise (trigger full path trace)

#### Scenario: Encode material animation delta

- **WHEN** material parameters are animated (emissive pulse, color change, roughness change)
- **THEN** compute per-material parameter deltas from previous frame
- **AND** encode: `material_delta = encoder([mat_id, param_deltas])`
- **AND** pass to predictor as part of scene delta

### Requirement: Surprise detection for unexpected scene changes

Omen SHALL implement surprise detection (based on LeWM's surprise evaluation) to identify frames where JEPA prediction is unreliable. Surprise SHALL trigger full path tracing instead of prediction. Surprise is measured by comparing predicted latent with actual rendered latent.

#### Scenario: Detect surprise during prediction

- **WHEN** JEPA predicts frame N latent from history
- **AND** frame N has been path-traced (for validation or periodic check)
- **THEN** compute surprise score: `surprise = MSE(predicted_latent, actual_render_latent)`
- **AND** normalize surprise score relative to running average
- **AND** if surprise > threshold (configurable, default 2σ):
  - Log "Surprise detected at frame N (score: X, threshold: Y)"
  - Flag this scene region for retraining
  - Mark prediction as unreliable for subsequent frames

#### Scenario: Detect new scene elements as surprise

- **WHEN** scene graph diff between frames shows:
  - New emitter (fluid, smoke, fire, light)
  - Deleted object
  - Material type change (diffuse → glass)
- **THEN** immediately classify as high-surprise WITHOUT computing prediction
- **AND** trigger full path trace for this frame
- **AND** clear history buffer (prediction context is invalid)
- **AND** restart prediction from this frame as new anchor

#### Scenario: Periodic validation with path trace

- **WHEN** JEPA has predicted N consecutive frames without path tracing (configurable, default 5)
- **THEN** trigger a validation path trace at current frame
- **AND** compare predicted latent with actual render latent
- **AND** compute surprise score
- **AND** if surprise < threshold: prediction is reliable, continue predicting
- **AND** if surprise > threshold: prediction has drifted, re-anchor with this render and clear old history

### Requirement: Topology-based scene hashing for animation

Omen SHALL compute scene hash based on topology (face connectivity, material type assignments, light types) rather than vertex positions, so that animated scenes maintain a stable cache key across frames. Vertex positions are treated as dynamic data, not scene identity.

#### Scenario: Compute topology hash for animated scene

- **WHEN** fine-tuned model cache lookup is needed
- **THEN** compute topology hash from:
  - Face connectivity (triangle adjacency, not vertex positions)
  - Material type IDs per face (diffuse, glass, metal, etc.)
  - Light type IDs (point, area, spot, environment)
  - Object count and hierarchy
- **AND** exclude from hash: vertex positions, light intensities, material parameter values, camera transform
- **AND** use topology hash for cache lookup (not vertex hash)
- **AND** verify: rotating an object does NOT change topology hash
- **AND** verify: deforming a mesh does NOT change topology hash
- **AND** verify: adding a new light DOES change topology hash

#### Scenario: Cache hit on animated scene

- **WHEN** rendering frame N of an animation
- **AND** topology hash matches cached model
- **THEN** load fine-tuned model for this scene
- **AND** use for prediction across all animation frames
- **AND** no retraining needed between frames of same scene

### Requirement: Animation render pipeline with prediction

Omen SHALL implement a multi-mode animation pipeline that uses JEPA world model prediction for most frames and falls back to path tracing only when needed. Pipeline SHALL achieve path-traced quality at real-time speeds (>30fps for 256x256, >10fps for 1080p).

#### Scenario: Render animation sequence with prediction

- **WHEN** rendering animation frames 0 through N
- **THEN** for frame 0 (anchor):
  - Path trace at 4spp → JEPA denoise → store latent in history buffer
  - This is the "anchor frame" (higher quality bootstrap)
- **AND** for frames 1 through N:
  - **ALWAYS** render at 1spp: `mi.render(scene, spp=1)` → dirty frame (~30ms at 256x256)
  - Extract scene graph for current frame (exact geometry, materials, lights)
  - Compute scene delta from previous frame
  - Encode dirty frame + scene graph into current latent
  - Predict clean latent: `predict(history[-H:], current_latent, delta)`
  - Decode to RGB → output clean frame
  - If surprise detected:
    - Re-render at 4spp → JEPA denoise → higher quality anchor
    - Replace predicted latent with actual high-quality latent in history
  - Else:
    - Store predicted latent in history buffer

#### Scenario: Handle camera jump cut in animation

- **WHEN** camera transform delta exceeds threshold (translation > 1 unit or rotation > 45°)
- **THEN** detect as jump cut
- **AND** clear history buffer (old context is irrelevant)
- **AND** path trace new anchor frame at 4spp → JEPA denoise
- **AND** restart prediction from new anchor
- **AND** log "Jump cut detected at frame N, re-anchoring"

#### Scenario: Handle fluid/smoke introduction mid-animation

- **WHEN** scene graph diff shows new volume emitter at frame K
- **THEN** detect as high-surprise event
- **AND** path trace frame K fully (4spp → JEPA denoise)
- **AND** update scene graph with new emitter
- **AND** re-encode scene delta to include new element
- **AND** restart prediction with new scene context
- **AND** expect 2-3 frames of reduced prediction quality as model adapts to new element
- **AND** after 2-3 frames: prediction quality recovers as history includes new element

#### Scenario: Handle new light introduction mid-animation

- **WHEN** scene graph diff shows new light source at frame K
- **THEN** detect as high-surprise event
- **AND** path trace frame K fully (4spp → JEPA denoise)
- **AND** update scene graph with new light
- **AND** restart prediction with new scene context
- **AND** expect 1-2 frames of reduced quality as global illumination adapts

### Requirement: SIGReg loss for stable latent space

Omen SHALL implement the SIGReg (Sketch Isotropic Gaussian Regularizer) loss from LeWM to maintain well-behaved latent embeddings during training. SIGReg enforces Gaussian distribution on latents, preventing representation collapse with only one hyperparameter.

#### Scenario: Apply SIGReg during training

- **WHEN** JEPA training is running
- **THEN** compute prediction loss: `L_pred = MSE(predicted_latent, target_latent)`
- **AND** compute SIGReg loss: `L_sigreg = SIGReg(embeddings)` (enforces Gaussian)
- **AND** total loss: `L = L_pred + λ * L_sigreg` (λ configurable, default 1.0)
- **AND** this is the ONLY loss needed (no EMA, no pretrained encoder, no auxiliary losses)
- **AND** SIGReg prevents collapse: latents stay well-distributed, not degenerate

### Requirement: Real-time production rendering speed targets

Omen SHALL achieve path-traced animation quality at speeds competitive with EEVEE and Unreal Engine's Lumen. Target speeds assume NVIDIA RTX 3060 or better.

#### Scenario: Animation render speed benchmark

- **WHEN** rendering a 100-frame animation at 256x256 with moderate scene changes
- **THEN** measure per-frame timing for each mode:
  - **Full path trace every frame** (baseline): ~2s/frame → 200s total
  - **JEPA predicted frames** (no path trace): <5ms/frame → 0.5s for predicted frames
  - **Anchor + prediction** (Omen Mode 4): ~10% frames path-traced, 90% predicted
  - **Expected speedup**: 10-50x over full path tracing
- **AND** verify quality: SSIM > 0.90 for predicted frames vs full path trace
- **AND** verify temporal coherence: no flickering between predicted frames
- **AND** log: "100 frames in X seconds (Y fps, Z% predicted)"

#### Scenario: Handle complex animation with frequent surprises

- **WHEN** animation has frequent scene changes (every 5-10 frames: new lights, fluids, camera cuts)
- **THEN** Omen falls back to path tracing more often (surprise detection triggers)
- **AND** predicted frame ratio drops to 50-70% (still faster than full path trace)
- **AND** quality remains high (no degradation from prediction during surprises)
- **AND** log: "Surprise-heavy animation: 50% predicted, 2x speedup"

### Requirement: Scene delta training data generation

Omen SHALL generate training pairs for the temporal prediction model by rendering consecutive animation frames and extracting scene deltas. Training data SHALL include frames with and without surprises.

#### Scenario: Generate temporal training pairs

- **WHEN** training mode is enabled for temporal prediction
- **THEN** render consecutive frame pairs at 4spp + 256spp:
  - Frame T: render, denoise → latent_T, clean_T
  - Frame T+1: render, denoise → latent_T1, clean_T1
  - Scene delta T→T+1: extract from timeline
- **AND** training sample: `(latent_T, delta_T→T+1) → latent_T1` with ground truth `clean_T1`
- **AND** generate 100+ frame sequences from Cornell box with:
  - Smooth camera orbits (predictable, low surprise)
  - Camera jump cuts (high surprise, should clear buffer)
  - Light intensity ramps (medium surprise)
  - New light introduction (high surprise)
  - Object rotation/translation (low surprise)

#### Scenario: Train temporal predictor

- **WHEN** temporal training pairs are available
- **THEN** train autoregressive predictor with LeWM loss:
  - `L_pred = MSE(predict(latent_history, delta), target_latent)`
  - `L_sigreg = SIGReg(latents)`
  - `L_total = L_pred + λ * L_sigreg`
- **AND** train for 500 iterations on Cornell box animation data
- **AND** validate: predicted frame SSIM > 0.85 vs actual render
- **AND** validate: surprise detection correctly identifies jump cuts and new elements
