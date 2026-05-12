## ADDED Requirements

### Requirement: Save model checkpoints during training

Omen SHALL save JEPA model checkpoints at regular intervals during training to enable resumption and prevent loss of training progress. Checkpoints SHALL contain model weights, optimizer state, training iteration, and performance metrics.

#### Scenario: Save checkpoint during training

- **WHEN** JEPA training is running and checkpoint interval is reached
- **THEN** serialize model weights to file via Nabla's `state_dict()`
- **AND** serialize optimizer state (Adam moments, learning rate)
- **AND** save training iteration number
- **AND** save performance metrics (loss, SSIM, PSNR)
- **AND** write to `checkpoint_iter_<N>.omen` in training directory
- **AND** maintain symlink `latest.omen` to most recent checkpoint

#### Scenario: Resume training from checkpoint

- **WHEN** training is interrupted (crash, user abort, system shutdown)
- **AND** checkpoint file exists in training directory
- **THEN** load model weights from `latest.omen`
- **AND** restore optimizer state
- **AND** resume training from saved iteration number
- **AND** log "Resumed from iteration N"
- **AND** continue training without repeating completed iterations

### Requirement: Version model checkpoints with architecture metadata

Omen SHALL store model architecture metadata alongside weights to enable compatibility checking and prevent loading incompatible models. Metadata SHALL include layer dimensions, JEPA version, and Nabla version.

#### Scenario: Save checkpoint with metadata

- **WHEN** saving checkpoint file
- **THEN** create JSON metadata file `<checkpoint>.omen.meta.json`
- **AND** include architecture hash (layer dimensions, token counts)
- **AND** include Omen version string
- **AND** include Nabla version string
- **AND** include training timestamp
- **AND** include training dataset hash (scene type, resolution range)

#### Scenario: Validate checkpoint before loading

- **WHEN** loading checkpoint file
- **THEN** read metadata JSON from `<checkpoint>.omen.meta.json`
- **AND** compute current architecture hash
- **AND** verify architecture hash matches metadata
- **AND** verify Omen version is compatible (semver check)
- **AND** raise error if incompatible: "Cannot load checkpoint: architecture mismatch"
- **AND** log warning if version differs but compatible: "Loading checkpoint from Omen v0.1.2, current is v0.1.5"

### Requirement: Store base pre-trained model in global cache

Omen SHALL provide a base pre-trained JEPA model trained on Cornell box variants and stored in global user cache directory. Base model SHALL be automatically downloaded on first use if not present locally.

#### Scenario: Download base model on first use

- **WHEN** Omen is invoked for first time
- **AND** no model exists in `~/.cache/omen/models/base_v0.omen`
- **THEN** check for base model at bundled location `<install_dir>/models/base_v0.omen`
- **AND** if not bundled, download from `https://omen-render.org/models/base_v0.omen`
- **AND** verify SHA256 checksum matches expected value
- **AND** save to `~/.cache/omen/models/base_v0.omen`
- **AND** create metadata file `base_v0.omen.meta.json`
- **AND** log "Downloaded base model (52MB, SHA256 verified)"

#### Scenario: Load base model for inference

- **WHEN** JEPA inference is requested
- **AND** no scene-specific fine-tuned model exists
- **THEN** load `~/.cache/omen/models/base_v0.omen`
- **AND** validate architecture compatibility
- **AND** use for inference directly (no fine-tuning needed)
- **AND** log "Using base model (pre-trained on Cornell box variants)"

### Requirement: Fine-tune base model per scene and cache results

Omen SHALL fine-tune the base JEPA model on scene-specific training data and cache the fine-tuned weights for fast subsequent renders. Scene-specific models SHALL be indexed by scene hash to enable automatic discovery.

#### Scenario: Generate scene-specific fine-tuned model

- **WHEN** user renders same scene 3+ times with JEPA enabled
- **THEN** compute scene hash from geometry + materials + lights (not camera position)
- **AND** check `~/.cache/omen/models/scenes/<hash>/fine_tuned.omen`
- **AND** if not exists, trigger fine-tuning (50 iterations, ~1-5 minutes)
- **AND** render training pairs from current scene (4spp + 256spp, different camera angles)
- **AND** train starting from base model weights
- **AND** save fine-tuned weights to scene-specific cache
- **AND** log "Fine-tuned model for scene (hash: <hash>, 50 iterations, SSIM improved from 0.91 to 0.97)"

#### Scenario: Load scene-specific cached model

- **WHEN** rendering scene with JEPA enabled
- **THEN** compute scene hash from current Mitsuba scene
- **AND** check if fine-tuned model exists: `~/.cache/omen/models/scenes/<hash>/fine_tuned.omen`
- **AND** if exists, load fine-tuned model instead of base model
- **AND** validate scene hash matches cached metadata
- **AND** use fine-tuned model for inference
- **AND** log "Using scene-specific model (SSIM: 0.97, trained on 20 frames)"

### Requirement: Aggregate learning from user scenes into improved base model

Omen SHALL support optional aggregation of scene-specific fine-tuned models to improve the base model over time. Aggregation SHALL be opt-in with local-only default to preserve user privacy.

#### Scenario: Opt-in to model improvement program

- **WHEN** user enables "Help improve Omen" setting in configuration
- **THEN** prompt: "Omen can learn from your scenes to improve rendering quality for everyone. Your scene data (geometry, materials, lights) will be used to train future models. No uploads without explicit consent."
- **AND** offer options:
  - "Local only: Improve models on my machine only (default)"
  - "Anonymous upload: Contribute de-identified models to improve Omen for everyone"
- **AND** save user preference to `~/.config/omen/config.yaml`

#### Scenario: Local model aggregation

- **WHEN** "Local only" mode is enabled
- **AND** 5+ scene-specific fine-tuned models exist
- **THEN** trigger periodic background aggregation (weekly, or manual trigger)
- **AND** load base model + all fine-tuned models
- **AND** compute weight updates via federated averaging
- **AND** update base model with aggregated improvements
- **AND** save as `~/.cache/omen/models/base_v1_local.omen`
- **AND** log "Aggregated 7 scene models → improved base model (SSIM +0.03 on average)"
- **AND** no data leaves user's machine

#### Scenario: Anonymous model contribution

- **WHEN** "Anonymous upload" mode is enabled
- **AND** user has trained scene-specific model
- **THEN** prompt: "Share fine-tuned model for scene <hash> to help improve Omen? Scene data (geometry, materials) will be de-identified before upload."
- **AND** if user consents:
  - Remove identifiable features from scene graph (exact coordinates, specific textures)
  - Upload only weight deltas (difference from base model)
  - Include metadata only: scene type (indoor, outdoor), complexity metrics, training performance
  - Send to `https://omen-render.org/api/contribute`
- **AND** log "Contributed model for scene <hash> (de-identified, 2.1MB upload)"

### Requirement: Detect similar scenes and reuse cached models

Omen SHALL analyze scene features to detect similarity with previously cached models and automatically reuse appropriate fine-tuned models without retraining.

#### Scenario: Scene similarity detection

- **WHEN** computing scene hash for cache lookup
- **AND** no exact match exists
- **THEN** compute scene feature vector:
  - Number of meshes, triangles, materials, lights
  - Material type distribution (glass, metal, diffuse count)
  - Scene bounding box, light position variance
- **AND** query scene database for similar feature vectors (cosine similarity > 0.85)
- **AND** if similar scene found:
  - Load cached model from similar scene
  - Run 10-iteration quick adaptation on current scene
  - Save as new fine-tuned model
- **AND** log "Reused model from similar scene (similarity: 0.92), adapted in 10 iterations"

#### Scenario: Scene feature database

- **WHEN** fine-tuned model is saved
- **THEN** extract scene feature vector
- **AND** save to `~/.cache/omen/models/scene_index.json`
- **AND** include mapping: feature hash → model path, scene type, metadata
- **AND** enable fast similarity queries without loading all models

### Requirement: GPU rendering configuration for zero-copy buffers

Omen SHALL detect Mitsuba rendering backend (CPU/GPU) and configure JEPA inference to use zero-copy buffer passing when both Mitsuba and JEPA are on the same GPU device.

#### Scenario: Detect GPU rendering backend

- **WHEN** JEPABridge is initialized
- **THEN** query Mitsuba variant via `mi.variant()`
- **AND** parse variant string:
  - `cuda_ad_*` → NVIDIA CUDA GPU
  - `llvm_ad_*` → AMD ROCm GPU
  - `metal_ad_*` → Apple Metal GPU
  - `cpu_ad_*` → CPU rendering
- **AND** store `is_gpu_render` flag (True for GPU variants)

#### Scenario: Configure zero-copy buffer passing

- **WHEN** both Mitsuba and JEPA are on GPU
- **AND** `mi.variant()` is GPU variant (e.g., `cuda_ad_rgb`)
- **THEN** detect GPU device ID from Mitsuba context
- **AND** pass `gpu_device_id` parameter to Mojo `omen_denoise()`
- **AND** wrap Mitsuba GPU tensor pointer as `UnsafePointer` with `owning=False`
- **AND** Mojo creates `DeviceBuffer` from pointer without `memcpy`
- **AND** log "Zero-copy mode: Mitsuba CUDA device 0 → Mojo CUDA device 0"

#### Scenario: Fallback to CPU-GPU memcpy

- **WHEN** Mitsuba is rendering on CPU (`cpu_ad_*` variant)
- **AND** JEPA is running on GPU
- **THEN** detect mismatch: `is_gpu_render = False`, `jepa_device_id >= 0`
- **AND** allocate GPU buffer in Mojo
- **AND** copy CPU render data to GPU via `memcpy`
- **AND** log "Memcpy mode: CPU → GPU (10-50ms overhead)"
- **AND** run JEPA inference on GPU
- **AND** copy result back to CPU if needed

### Requirement: GPU memory budget management

Omen SHALL monitor GPU memory usage during JEPA inference and training to prevent out-of-memory errors. System SHALL pre-allocate memory budget and gracefully degrade if insufficient memory available.

#### Scenario: Detect GPU memory availability

- **WHEN** JEPABridge initializes
- **AND** GPU device is selected
- **THEN** query GPU memory info via CUDA/HIP/Metal API
- **AND** record total memory, free memory, reserved memory
- **AND** compute available budget (free - 500MB safety margin)
- **AND** log "GPU memory: 8GB total, 6GB free, 5.5GB available for Omen"

#### Scenario: Allocate memory within budget

- **WHEN** loading JEPA model for inference
- **THEN** estimate memory requirements:
  - Model weights: ~500MB
  - Scene graph tensors: ~100MB (varies by scene complexity)
  - Render buffers: ~50MB (varies by resolution)
  - Workspace (activations): ~200MB
- **AND** verify total estimate < available budget
- **AND** if insufficient:
  - Log warning: "Insufficient GPU memory (need 2.5GB, have 1.8GB)"
  - Fall back to CPU inference or reduce resolution
- **AND** if sufficient:
  - Allocate model and buffers
  - Proceed with inference

#### Scenario: Training memory budget

- **WHEN** starting JEPA training
- **THEN** estimate training memory:
  - Model weights: ~500MB
  - Optimizer state (Adam moments): ~1GB (2× model weights)
  - Gradients: ~500MB
  - Batch data: ~200MB
  - Workspace: ~500MB
- **AND** verify total < available budget
- **AND** if insufficient:
  - Reduce batch size (e.g., 8 → 4 → 2 → 1)
  - Log "Reduced batch size to 1 due to memory constraints"
  - If still insufficient, fall back to CPU training
