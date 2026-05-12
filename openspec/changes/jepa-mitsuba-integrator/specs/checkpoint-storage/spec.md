## ADDED Requirements

### Requirement: Save model checkpoints during training

Omen SHALL save JEPA model checkpoints at regular intervals during training to enable resumption after crash. Checkpoints SHALL use Nabla's `state_dict()` for weight serialization.

#### Scenario: Save checkpoint every 10 iterations

- **WHEN** JEPA training is running and `iteration % 10 == 0`
- **THEN** serialize model weights via Nabla `state_dict()` → ordered dict of tensor name → tensor data
- **AND** serialize optimizer state: AdamW moments (m, v), learning rate, step count
- **AND** save training metadata: iteration number, loss value, SSIM, PSNR
- **AND** write checkpoint file: `~/.cache/omen/checkpoints/checkpoint_iter_{N}.omen`
- **AND** update symlink: `latest.omen → checkpoint_iter_{N}.omen`
- **AND** log: "Checkpoint saved at iteration {N} (loss={loss:.4f})"

#### Scenario: Resume training from checkpoint

- **WHEN** training starts and `latest.omen` exists in checkpoint directory
- **THEN** load model weights from checkpoint via Nabla `load_state_dict()`
- **AND** restore AdamW optimizer state (momentum buffers m, v)
- **AND** resume from saved iteration number
- **AND** log "Resumed from iteration {N}"
- **AND** continue training without repeating completed iterations

### Requirement: Version model checkpoints with architecture metadata

Omen SHALL store model architecture metadata alongside weights to prevent loading incompatible models.

#### Scenario: Save checkpoint with metadata JSON

- **WHEN** saving checkpoint file `checkpoint_iter_N.omen`
- **THEN** create `checkpoint_iter_N.omen.meta.json` alongside with:
  - `architecture_hash`: SHA256 of layer dimensions string (e.g., "ViT-Tiny-192-3-12_AR-6-16-64-2048")
  - `omen_version`: e.g., "0.1.0"
  - `nabla_version`: from Nabla library version string
  - `training_timestamp`: ISO 8601 datetime
  - `training_config`: `{lr, weight_decay, batch_size, total_iterations}`
  - `metrics`: `{loss, ssim, psnr}` at checkpoint time

#### Scenario: Validate checkpoint before loading

- **WHEN** loading checkpoint file
- **THEN** read metadata JSON from companion `.meta.json` file
- **AND** compute current architecture hash from running model
- **AND** if hashes match: proceed with loading
- **AND** if hashes differ: raise `RuntimeError("Architecture mismatch: checkpoint has {ckpt_hash}, current model has {model_hash}")`
- **AND** if version differs but compatible (same major.minor): log warning, proceed
- **AND** if version incompatible: raise error

### Requirement: Store base pre-trained model in global cache

Omen SHALL provide a base pre-trained JEPA model trained on Cornell box variants, stored in global user cache, auto-downloaded on first use.

#### Scenario: Download base model on first use

- **WHEN** Omen is invoked for first time AND `~/.cache/omen/models/base_v0.omen` does not exist
- **THEN** check bundled location: `<install_dir>/models/base_v0.omen`
- **AND** if not bundled: download from `https://omen-render.org/models/base_v0.omen`
- **AND** verify SHA256 checksum matches expected value from `base_v0.omen.sha256`
- **AND** save to `~/.cache/omen/models/base_v0.omen`
- **AND** create `~/.cache/omen/models/` directory if needed
- **AND** log: "Downloaded base model ({size}MB, SHA256 verified)"

#### Scenario: Load base model for inference

- **WHEN** JEPA inference is requested AND no scene-specific model exists
- **THEN** load `~/.cache/omen/models/base_v0.omen`
- **AND** validate architecture compatibility
- **AND** use for inference directly (no fine-tuning)
- **AND** log: "Using base model (pre-trained on Cornell box variants)"

### Requirement: Fine-tune base model per scene and cache results

Omen SHALL fine-tune the base JEPA model on scene-specific data and cache fine-tuned weights indexed by scene hash.

#### Scenario: Generate scene-specific fine-tuned model

- **WHEN** user renders same scene 3+ times with JEPA enabled
- **THEN** compute scene hash from geometry + materials + lights (exclude camera position)
- **AND** check cache: `~/.cache/omen/models/scenes/<hash>/fine_tuned.omen`
- **AND** if not exists:
  - Generate training pairs: render scene at 4spp + 256spp from different camera angles
  - Fine-tune from base model for 50 iterations using NablaAdamW(lr=5e-5, weight_decay=1e-3)
  - Save to scene-specific cache directory
  - Log: "Fine-tuned model for scene (hash: {hash}, 50 iters, SSIM 0.91 → 0.97)"

#### Scenario: Load scene-specific cached model

- **WHEN** rendering scene with JEPA enabled
- **THEN** compute scene hash from current Mitsuba scene
- **AND** check `~/.cache/omen/models/scenes/<hash>/fine_tuned.omen`
- **AND** if exists: load fine-tuned model (skip base model)
- **AND** validate scene hash matches cached metadata
- **AND** log: "Using scene-specific model (SSIM: 0.97, trained on {N} frames)"

### Requirement: Topology-based scene hashing for animation cache

Omen SHALL compute scene hash based on topology (face connectivity, material types, light types) NOT vertex positions, so animated scenes maintain stable cache key across frames.

#### Scenario: Compute topology hash

- **WHEN** cache lookup is needed for animated scene
- **THEN** hash from:
  - Face connectivity (triangle adjacency indices, NOT vertex positions)
  - Material type IDs per face (diffuse=0, glass=1, metal=2, etc.)
  - Light type IDs (point=0, area=1, environment=2)
  - Object count and hierarchy structure
- **AND** exclude: vertex positions, light intensities, material parameter values, camera transform
- **AND** verify: rotating object does NOT change hash
- **AND** verify: deforming mesh does NOT change hash
- **AND** verify: adding new light DOES change hash

### Requirement: Aggregate learning from user scenes

Omen SHALL support optional local aggregation of scene-specific models to improve the base model. Opt-in with local-only default.

#### Scenario: Local federated averaging

- **WHEN** "Local only" mode enabled AND 5+ scene-specific models exist
- **THEN** load base model + all fine-tuned models
- **AND** compute federated average of weight updates
- **AND** save as `~/.cache/omen/models/base_v1_local.omen`
- **AND** no data leaves user's machine
- **AND** log: "Aggregated {N} scene models → improved base model"

#### Scenario: Anonymous contribution (opt-in)

- **WHEN** "Anonymous upload" mode enabled AND user consents
- **THEN** de-identify scene data: remove coordinates, textures, keep only structure
- **AND** compute weight deltas (difference from base model)
- **AND** upload only deltas + anonymized metadata to `https://omen-render.org/api/contribute`

### Requirement: GPU rendering backend detection for zero-copy

Omen SHALL detect Mitsuba rendering backend and configure zero-copy buffer passing when both Mitsuba and JEPA are on the same GPU.

#### Scenario: Detect GPU backend from variant

- **WHEN** JEPABridge initializes
- **THEN** query `mi.variant()`:
  - `cuda_ad_rgb` → NVIDIA CUDA GPU, `is_gpu=True`
  - `llvm_ad_rgb` → CPU/ROCm, check for GPU
  - `scalar_rgb` → CPU only, `is_gpu=False`
- **AND** store `is_gpu_render` flag and `gpu_device_id`

#### Scenario: Zero-copy on same GPU

- **WHEN** both Mitsuba and JEPA are on GPU (same device)
- **THEN** pass GPU pointer directly via C ABI: `ctypes.c_void_p(ptr_value)`
- **AND** Mojo wraps as `DeviceBuffer(ctx, raw_ptr, count, owning=False)`
- **AND** NO memcpy — Mojo reads Mitsuba's GPU memory directly

#### Scenario: CPU-to-GPU memcpy fallback

- **WHEN** Mitsuba on CPU and JEPA on GPU
- **THEN** allocate GPU buffer in Mojo
- **AND** copy CPU data → GPU via `ctx.enqueue_copy(dst_buf=dev_buf, src_buf=host_buf)`
- **AND** log: "Memcpy mode: CPU → GPU (10-50ms overhead)"
- **AND** run JEPA inference on GPU, copy result back to CPU

### Requirement: GPU memory budget management

Omen SHALL monitor GPU memory and gracefully degrade when insufficient.

#### Scenario: Check memory availability

- **WHEN** JEPABridge initializes with GPU
- **THEN** query GPU memory: total, free, available = free - 500MB safety margin
- **AND** log: "GPU memory: {total}GB total, {free}GB free, {available}GB for Omen"

#### Scenario: Inference memory estimate

- **WHEN** loading model for inference
- **THEN** estimate: model ~500MB + scene graph ~100MB + buffers ~50MB + workspace ~200MB ≈ 850MB
- **AND** if estimate > available: fall back to CPU inference or reduce resolution
- **AND** if sufficient: allocate and proceed

#### Scenario: Training memory management

- **WHEN** starting training
- **THEN** estimate: model ~500MB + optimizer ~1GB (Adam moments) + gradients ~500MB + batch ~200MB ≈ 2.2GB
- **AND** if insufficient: reduce batch size (8 → 4 → 2 → 1)
- **AND** if still insufficient: fall back to CPU training
