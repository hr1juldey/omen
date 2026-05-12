## ADDED Requirements

### Background: Nabla Python API (not C ABI)

Omen uses Nabla's Python API for all JEPA inference. Both Mitsuba and Nabla are Python-callable — no ctypes, no shared libraries, no C ABI. Tensor transfer uses DLPack zero-copy: `nb.Tensor.from_dlpack(dr_tensor)`.

### Requirement: Load JEPA model from Nabla checkpoint

Omen SHALL load JEPA inference model from Nabla checkpoint files using `nb.nn.Module.load_state_dict()`. Model is a Python `nb.nn.Module` subclass.

#### Scenario: Load base model for inference

- **WHEN** JEPA inference is first requested
- **THEN** import Nabla: `import nabla as nb`
- **AND** instantiate model: `model = OmenJEPA()`
- **AND** load base weights: `model.load_state_dict(nb.load('~/.cache/omen/models/base_v0.omen'))`
- **AND** set eval mode: `model.eval()`
- **AND** optionally compile for speed: `compiled_model = nb.compile(model)`
- **AND** set `self.available = True`

#### Scenario: Load scene-specific fine-tuned model

- **WHEN** scene-specific model exists at `~/.cache/omen/models/scenes/<hash>/fine_tuned.omen`
- **THEN** compute scene topology hash from current Mitsuba scene
- **AND** check cache for fine-tuned model
- **AND** if exists: load fine-tuned weights instead of base model
- **AND** validate architecture hash matches current model
- **AND** log: "Using scene-specific model (SSIM: 0.97)"

#### Scenario: Handle Nabla import failure (graceful degradation)

- **WHEN** `import nabla as nb` raises `ImportError`
- **THEN** log: "Nabla not installed. Install with: pip install nabla-ml"
- **AND** set `self.available = False`
- **AND** all subsequent bridge calls return unmodified input (passthrough mode)
- **AND** no exception raised — existing Mitsuba path tracing works normally

#### Scenario: Handle model load failure

- **WHEN** checkpoint file not found or corrupted
- **THEN** catch exception from `load_state_dict()`
- **AND** log: "JEPA model not found at {path}. Falling back to standard path tracing."
- **AND** set `self.available = False`

### Requirement: Transfer render data via DLPack

Omen SHALL transfer Mitsuba render data to Nabla tensors via DLPack zero-copy protocol. Both Dr.Jit and Nabla support DLPack.

#### Scenario: DLPack zero-copy (same GPU)

- **WHEN** Mitsuba variant is `cuda_ad_rgb` and Nabla has CUDA context
- **THEN** render: `image = mi.render(scene, spp=4)` -> Dr.Jit TensorXf on GPU
- **AND** convert to DLPack: `dl_tensor = image.dlpack()` or `nb.Tensor.from_dlpack(image)`
- **AND** result is a Nabla tensor sharing the SAME GPU memory — no memcpy
- **AND** add alpha channel if needed: `rgba = nb.concatenate([tensor, ones], axis=-1)` -> `(H, W, 4)`
- **AND** log: "DLPack zero-copy: Mitsuba CUDA -> Nabla CUDA"

#### Scenario: NumPy fallback (CPU render)

- **WHEN** Mitsuba variant is `scalar_rgb` or `llvm_ad_rgb`
- **THEN** convert to numpy first: `np_array = np.array(image)` -> host memory
- **AND** create Nabla tensor: `nb_tensor = nb.ndarray(np_array)`
- **AND** move to GPU if available: `nb_tensor = nb_tensor.cuda()`
- **AND** log: "CPU render -> NumPy -> Nabla (10-50ms copy overhead for 256x256)"

#### Scenario: Transfer scene graph as Nabla tensors

- **WHEN** scene graph dict is extracted (Python dict with numpy arrays)
- **THEN** convert each field to Nabla tensor: `nb.Tensor.from_dlpack(np_array)` or `nb.ndarray(np_array)`
- **AND** scene graph becomes dict of Nabla tensors: `{geometry: nb.Tensor, materials: nb.Tensor, lights: nb.Tensor, camera: nb.Tensor}`
- **AND** pass directly to model forward methods

### Requirement: Invoke JEPA denoise inference

Omen SHALL call JEPA model's forward method directly in Python with Nabla tensors. No C function calls.

#### Scenario: Successful denoise inference

- **WHEN** model is loaded and render data is transferred
- **THEN** encode: `latent = model.encode(scene_graph, rgba_render)` -> shape `(1, 192)`
- **AND** decode with MLA-compressed skips and tile-based MoE routing:
  - U-Net encoder: extract multi-scale features, compress skips via MLA (16× reduction)
  - Bottleneck: Swin Transformer + MoE FFN routed per 8×8 tile using cryptomatte-style material/light/geo masks
  - U-Net decoder: reconstruct MLA skips, produce clean pixels
- **AND** `clean_rgba = model.decode(latent)` -> shape `(1, H, W, 4)`
- **AND** convert back to numpy: `output = clean_rgba.numpy()` -> `(H, W, 4)`
- **AND** return as numpy array
- **AND** total inference time target: <100ms at 256x256 on GPU

#### Scenario: Compiled inference path

- **WHEN** `@nb.compile` has been applied to the model
- **THEN** first call triggers JIT compilation (may take seconds)
- **AND** subsequent calls use compiled MAX graph (faster than eager)
- **AND** compilation is cached per input shape (LRU cache)
- **AND** use `dynamic_dims` for variable-resolution support

#### Scenario: Handle inference failure

- **WHEN** model forward raises exception (OOM, shape mismatch)
- **THEN** catch exception and log: "JEPA denoise failed: {exception}"
- **AND** return original noisy render unchanged
- **AND** if OOM: try reducing resolution or falling back to CPU

### Requirement: Invoke confidence prediction

Omen SHALL run model forward to get both denoised image and confidence map.

#### Scenario: Predict confidence on preview render

- **WHEN** mode=2 (adaptive) and preview render (4 spp) is available
- **THEN** encode: `latent = model.encode(scene_graph, rgba_preview)`
- **AND** denoise: `clean_rgba = model.decode(latent)` -> `(H, W, 4)`
- **AND** confidence: `confidence = model.predict_confidence(latent)` -> `(H, W, 1)` in [0, 1]
- **AND** return tuple: `(clean_preview: np.ndarray, confidence: np.ndarray)`

### Requirement: Invoke multi-resolution merge

Omen SHALL run model forward to merge low-res clean and high-res noisy inputs.

#### Scenario: Merge multi-resolution inputs

- **WHEN** mode=3 (multires) and both renders are available
- **THEN** encode low-res: `latent_low = model.encode(scene_graph, low_res_rgba)`
- **AND** encode high-res: `latent_high = model.encode(scene_graph, high_res_rgba)`
- **AND** merge: `merged = model.merge(latent_low, latent_high, scene_graph, scale=4)` -> `(H, W, 4)`
- **AND** return merged RGBA as numpy array

### Requirement: Live per-scene fine-tuning

Omen SHALL fine-tune the base JEPA model on-the-fly when the user renders the same scene multiple times.

#### Scenario: Trigger fine-tuning on repeated renders

- **WHEN** same scene has been rendered 3+ times (detected via topology hash)
- **THEN** collect training pairs from previous renders (4spp + 256spp pairs)
- **AND** initialize LoRA adapter: `init_lora_adapter(model, rank=8)`
- **AND** run background fine-tuning: 50 iterations with `NablaAdamW(lr=5e-5, weight_decay=1e-3)`
- **AND** save fine-tuned model to scene cache
- **AND** subsequent renders automatically use fine-tuned model
- **AND** log: "Fine-tuned model for scene (hash: {hash}, 50 iters)"

#### Scenario: Training uses Nabla Python API

- **WHEN** fine-tuning is running
- **THEN** use Nabla PyTorch-style training loop:
  - `model.train()` to enable gradient tracking
  - `optimizer = nb.nn.optim.AdamW(model.trainable_params(), lr=5e-5, weight_decay=1e-3)`
  - Forward pass, compute loss, backward, optimizer step
- **AND** loss: `L = L_pred + 0.09 * L_sigreg` (prediction + SIGReg regularization)
- **AND** gradient clip: clip norm to 1.0 before optimizer step
- **AND** training data generated by Dr.Jit (Mitsuba differentiable rendering)
