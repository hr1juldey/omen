## ADDED Requirements

### Background: Two Autograd Systems

Omen uses TWO separate autograd systems that do NOT interact:
1. **Dr.Jit autodiff** (Python): Handles gradient flow through Mitsuba's path tracer. Used for generating training data and computing rendering losses.
2. **Nabla autograd** (Mojo): Handles gradient flow through JEPA model weights. Used for training the neural network.

The bridge between them: Dr.Jit renders training pairs (noisy + GT) then passes tensors to Nabla via DLPack zero-copy where Nabla trains JEPA on those tensors. No gradient flows between the two systems.

### Requirement: Mitsuba differentiable rendering for training data

Omen SHALL use Mitsuba's `cuda_ad_rgb` (or `llvm_ad_rgb`) variant with Dr.Jit autodiff to render training pairs. Dr.Jit handles gradient tracking through the path tracer; Nabla handles JEPA weight updates.

#### Scenario: Set up differentiable rendering variant

- **WHEN** JEPA training mode is initialized
- **THEN** verify Mitsuba variant supports AD: `mi.variant()` must contain `'_ad_'` (e.g., `cuda_ad_rgb`)
- **AND** if variant is scalar (`scalar_rgb`): raise error "Training requires AD variant (cuda_ad_rgb or llvm_ad_rgb)"
- **AND** import Dr.Jit: `import drjit as dr`
- **AND** import Dr.Jit optimizer: `from drjit.opt import Adam as DrJitAdam` (for scene parameter optimization, NOT JEPA weights)

#### Scenario: Generate supervised training pair (denoiser)

- **WHEN** generating a denoiser training pair
- **THEN** render noisy input: `noisy = mi.render(scene, spp=4, seed=42)` then TensorXf becomes `(H, W, 3)`
- **AND** render ground truth: `gt = mi.render(scene, spp=256, seed=42)` then TensorXf becomes `(H, W, 3)`
- **AND** note: same seed ensures identical sample positions, only spp differs
- **AND** pass both tensors to Nabla via DLPack: `noisy_nb = nb.Tensor.from_dlpack(noisy)`, `gt_nb = nb.Tensor.from_dlpack(gt)`
- **AND** Nabla computes loss and backward pass on JEPA weights
- **AND** Dr.Jit is NOT involved in backward pass on JEPA weights — only renders the data

#### Scenario: Generate training pair with scene parameter variation

- **WHEN** generating diverse training data by varying scene parameters
- **THEN** enable gradient tracking: `dr.enable_grad(params['key'])` and `dr.schedule(params['key'])`
- **AND** modify scene parameters:
  ```python
  params = mi.traverse(scene)
  params['emitter.intensity'] = new_intensity  # vary light
  params.update()  # apply changes
  ```
- **AND** render with modified scene: `noisy = mi.render(scene, spp=4)`
- **AND** render GT: `gt = mi.render(scene, spp=256)`
- **AND** pass pair to Nabla for JEPA training

### Requirement: Dr.Jit autodiff API for advanced training

Omen SHALL use Dr.Jit's autodiff primitives for advanced training scenarios. Key APIs: `dr.enable_grad`, `dr.backward`, `dr.forward`, `dr.schedule`, `dr.eval`.

#### Scenario: Reverse-mode autodiff (backward)

- **WHEN** computing gradient of rendering loss with respect to scene parameters
- **THEN** enable grad: `dr.enable_grad(param)`
- **AND** schedule for deferred processing: `dr.schedule(param)`
- **AND** render: `image = mi.render(scene, spp=1)`
- **AND** compute loss: `loss = dr.mean(dr.square(image - gt))`
- **AND** backward: `dr.backward(loss)`
- **AND** read gradient: `grad = dr.grad(param)` — tells how scene parameter affects render

#### Scenario: Forward-mode autodiff (sensitivity)

- **WHEN** computing how a specific parameter direction affects the render
- **THEN** enable grad: `dr.enable_grad(param)`
- **AND** set seed gradient: `dr.set_grad(param, direction_vector)`
- **AND** render: `image = mi.render(scene, spp=1)`
- **AND** forward propagate: `dr.forward(loss)` or `dr.forward_to(image)`
- **AND** read resulting gradient image: `dr.grad(image)` — shows how render changes with parameter

#### Scenario: Deferred evaluation with schedule then processing

- **WHEN** building complex computation graphs before executing
- **THEN** mark parameters: `dr.schedule(param)` — queues for later evaluation
- **AND** build computation: render, loss, etc.
- **AND** force evaluation: `dr.process()` or `dr.eval(result)` — executes all queued operations
- **AND** this enables efficient batching of GPU operations

### Requirement: Nabla autograd for JEPA weight updates

Omen SHALL use Nabla's autograd (NOT Dr.Jit, NOT PyTorch) for all JEPA model weight updates during training. Training runs in Python via Nabla API with Mojo/MAX backend.

#### Scenario: Forward and backward in Nabla (Python side)

- **WHEN** `model.train()` is called and training pair is available
- **THEN** receive training pair: `(noisy_rgba, gt_rgba, scene_graph)` as Nabla tensors via DLPack
- **AND** forward pass in Nabla:
  - U-Net denoise: `denoised = model.denoise(scene_graph, noisy, prev_clean=None)`
  - Compute `pred_loss = nb.mean(nb.square(denoised - gt))`
  - Compute `sigreg_loss = SIGReg(model.embeddings)`
  - Compute `energy_loss = nb.mean(nb.relu(nb.sum(denoised, axis=-1) - nb.sum(noisy, axis=-1) - 0.01))`
  - Total: `total_loss = pred_loss + 0.1 * energy_loss + 0.09 * sigreg_loss`
- **AND** backward pass in Nabla: `total_loss.backward()`
- **AND** optimizer step: `model = optimizer.step()`
- **AND** gradient clip: clip gradient norm to 1.0 before optimizer step
- **AND** return loss value to Python: write total_loss to loss_out pointer

#### Scenario: Training loop structure

- **WHEN** training is running
- **THEN** data generation happens in Python (Dr.Jit renders pairs)
- **AND** model training happens in Nabla Python (Nabla autograd plus AdamW, Mojo/MAX backend)
- **AND** bridge: Python passes tensors via DLPack zero-copy between Dr.Jit and Nabla
- **AND** optimization config from lewm.yaml:
  - Optimizer: NablaAdamW(lr=5e-5, weight_decay=1e-3)
  - Precision: BF16
  - Gradient clip: 1.0
  - Batch size: 128 (or less if GPU memory constrained)
- **AND** checkpoint every 10 iterations via Nabla state_dict serialization

### Requirement: Cornell box training schedule (4 phases)

Omen SHALL follow a 4-phase training protocol on Cornell box.

#### Scenario: Phase 1 — Bootstrap denoiser (iterations 1-100)

- **WHEN** Phase 1 training starts
- **THEN** render Cornell box at 4spp + 256spp (Python/Dr.Jit)
- **AND** train denoiser: MSE between U-Net denoised output and ground truth
- **AND** energy conservation loss: `L_energy = mean(relu(E_out - E_in - 0.01))`
- **AND** SIGReg regularization on embeddings
- **AND** target: SSIM greater than 0.95 vs 256spp after 100 iterations

#### Scenario: Phase 2 — Confidence head (iterations 101-200)

- **WHEN** Phase 2 starts
- **THEN** render Cornell box 8 times at 4spp then compute variance map (Python/Dr.Jit)
- **AND** train ConfidenceHead: MSE(predicted_confidence, 1 - normalized_variance)
- **AND** target: confidence correlates with variance (r greater than 0.7)

#### Scenario: Phase 3 — Multi-res merge (iterations 201-300)

- **WHEN** Phase 3 starts
- **THEN** render 25 percent res 256spp + 100 percent res 4spp + 100 percent res 256spp GT (Python/Dr.Jit)
- **AND** train merge: L1(merged, GT)
- **AND** target: PSNR greater than 30dB vs GT

#### Scenario: Phase 4 — Temporal prediction (iterations 301-500)

- **WHEN** Phase 4 starts
- **THEN** render consecutive animation frames at 1spp + 256spp (Python/Dr.Jit)
- **AND** compute scene deltas (camera orbit, light changes)
- **AND** train ARPredictor: `L = L_pred + 0.09 * L_sigreg`
- **AND** target: predicted frame SSIM greater than 0.85 vs 256spp GT

### Requirement: Self-supervised data generation

Omen SHALL generate unlimited training data by rendering the same scene at different spp, camera angles, and parameter variations.

#### Scenario: Generate denoiser training pairs

- **WHEN** denoiser training data is needed
- **THEN** for each training sample:
  - Random camera position: modify `params['sensor.to_world']` with random rotation and translation
  - Render noisy: `mi.render(scene, spp=4, seed=random_seed)` as input
  - Render GT: `mi.render(scene, spp=256, seed=random_seed)` as target
  - Extract scene graph for this camera angle
- **AND** store as training pair: input, target, scene_graph

#### Scenario: Generate temporal training data with animation

- **WHEN** temporal prediction training data is needed
- **THEN** for a 100-frame animation sequence:
  - Camera orbit: compute camera position per frame using spherical coordinates
  - Frame T: render 4spp then denoise then store latent_T
  - Frame T+1: render 4spp then denoise then store latent_T+1
  - GT T+1: render 256spp then store gt_T+1
  - Scene delta_T from camera position difference
- **AND** include surprise scenarios:
  - Frame 50: turn off light (medium surprise)
  - Frame 70: camera jump cut to opposite side (high surprise)
  - Frame 85: new point light appears (high surprise)
- **AND** validate: surprise detection catches more than 90 percent of these events

### Requirement: Dr.Jit/PyTorch interop bridge (temporary migration path)

Omen MAY use the `dr.wrap` decorator if any PyTorch components are needed temporarily during development. Long-term goal: all ML in Mojo/Nabla.

#### Scenario: Bridge PyTorch and Dr.Jit (temporary)

- **WHEN** a component still uses PyTorch but needs Dr.Jit tensors
- **THEN** use the Dr.Jit wrapping decorator that auto-converts tensor types
- **AND** Dr.Jit tensors convert to PyTorch tensors on function entry
- **AND** PyTorch tensors convert back to Dr.Jit on return
- **AND** gradients flow through the bridge in both directions
- **AND** this is a TEMPORARY measure during Mojo/Nabla migration
- **AND** final version: all ML in Mojo/Nabla, no PyTorch dependency

### Requirement: Multicam shared world model training

Omen SHALL support training and inference with multiple simultaneous cameras sharing a single scene latent, enabling N-camera rendering at ~1.3× single-camera cost.

#### Scenario: Generate multicam training pairs

- **WHEN** pre-training with multicam scenes
- **THEN** for each scene, place N cameras (N=2 to 4) at different viewpoints
- **AND** render each camera view at 4spp and 256spp using the same scene
- **AND** extract ONE shared scene graph (scene encoding is camera-independent)
- **AND** store as: `(scene_latent, [(dirty_cam1, gt_cam1), (dirty_cam2, gt_cam2), ...])`
- **AND** cost: 1 scene encode + N × render ≈ 1.3× single camera for 3 cameras

#### Scenario: Train multicam consistency

- **WHEN** multicam training pairs are available
- **THEN** enforce cross-camera consistency loss: shared scene_latent must denoise all N views correctly
- **AND** train U-Net to use scene_latent (not camera-specific features) for denoising
- **AND** validate: denoised quality across cameras has SSIM variance < 0.02
