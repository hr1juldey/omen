## ADDED Requirements

### Requirement: Pure functional forward pass uses params dict directly
The system SHALL provide a `pure_loss_fn(params, noisy, gt, scene_latent, config)` function that computes the training loss using the flat params dict directly. It SHALL NOT call `model.load_state_dict` or any Python object mutation. It SHALL extract weights from params by name and apply nabla operations directly.

#### Scenario: Identical output to imperative forward
- **WHEN** `pure_loss_fn(params, noisy, gt, sg, config)` is called with the same params and inputs as `compute_training_loss(params, model, noisy, gt, sg, config)`
- **THEN** the output tensor SHALL be numerically identical (within float32 tolerance)

#### Scenario: No Python side effects
- **WHEN** `pure_loss_fn` is called multiple times with different params
- **THEN** no Python object state SHALL be mutated — all computation is pure functional

### Requirement: Functional scene encoder
The system SHALL provide a functional scene encoder that takes a params prefix dict and scene graph, producing a scene latent tensor. It SHALL use params like `params["scene_encoder.gnn.layers_0.weight"]` directly in nabla operations.

#### Scenario: Scene graph encoding
- **WHEN** a scene graph dict with vertices (B, N, 3) and material/light params is passed
- **THEN** the functional encoder SHALL produce a scene latent tensor of shape (B, latent_dim) matching the imperative SceneGraphEncoder output

### Requirement: Functional render encoder
The system SHALL provide a functional render encoder that takes a params prefix dict and RGBA render tensor, producing a render latent. It SHALL use `nb.conv2d(x, params["render_encoder.filter_N"], stride, padding)` directly.

#### Scenario: Render feature encoding
- **WHEN** a noisy RGBA render tensor (B, H, W, 4) is passed
- **THEN** the functional render encoder SHALL produce a render latent matching the imperative RenderFeatureEncoder output

### Requirement: Functional cross attention fusion
The system SHALL provide a functional cross attention that takes render latent and scene latent, producing a fused latent. It SHALL extract attention weights from params dict directly.

#### Scenario: Fusion produces correct shape
- **WHEN** render latent (B, D) and scene latent (B, D) are fused
- **THEN** output SHALL be shape (B, D) matching imperative CrossAttentionFusion

### Requirement: Functional decoder
The system SHALL provide a functional U-Net decoder that takes fused latent and noisy image, predicting residual noise. It SHALL use conv2d with params dict weights directly.

#### Scenario: Noise prediction
- **WHEN** fused latent and noisy image are decoded
- **THEN** output SHALL be shape (B, H, W, 3) residual noise prediction matching imperative Decoder

### Requirement: SIGReg physics loss in functional mode
The system SHALL include SIGReg structural similarity + gradient regularization loss as a pure function of the predicted latent. It SHALL have 0 learnable params and use only nabla tensor operations.

#### Scenario: Physics loss computation
- **WHEN** predicted latent is passed to the SIGReg functional loss
- **THEN** it SHALL compute structural similarity and gradient regularization without accessing any model state

### Requirement: GPU-safe activations in functional forward
The functional forward pass SHALL use `sigmoid_gpu` and `silu_gpu` from `omen.kernels.activations` instead of `nb.sigmoid` and `nb.silu`. It SHALL NOT use `nb.tanh`, `nb.gelu`, or `nb.relu` — these have VJP rules that create CPU scalar constants causing backward failures.

#### Scenario: No CPU scalar in backward chain
- **WHEN** `value_and_grad(pure_loss_fn, argnums=0)` computes gradients
- **THEN** no `ensure_tensor(float)` call SHALL create a CPU scalar — all backward ops use tensor-only chains
