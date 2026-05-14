## ADDED Requirements

### Requirement: Conv2dOp Operation with matmul-based vjp_rule
Custom nabla `Operation` subclass that provides forward via im2col+matmul and backward via Nabla matmul in vjp_rule.

#### Scenario: Forward pass produces correct convolution output
- **WHEN** `Conv2dOp` is called with input `(B, H, W, C_in)` and filter `(K, K, C_in, C_out)` with stride and padding
- **THEN** output shape is `(B, H_out, W_out, C_out)` where `H_out = (H + 2*pad - K) / stride + 1`
- **AND** output values match `nb.conv2d` to within float32 tolerance (1e-5)

#### Scenario: vjp_rule computes filter gradient via matmul
- **WHEN** `vjp_rule(primals, cotangents, outputs, kwargs)` is called during backward pass
- **THEN** `grad_filter = im2col(x).T @ cotangent.reshape(-1, C_out)` (pure Nabla matmul)
- **AND** `grad_filter` shape matches filter shape `(K, K, C_in, C_out)`
- **AND** no Mojo kernels are called from vjp_rule (only nb.matmul, nb.reshape, nb.permute)

#### Scenario: vjp_rule computes input gradient via col2im
- **WHEN** `vjp_rule` is called during backward pass
- **THEN** `col = cotangent_flat @ filter.reshape(K*K*C_in, C_out).T` (Nabla matmul)
- **AND** `grad_input = col2im(col)` via Mojo GPU kernel (gather pattern, no atomics)
- **AND** `grad_input` shape matches input shape `(B, H, W, C_in)`

#### Scenario: Bias gradient handled automatically
- **WHEN** bias is provided to `conv2d_safe` and backward pass runs
- **THEN** bias gradient flows through nabla's built-in addition autograd (no custom kernel needed)

### Requirement: conv2d_safe drop-in replacement
Python function `conv2d_safe(x, filter, stride=1, padding=0, bias=None)` API-compatible with `nb.conv2d`.

#### Scenario: Integer stride/padding normalized to tuples
- **WHEN** `stride=2` or `padding=1` is passed as integer
- **THEN** internally normalized to `(2, 2)` or `(1, 1)` tuples

#### Scenario: No bias when bias=None
- **WHEN** `conv2d_safe(x, filter)` is called without bias
- **THEN** only conv2d forward is computed, no addition step

### Requirement: im2col Mojo GPU kernel
Mojo GPU kernel registered as `@compiler.register("conv2d_im2col")`.

#### Scenario: Patch extraction with gather pattern
- **WHEN** kernel is dispatched with input `(B, H, W, C_in)` and stride/padding/kernel_size
- **THEN** output is `(B·H_out·W_out, K·K·C_in)` where each row is one flattened patch
- **AND** out-of-bounds positions (padding) contribute zero
- **AND** each output element written by exactly one thread (no race conditions)

#### Scenario: All model conv2d shapes supported
- **WHEN** kernel runs with decoder filters (3×3, up to 512 input channels) and scene_encoder filters (3×3, stride=2)
- **THEN** kernel produces correct patches without errors

### Requirement: col2im Mojo GPU kernel
Mojo GPU kernel registered as `@compiler.register("conv2d_col2im")`.

#### Scenario: Scatter-back with gather pattern (no atomics)
- **WHEN** kernel is dispatched with column matrix `(B·H_out·W_out, K·K·C_in)` and target shape
- **THEN** output is `(B, H, W, C_in)` where each position accumulates contributions from overlapping patches
- **AND** kernel iterates over OUTPUT positions `(b, ih, iw, ci)` and gathers from column entries (no concurrent writes)

### Requirement: Gradient numerical correctness
Custom VJP gradients must be numerically correct.

#### Scenario: Finite-difference gradient check
- **WHEN** forward output and VJP gradients are compared against finite-difference approximation (epsilon=1e-4)
- **THEN** relative error < 1e-3 for both filter and input gradients

### Requirement: Resolution-agnostic operation
Conv2dOp works at any spatial resolution within VRAM budget.

#### Scenario: 1024×1024 training tile
- **WHEN** conv2d_safe processes 1024×1024 input with mixed precision
- **THEN** VRAM usage is within 2.25 GB budget (model + workspace + staging)

#### Scenario: 4K inference via tiling
- **WHEN** 4096×4096 image is split into 16 tiles of 1024×1024
- **THEN** each tile processed independently with 2px overlap for boundary handling
- **AND** peak VRAM equals single-tile budget (~2.25 GB)
