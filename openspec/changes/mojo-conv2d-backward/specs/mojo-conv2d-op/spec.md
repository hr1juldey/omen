## ADDED Requirements

### Requirement: Conv2dMojoOp forward pass
The system SHALL provide a custom nabla `Operation` subclass named `Conv2dMojoOp` that performs 2D convolution via im2col + matmul. The forward pass SHALL accept NHWC input `(B, H, W, C_in)` and HWIO filter `(Kh, Kw, C_in, C_out)` and produce output `(B, H_out, W_out, C_out)`.

#### Scenario: Forward pass produces correct output shape
- **WHEN** input is `(1, 16, 16, 4)` and filter is `(3, 3, 4, 8)` with padding=1, stride=1
- **THEN** output shape SHALL be `(1, 16, 16, 8)`

#### Scenario: Forward pass falls back to pure-nabla on kernel failure
- **WHEN** `call_custom_kernel` raises any exception
- **THEN** forward SHALL fall back to pure-nabla im2col (pad/slice/concat) without crashing

### Requirement: Conv2dMojoOp VJP rule
The system SHALL implement `vjp_rule` on `Conv2dMojoOp` that computes gradients using only nabla `matmul`, `reshape`, `pad`, and `concatenate`. The VJP rule SHALL NOT call `conv2d_transpose` or any cuDNN-dependent operation.

#### Scenario: VJP computes correct grad_input
- **WHEN** forward produces `Y = patches @ filter_flat` and cotangent `dL/dY` is provided
- **THEN** `grad_input` SHALL be computed as `col2im(cotangent_flat @ filter_flat.T)` reshaped to input shape

#### Scenario: VJP computes correct grad_filter
- **WHEN** forward produces `Y = patches @ filter_flat` and cotangent `dL/dY` is provided
- **THEN** `grad_filter` SHALL be computed as `reshape(patches.T @ cotangent_flat, filter.shape)`

#### Scenario: VJP works with multiple conv2d layers on GPU
- **WHEN** a loss function contains 2 or more `Conv2dMojoOp` calls and `nb.value_and_grad` is invoked with GPU tensors
- **THEN** all gradients SHALL be computed without SIGABRT or `cudnnCreate symbol not found`

### Requirement: Conv2dMojoOp compute_physical_shape
The system SHALL implement `compute_physical_shape` to return the correct output shape `(B, H_out, W_out, C_out)` given input and filter shapes, stride, and padding.

#### Scenario: Shape computation with padding
- **WHEN** input is `(1, 16, 16, 4)`, filter is `(3, 3, 4, 8)`, padding=(1,1,1,1), stride=1
- **THEN** output physical shape SHALL be `[(1, 16, 16, 8)]`

#### Scenario: Shape computation without padding
- **WHEN** input is `(1, 8, 8, 4)`, filter is `(3, 3, 4, 8)`, padding=(0,0,0,0), stride=1
- **THEN** output physical shape SHALL be `[(1, 6, 6, 8)]`
