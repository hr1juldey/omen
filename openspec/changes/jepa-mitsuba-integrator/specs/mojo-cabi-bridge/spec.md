## ADDED Requirements

### Background: DLPack Bridge (NOT C ABI)

Both Mitsuba (Dr.Jit) and Nabla are Python libraries. No C ABI bridge is needed — tensor interop uses DLPack via `nb.Tensor.from_dlpack()`. Nabla's `@nb.compile` JIT-compiles Python to MAX/Mojo for GPU execution. Custom Mojo kernels are loaded on-demand via `call_custom_kernel()`.

This spec replaces the original C ABI bridge design. No `libomen.so`, no ctypes, no C header files.

### Requirement: DLPack tensor bridge between Mitsuba and Nabla

Omen SHALL transfer rendered tensors from Mitsuba/Dr.Jit to Nabla using DLPack zero-copy.

#### Scenario: Transfer Dr.Jit render to Nabla tensor

- **WHEN** Mitsuba render produces a Dr.Jit TensorXf
- **THEN** convert to numpy: `noisy_np = np.array(mi.render(scene, spp=4))`
- **AND** create Nabla tensor: `noisy_nb = nb.Tensor.from_dlpack(noisy_np)`
- **AND** transfer to GPU: `noisy_nb = noisy_nb.cuda()`
- **AND** verify shape: `noisy_nb.shape == (H, W, 3)` or `(H, W, 4)` with alpha
- **AND** no C ABI, no ctypes, no shared library — pure Python

#### Scenario: Transfer Nabla output back to numpy

- **WHEN** JEPA inference produces a Nabla tensor
- **THEN** transfer to CPU: `output_cpu = clean.cpu()`
- **AND** convert to numpy: `output_np = output_cpu.to_numpy()`
- **AND** save or display: `imageio.imwrite("output.exr", output_np)`

#### Scenario: Handle GPU memory allocation

- **WHEN** both Mitsuba and Nabla use the same GPU
- **THEN** Nabla tensors allocated via `.cuda()` use the same GPU memory space
- **AND** DLPack handles zero-copy when possible (no memcpy)
- **AND** if GPU unavailable: fall back to CPU tensors (no error)

### Requirement: Custom Mojo GPU kernel loading via Nabla

Omen SHALL load custom Mojo GPU kernels (SIGReg, merge) via Nabla's `call_custom_kernel()`.

#### Scenario: Load SIGReg kernel

- **WHEN** SIGRegOp is instantiated
- **THEN** define `class SIGRegOp(UnaryOperation)` with:
  - `name` property returns `"sigreg_kernel"`
  - `kernel(args, kwargs)` calls `call_custom_kernel("sigreg_kernel", kernel_dir, ...)`
  - `vjp_rule()` provides reverse-mode autograd rule
- **AND** kernel lives in `kernels/sigreg_kernel.mojo`
- **AND** Nabla compiles the Mojo kernel on-demand (no separate build step)

#### Scenario: Load merge kernel

- **WHEN** merge operation is needed for multi-res mode
- **THEN** define `class MergeOp(Operation)` with:
  - `name` property returns `"merge_kernel"`
  - `kernel()` calls `call_custom_kernel("merge_kernel", kernel_dir, low_res, high_res, edge_map, ...)`
  - `vjp_rule()` provides gradient for training
- **AND** kernel lives in `kernels/merge_kernel.mojo`

#### Scenario: Custom op autograd integration

- **WHEN** custom op is used inside `nb.grad` or `loss.backward()`
- **THEN** Nabla calls `vjp_rule()` to compute gradients through the custom op
- **AND** custom ops compose with `nb.vmap`, `@nb.compile`
- **AND** for elementwise ops: implement `_derivative(primals, output)` instead of full vjp

### Requirement: Inference compilation

Omen SHALL compile inference paths using `@nb.compile` for production performance.

#### Scenario: Compile denoiser inference

- **WHEN** model is loaded for production inference
- **THEN** define compiled function:
  ```python
  @nb.compile
  def omen_denoise_compiled(model_weights, noisy_render, scene_features):
      latent = encode(model_weights, noisy_render, scene_features)
      clean = decode(model_weights, latent)
      return clean
  ```
- **AND** Nabla traces the function, optimizes the graph, compiles to MAX
- **AND** subsequent calls execute the compiled graph (no Python overhead)
- **AND** LRU cache handles different input shapes automatically

#### Scenario: Production export (MAX Engine C API)

- **WHEN** model needs to run without Python ML runtime
- **THEN** compile model to MAX format: `.max` file
- **AND** load via MAX Engine C API: `M_compileModel()` → `M_initModel()` → `M_executeModelSync()`
- **AND** only Mitsuba (Python) needed at runtime — no Nabla/Python ML dependency
- **AND** this is a FUTURE optimization — initial version uses Python Nabla

### Requirement: Error handling for bridge operations

Omen SHALL gracefully handle tensor transfer failures.

#### Scenario: GPU memory insufficient

- **WHEN** `.cuda()` fails due to OOM
- **THEN** catch the error
- **AND** log: "GPU OOM, falling back to CPU inference"
- **AND** run inference on CPU (slower but functional)

#### Scenario: DLPack conversion fails

- **WHEN** `nb.Tensor.from_dlpack()` raises an error
- **THEN** fall back: `np.array()` intermediate copy
- **AND** log: "DLPack failed, using numpy copy (10-50ms overhead)"
- **AND** proceed with copied tensor
