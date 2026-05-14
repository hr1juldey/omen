## 1. Mojo GPU im2col Kernel

- [ ] 1.1 Create `src/omen/kernels/conv2d_im2col.mojo` — `@compiler.register("conv2d_im2col")` struct with `execute` using `foreach`. Input: `(B, H, W, C_in)` tensor + stride/padding/kernel_size scalars. Output: `(B·H_out·W_out, K·K·C_in)` flat patch matrix. Each output element reads one value from input (gather pattern, zero-padding for out-of-bounds).
- [ ] 1.2 Test im2col standalone: compile and run with a 1×4×4×3 input, 3×3 kernel, stride=1, padding=1. Verify patches against numpy `as_strided` reference. Test stride=2 case separately.

## 2. Mojo GPU col2im Kernel

- [ ] 2.1 Create `src/omen/kernels/conv2im.mojo` — `@compiler.register("conv2im_col2im")` struct with `execute` using `foreach`. Input: `(B·H_out·W_out, K·K·C_in)` column matrix + target_shape/stride/padding/kernel_size. Output: `(B, H, W, C_in)`. **Gather pattern**: iterate over output `(b, ih, iw, ci)`, accumulate from all column entries where `oh·stride+kh-pad == ih` (sequential inner loop, no atomics).
- [ ] 2.2 Test col2im standalone: verify inverse of im2col for stride=1 and stride=2 cases. `col2im(im2col(x))` should approximate `x` for stride=1 (exact when padding accounts for overlap).

## 3. Python Conv2dOp + conv2d_safe

- [ ] 3.1 Create `src/omen/kernels/conv2d.py` — `Im2colOp(Operation)` wrapping the Mojo im2col kernel. `compute_physical_shape` returns `[(B·H_out·W_out, K·K·C_in)]`. `kernel()` calls `call_custom_kernel("conv2d_im2col", ...)`.
- [ ] 3.2 Add `Col2imOp(Operation)` in same file — wraps Mojo col2im kernel. `compute_physical_shape` returns `[(B, H, W, C_in)]`. `kernel()` calls `call_custom_kernel("conv2im_col2im", ...)`.
- [ ] 3.3 Add `Conv2dOp(Operation)` — `compute_physical_shape` returns output NHWC shape. `kernel()` calls `Im2colOp` to get patches, then `nb.matmul(patches, filter_flat)`, then reshape to `(B, H_out, W_out, C_out)`.
- [ ] 3.4 Add `Conv2dOp.vjp_rule()` — pure Nabla ops: `im2col(x).T @ cot_flat` for grad_filter, `col2im(cot_flat @ filter_flat.T)` for grad_input. Returns `[grad_x, grad_w]`.
- [ ] 3.5 Add `conv2d_safe(x, filter, stride=1, padding=0, bias=None)` wrapper: normalize stride/padding to tuples, dispatch to Conv2dOp, add bias if provided.
- [ ] 3.6 Add `conv2d_safe` export to `src/omen/kernels/__init__.py`.

## 4. Finite-Difference Gradient Check Test

- [ ] 4.1 Create `tests/test_conv2d_gradients.py` — parameterized test that computes gradients via custom vjp_rule and via finite differences (perturb each weight by epsilon=1e-4). Assert relative error < 1e-3 for both grad_filter and grad_input. Test cases: stride=1, stride=2, padding=0, padding=1, small filters (3×3) and various channel counts (4→32, 64→128).

## 5. Replace conv2d Call Sites

- [ ] 5.1 Replace 8 `nb.conv2d` calls in `src/omen/model/decoder.py` with `conv2d_safe` — import from `omen.kernels`, preserve all stride/padding args.
- [ ] 5.2 Replace 3 `nb.conv2d` calls in `src/omen/model/scene_encoder.py` `RenderFeatureEncoder.forward` with `conv2d_safe`.

## 6. Clean Up Trainer

- [ ] 6.1 Delete `CONV2D_BLOCKERS` frozenset from `src/omen/training/trainer/core.py`.
- [ ] 6.2 Delete `_to_real` helper function.
- [ ] 6.3 Simplify `_realize_grads` — remove conv2d blocker branch.
- [ ] 6.4 Simplify `_apply_optimizer_updates` — remove `_to_real` calls on optimizer outputs.

## 7. End-to-End Verification

- [ ] 7.1 Run `tests/test_functional_trainer.py` — all 4 tests pass.
- [ ] 7.2 Run `scripts/test_e2e_training.py` — update to 1024×1024 if feasible, otherwise verify at 32×32 with new conv2d_safe.
- [ ] 7.3 Run `ruff check --fix` + `ruff format` on all changed files.
