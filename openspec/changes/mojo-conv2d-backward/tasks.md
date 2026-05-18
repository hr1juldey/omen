## 0. Mandatory Skill Prerequisites

- [x] 0.1 **MANDATORY**: Before writing any Mojo code, invoke `/mojo-syntax` skill to ensure correct current Mojo syntax (not outdated docs/examples)
- [x] 0.2 **MANDATORY**: Before writing any Mojo GPU kernel code, invoke `/mojo-gpu-fundamentals` skill for correct GPU targeting patterns (NVIDIA, foreach, InputTensor/OutputTensor)
- [x] 0.3 **MANDATORY**: Before writing any Python-Mojo interop (call_custom_kernel, Operation wrapper), invoke `/mojo-python-interop` skill for correct bridge patterns
- [x] 0.4 **MANDATORY**: If any Mojo kernel fails with `std::bad_cast` or compilation error, re-invoke `/mojo-syntax` + `/mojo-gpu-fundamentals` before debugging — verify syntax matches current Mojo version

## 1. Core Operation Implementation

- [x] 1.1 Create `tests/test_gpu_mojo_conv2d_backward.py` with boilerplate: imports, device setup, RAM guard, helper functions (to_dev, to_cpu, _rand, _zeros)
- [x] 1.2 Implement pure-nabla `_im2col(x, kh, kw, sh, sw, ph, pw)` function (pad/slice/concat patch extraction) — NO Mojo needed, pure nabla ops
- [x] 1.3 Implement pure-nabla `_col2im_scatter(col, B, H, W, C_in, kh, kw, sh, sw, ph, pw)` function (scatter patches back to image via pad+add) — NO Mojo needed, pure nabla ops
- [x] 1.4 Implement `Conv2dMojoOp(Operation)` with `name`, `kernel`, `compute_physical_shape` — forward uses pure-nabla im2col + matmul (reuses 1.2)
- [x] 1.5 Implement `Conv2dMojoOp.vjp_rule` — grad_filter via `patches.T @ cotangent`, grad_input via `_col2im_scatter(cotangent @ filter.T)` (reuses 1.3)
- [x] 1.6 Add convenience function `mojo_conv2d(x, filter, stride=1, padding=0, bias=None)` wrapping the op
- [ ] 1.7 **OPTIONAL acceleration**: If pure-nabla path works, try adding Mojo GPU im2col via `call_custom_kernel("conv2d_im2col", ...)`. Use `/mojo-python-interop` for bridge. Fall back to pure-nabla on any failure.

## 2. Forward Correctness Verification

- [ ] 2.1 Write test: single conv2d forward on CPU (1, 16, 16, 4) → (1, 16, 16, 8) with 3x3 filter, padding=1 — compare output against `nb.conv2d` native
- [ ] 2.2 Write test: verify output shapes for different padding/stride configurations (no pad, stride=2)

## 3. Backward Correctness Verification

- [ ] 3.1 Write numerical gradient checker: finite differences with eps=1e-3 on filter parameters, max_diff tolerance 0.1
- [ ] 3.2 Write test: single conv2d backward on GPU — verify `nb.value_and_grad` produces finite gradients matching numerical check
- [ ] 3.3 Write test: two conv2d layers backward on GPU (1, 16, 16, 4→8→16) — THIS IS THE CRITICAL TEST that native `nb.conv2d` cannot pass

## 4. Progressive Scale-Up

- [ ] 4.1 Implement Phase 1: 16x16, 4→8, 1 conv2d forward+backward on GPU with gradient verification
- [ ] 4.2 Implement Phase 2: 16x16, 4→8→16, 2 conv2d forward+backward on GPU (multi-layer proof)
- [ ] 4.3 Add `gc.collect()` + RSS check between phases; `sys.exit(99)` if RSS > 20GB
- [ ] 4.4 Implement Phase 3: 32x32, 4→16→32, 2 conv2d forward+backward on GPU
- [ ] 4.5 Implement Phase 4: 64x64, 4→16→32, 2 conv2d forward+backward on GPU (target resolution)

## 5. Training Loop Proof

- [ ] 5.1 Implement Phase 5: 64x64, 2+ conv2d layers + linear decoder, 10-step AdamW training loop on GPU
- [ ] 5.2 Verify all loss values are finite (no NaN, no inf) and no SIGABRT across all 10 steps
- [ ] 5.3 Print summary: "ALL PASSED — Multi-layer Mojo conv2d backward works on GPU" or list of failures
