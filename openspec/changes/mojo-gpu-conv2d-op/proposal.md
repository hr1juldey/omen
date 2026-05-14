## Why

MAX compiler cannot infer `num_groups` when compiling backward graphs through `nb.conv2d`, blocking all 8 conv2d filter params from training and causing memory explosions during gradient realization. A custom `Conv2dOp(Operation)` with Mojo GPU im2col kernel + Nabla matmul backward bypasses the broken autodiff-generated conv2d backward entirely.

Training at 32×32 is semantically useless — no caustics, no indirect bounces, no penumbra. The system trains at **1024×1024 minimum** where real light transport phenomena appear. The training pipeline is **1 frame = 1 training point**: geometry nodes generate scene variation → Mitsuba renders GT (high SPP) + noisy (low SPP) + scene graph → Nabla trains 1 step → discard. No buffering, no streaming. Render and train live.

## What Changes

- New `Conv2dOp(Operation)` with Mojo GPU `im2col` forward kernel + `col2im` scatter kernel + Nabla `matmul` backward in `vjp_rule`
- New `conv2d_safe(x, filter, stride, padding, bias)` drop-in replacement for `nb.conv2d`
- 2 Mojo GPU kernels: `conv2d_im2col` (patch extraction), `conv2d_col2im` (scatter-back for grad_input)
- `vjp_rule` uses `nb.matmul` for gradient math — no custom Mojo backward kernels, no race conditions
- Replace all 11 `nb.conv2d` calls: 8 in `Decoder` + 3 in `RenderFeatureEncoder`
- Remove `CONV2D_BLOCKERS` freeze and `_to_real` workarounds from trainer
- Training at 1024×1024 with 10-channel AOV aux buffer (albedo 3 + normal 3 + depth 1 + material_id 1 + motion 2), mixed precision
- 4K inference via tiled processing (16 × 1024×1024 tiles)

## Capabilities

### New Capabilities
- `mojo-conv2d-op`: Custom nabla Operation with Mojo GPU im2col/col2im kernels, matmul-based vjp_rule, drop-in `conv2d_safe()` API. Resolution-agnostic.

### Modified Capabilities
- `online-training`: Remove conv2d filter freezes, enable all 139 params to train, 1024×1024 default tile size, render→train→discard pipeline

## Impact

- **New files**: `src/omen/kernels/conv2d.py`, `src/omen/kernels/conv2d_im2col.mojo`, `src/omen/kernels/conv2im_col2im.mojo`
- **Modified files**: `src/omen/kernels/__init__.py` (export), `src/omen/model/decoder.py` (8 call sites), `src/omen/model/scene_encoder.py` (3 call sites), `src/omen/training/trainer/core.py` (remove blockers)
- **Per-frame memory at 1024×1024**: RGB pair 25 MB + AOV aux 42 MB + scene graph 12 KB = **~67 MB**. Model + optimizer: 256 MB. Training workspace (mixed): ~1.5 GB. **Total peak: ~2 GB GPU**.
- **AOV channels**: Mitsuba renders 10 packed channels (albedo, normal, depth, material_id, motion_vectors). Must-have: albedo+normal+depth (7 ch). Nice-to-have: material_id+motion (3 ch, zero-filled if unavailable). NOT 47 layers (Cycles) — the code normalizes to 10 ch.
- **Pipeline**: 1 frame = 1 step. Mitsuba renders on CPU while Nabla trains previous frame on GPU. Geometry nodes auto-generate unlimited variations. Weekend training: ~86K frames × ~2 steps/frame = ~172K steps.
