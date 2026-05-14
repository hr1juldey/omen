## Context

Omen's trainer uses `nb.value_and_grad` to differentiate through the full model. The built-in `nb.conv2d` backward triggers MAX compiler `num_groups` inference failure — 8 decoder + 3 scene_encoder conv2d filter params are frozen. All 5 existing Mojo kernels use `UnaryOperation` without `vjp_rule`.

Training at 32×32 is semantically insufficient. At 1024×1024, caustics, multi-bounce indirect, area light soft shadows, subsurface scattering, and volume scattering all become visible — this is the minimum resolution where noise structure is complex enough for a denoiser to learn.

The live pipeline: Mitsuba renders on CPU (scalar_rgb variant) while Nabla trains on GPU. They are naturally parallel. Geometry nodes generate unlimited scene variations — the system trains over weekends on auto-generated data.

## Goals / Non-Goals

**Goals:**
- Replace all 11 `nb.conv2d` calls with `conv2d_safe` backed by Mojo im2col + Nabla matmul
- Provide custom `vjp_rule` using `nb.matmul` (no custom Mojo backward — avoids accumulation loops and race conditions)
- Enable all 139 params to train (remove CONV2D_BLOCKERS)
- Support 1024×1024 training tiles with 10-channel AOV aux buffer (albedo 3 + normal 3 + depth 1 + material_id 1 + motion 2)
- Support tiled 4K inference (16 × 1024 tiles)
- Keep identical numerical behavior to `nb.conv2d` (NHWC input, HWIO filter)

**Non-Goals:**
- Conv2dTranspose (decoder uses pixel shuffle, not transposed conv)
- Grouped or dilated convolutions (not used in the model)
- 4K single-pass inference (requires 32 GB VRAM — tiled processing only)
- Multi-GPU distribution (future work)
- Real-time 120fps at 4K on single GPU (~1-2 fps achievable, multi-GPU needed for production rate)

## Decisions

### 1. `Operation` base class (not `UnaryOperation`)

`Conv2dOp` has two inputs (x, filter) and needs `vjp_rule`. `UnaryOperation` only supports `_derivative` for elementwise ops. `Operation.vjp_rule(primals, cotangents, outputs, kwargs) -> list[Tensor | None]` is the correct API.

### 2. im2col + matmul (not direct convolution Mojo kernels)

**Why not direct Mojo backward kernels**: Writing accumulation loops in Mojo for `grad_filter` and `grad_input` means ~350 LOC of boundary handling, stride math, and sequential inner loops per thread. At 4K, the inner loop for `grad_filter` is O(8M) iterations per thread.

**Why im2col + matmul**: The hybrid approach uses Mojo only for data rearrangement (im2col/col2im), and Nabla's proven `nb.matmul` for the gradient math.

| | Direct (3 Mojo kernels) | Hybrid (2 Mojo + matmul) |
|---|---|---|
| Mojo LOC | ~350 | ~120 |
| Bug surface | accumulation loops × 3 | data rearrangement × 2 |
| Backward correctness | our code | Nabla's matmul (battle-tested) |
| vjp_rule | 2 extra Operation subclasses | pure nb.matmul/reshape/permute |
| Performance at 4K | O(8M) sequential per thread | GEMM (tiled, optimized by MAX) |

**Forward**: `im2col(x) → patches (B·H_out·W_out, K·K·C_in)`, then `patches @ filter_flat → output_flat → reshape(B, H_out, W_out, C_out)`

**vjp_rule filter gradient**: `grad_filter = im2col(x).T @ cotangent_flat → reshape(K, K, C_in, C_out)`
**vjp_rule input gradient**: `col = cotangent_flat @ filter_flat.T → col2im(col) → grad_x`

The matmul operations produce traceable Nabla tensors. The im2col/col2im are Mojo GPU kernels for the memory-bound data rearrangement. No race conditions — matmul handles parallelism internally, im2col/col2im are output-parallel (gather pattern).

### 3. im2col Mojo GPU kernel

Registered as `@compiler.register("conv2d_im2col")`. Each output element `(b, oh, ow, kh, kw, ci)` reads one value from `x[b, oh·s+kh-p, ow·s+kw-p, ci]` (with boundary zero). Output is flat: `(B·H_out·W_out, K·K·C_in)`.

Gather pattern — each output position is written by exactly one thread. No race conditions.

### 4. col2im Mojo GPU kernel

Registered as `@compiler.register("conv2d_col2im")`. Input: flat column matrix `(B·H_out·W_out, K·K·C_in)`. For each position `(b, kh, kw, ci)`, **accumulates** (atomic add) into `grad_x[b, oh·s+kh-p, ow·s+kw-p, ci]`.

**This IS scatter-add** — multiple column entries map to the same output position (overlapping patches). Requires atomic adds. This is the one place where we need atomics, but:
- The accumulation is a simple FP32 add (Mojo `Atomic.fetch_add`)
- Only happens when stride < kernel_size (which is always for our 3×3 with stride=1 convs)
- For stride=2, fewer overlaps → fewer atomic collisions

Alternative: iterate over output positions (gather pattern), accumulate by reading from column matrix. This avoids atomics but requires indexing into the flat column matrix — more complex but doable. **Decision: use gather pattern for col2im too** — iterate over `(b, ih, iw, ci)`, gather from all patches that contribute.

### 5. `conv2d_safe()` Python wrapper

Drop-in replacement for `nb.conv2d`:
```python
def conv2d_safe(x, filter, stride=1, padding=0, bias=None):
    """Mojo GPU im2col + Nabla matmul conv2d. Drop-in for nb.conv2d."""
```
Handles stride/padding normalization (int → tuple), bias addition, and dispatches to `Conv2dOp`.

### 6. `vjp_rule` implementation

Pure Nabla operations — no Mojo kernels called from vjp_rule:
```python
def vjp_rule(self, primals, cotangents, outputs, kwargs):
    x, w = primals
    cot = cotangents[0]
    stride, padding = kwargs['stride'], kwargs['padding']

    # im2col(x) — Mojo GPU kernel (through Im2colOp)
    patches = im2col_op(x, stride, padding, w.shape[:2])  # (B*OH*OW, KK*Ci)

    # grad_filter = patches.T @ cot_flat  → (KK*Ci, Co)
    cot_flat = cot.reshape((-1, Co))
    grad_w = (patches.T @ cot_flat).reshape(w.shape)

    # grad_input = col2im(cot_flat @ w_flat.T)  → (B, H, W, Ci)
    w_flat = w.reshape((K*K*Ci, Co))
    col = cot_flat @ w_flat.T
    grad_x = col2im_op(col, x.shape, stride, padding, w.shape[:2])

    return [grad_x, grad_w]
```

### 7. Training resolution and memory

| Resolution | FP32 | Mixed (FP16 fwd / FP32 bwd) | Fits RTX 3060? |
|---|---|---|---|
| 1024×1024 (training) | 2.81 GB | 2.25 GB | Yes (8 GB headroom) |
| 2048×2048 | 10.46 GB | 8.24 GB | Borderline |
| 4K (tiled, 16×1024) | 2.81 GB per tile | 2.25 GB per tile | Yes |

Training at 1024×1024 with mixed precision: **~1.8 GB GPU VRAM** (256 MB model + 1.5 GB workspace + 67 MB per-frame data).

Per-frame data at 1024×1024 (render→train→discard, NOT buffered):
- RGB pair (GT + noisy): 2 × 1024 × 1024 × 3 × 4 = 25 MB
- AOV aux buffer (10 ch packed): 1024 × 1024 × 10 × 4 = 42 MB
- Scene graph: ~12 KB
- **Total per frame: ~67 MB**

### 8. Live render→train→discard pipeline

**1 frame = 1 training point. No streaming, no buffering.**

```
Geometry Nodes → scene variation
       ↓
CPU (Mitsuba)              GPU (Nabla)
─────────────────          ─────────────────
Render frame N    ──copy──→  Train 1 step on frame N
 GT(high SPP)       5ms       then DISCARD frame N
 +noisy(low SPP)
 +scene_graph
 +AOV(10ch)
       ↓                         ↓
  Render frame N+1          Model updated
  (next variation)          Ready for N+1
```

- Mitsuba renders on CPU (scalar_rgb, ~2-10s per 1024×1024 frame at 32-128 spp)
- Nabla trains 1 step on GPU (~1s per step at 1024×1024)
- CPU→GPU transfer: ~67 MB per frame (GT + noisy + AOV + scene_graph), ~1ms over PCIe
- **No double-buffer**: render→train→discard. Only one frame in GPU memory at a time.
- Geometry nodes generate scene variations automatically (camera, materials, lights)
- If logging needed: save image to disk. Else: discard and move on.
- Weekend (48h): ~86K renders × 2 train steps = ~172K training steps

AOV channels (Mitsuba, 10 packed):
- **Must-have**: albedo (3 ch) + normal (3 ch) + depth (1 ch) = 7 channels — always available
- **Nice-to-have**: material_id (1 ch) + motion_vectors (2 ch) = 3 channels — zero-filled if unavailable
- Packed into `(H, W, 10)` aux buffer by existing `aov_pack` Mojo kernel

### 9. Parallelization strategy (no race conditions)

All kernels use **gather pattern** (iterate over output, read from input):
- **im2col**: `foreach` over `(flat_idx)` → writes `patches[flat_idx, kh·kw·ci]` by reading `x[b, oh·s+kh-p, ow·s+kw-p, ci]`. One thread per output element.
- **col2im**: `foreach` over `(b, ih, iw, ci)` → accumulates from all `(oh, ow, kh, kw)` positions in column matrix where `oh·s+kh-p == ih`. One thread per output element, sequential inner accumulation.
- **matmul**: Handled by Nabla/MAX compiler internally (GEMM, proven).

## Risks / Trade-offs

- **First `Operation` with `vjp_rule`**: Pattern proven in nabla's built-in ops but first in our kernels. Mitigated by matching exact signature from nabla source.
- **im2col memory expansion**: Expands input by K·K factor (9× for 3×3 filters). At 1024×1024 with 3 input channels: 27M elements = ~108 MB. At 256 channels: ~2.3 GB. Largest im2col is `d4` concat (512 channels) at 256×256: ~590 MB. Acceptable within 2 GB budget.
- **col2im gather vs atomic**: Gather pattern avoids atomics but requires non-trivial indexing into flat column matrix. Must correctly compute which column entries contribute to each output position.
- **4K production 120fps**: Single RTX 3060 achieves ~1-2 fps at 4K tiled. Production 120fps requires multi-GPU or dedicated inference hardware. Training and inference have different GPU requirements — training is the bottleneck we solve now.
- **Bias gradient**: Not handled in custom kernels — flows through Nabla's built-in addition autograd automatically. No kernel needed.
