# LeWorldModel (LeWM) -- Exact Technical Reference for Mojo Port

## 1. ARPredictor Architecture

### Top-Level Assembly (train.py)
```
encoder = ViT-Tiny(patch_size=14, image_size=224, pretrained=False)
projector = MLP(192 -> 2048 -> 192, norm=BatchNorm1d)
action_encoder = Embedder(input_dim=10, emb_dim=192)
predictor = ARPredictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
                        depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1)
pred_proj = MLP(192 -> 2048 -> 192, norm=BatchNorm1d)
model = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
```

### Tensor Shape Flow During Training

**Encoding phase:**
```
pixels: (B, T, C=3, H=224, W=224)
  -> rearrange to (B*T, 3, 224, 224)
  -> ViT-Tiny encoder (patch_size=14, 16x16=256 patches, seq_len=257 with CLS)
  -> last_hidden_state[:, 0] (CLS token): (B*T, 192)
  -> projector MLP: (B*T, 192) -> (B*T, 2048) -> (B*T, 192)
  -> rearrange back: (B, T, 192)   # this is `emb`

action: (B, T, action_dim=2)
  -> NaN replaced with 0
  -> Embedder: (B, T, 10) -- wait, action is dim=2, but effective_act_dim = frameskip * action_dim = 5*2 = 10
     Actually: action raw data is stacked over frameskip, so shape is (B, T, 10)
  -> Conv1d(10, 10, kernel=1): permute, conv, permute back
  -> MLP: (B, T, 10) -> (B, T, 768) -> (B, T, 192)   # this is `act_emb`
```

**Training context window (history_size=3, num_preds=1, total T=4):**
```
emb:     (B, 4, 192)
act_emb: (B, 4, 192)

ctx_emb = emb[:, :3]     # (B, 3, 192) -- first 3 timesteps
ctx_act = act_emb[:, :3]  # (B, 3, 192)

tgt_emb = emb[:, 1:]     # (B, 1, 192) -- target is emb at timestep 3 (n_preds=1)
                          # WAIT: tgt_emb = emb[:, n_preds:] = emb[:, 1:] = (B, 3, 192)
                          # pred_emb = predictor(ctx_emb, ctx_act) = (B, 3, 192)
                          # Actually num_steps = history_size + num_preds = 3 + 1 = 4
                          # ctx = first 3, tgt = last 1 (emb[:, n_preds:] takes from index 1 onward)
                          # But prediction is on ctx (first 3), and target is tgt (from n_preds=1)
```

**Clarification on training slices:**
```python
ctx_len = cfg.wm.history_size  # 3
n_preds = cfg.wm.num_preds     # 1

ctx_emb = emb[:, :ctx_len]           # (B, 3, 192)
ctx_act = act_emb[:, :ctx_len]       # (B, 3, 192)

tgt_emb = emb[:, n_preds:]           # (B, 3, 192) -- shifted by n_preds
pred_emb = model.predict(ctx_emb, ctx_act)  # (B, 3, 192)

pred_loss = (pred_emb - tgt_emb).pow(2).mean()
```

### ConditionalBlock -- AdaLN-Zero Conditioning

```python
class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        self.attn = Attention(dim=192, heads=16, dim_head=64, dropout=0.1)
        self.mlp = FeedForward(dim=192, hidden_dim=2048, dropout=0.1)
        self.norm1 = LayerNorm(192, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm(192, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = Sequential(
            SiLU(),
            Linear(192, 6*192=1152, bias=True)
        )
        # Zero-init the adaLN linear layer:
        init.constant_(adaLN_modulation[-1].weight, 0)
        init.constant_(adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        # c is the action embedding (conditioning signal)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            adaLN_modulation(c).chunk(6, dim=-1)   # each: (B, T, 192)

        x = x + gate_msa * attn(modulate(norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * mlp(modulate(norm2(x), shift_mlp, scale_mlp))
        return x
```

**modulate(x, shift, scale):** `x * (1 + scale) + shift`

**Key insight:** AdaLN modulation is conditioned on `c` (action embedding), NOT on `x` (state embedding). The action embedding tells each block HOW to modulate the state representation.

### Transformer Internal Projections

```python
class Transformer(nn.Module):
    # input_dim=192, hidden_dim=192, output_dim=192
    self.input_proj = Identity()       # 192 == 192
    self.cond_proj = Identity()        # 192 == 192
    self.output_proj = Identity()      # 192 == 192
    self.norm = LayerNorm(192)
    self.layers = ModuleList([ConditionalBlock(...) for _ in range(6)])

    def forward(self, x, c=None):
        x = input_proj(x)
        c = cond_proj(c)  # only if c is not None
        for block in layers:
            x = block(x, c)  # ConditionalBlock
        x = norm(x)
        x = output_proj(x)
        return x
```

### ARPredictor Forward

```python
class ARPredictor:
    def forward(self, x, c):
        # x: (B, T=3, 192) state embeddings
        # c: (B, T=3, 192) action embeddings
        T = x.size(1)  # 3
        x = x + pos_embedding[:, :T]  # learnable pos: (1, 3, 192)
        x = dropout(x)                  # emb_dropout=0.0 (no-op)
        x = transformer(x, c)           # 6 ConditionalBlocks
        return x  # (B, 3, 192)
```

### Attention Layer

```python
class Attention:
    # dim=192, heads=16, dim_head=64, inner_dim=1024
    self.to_qkv = Linear(192, 1024*3=3072, bias=False)
    self.to_out = Sequential(Linear(1024, 192), Dropout(0.1))
    self.norm = LayerNorm(192)

    def forward(x, causal=True):
        x = norm(x)
        q, k, v = to_qkv(x).chunk(3, dim=-1)  # each: (B, T, 1024)
        q, k, v = rearrange(each, "b t (h d) -> b h t d", h=16)  # (B, 16, T, 64)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = rearrange(out, "b h t d -> b t (h d)")  # (B, T, 1024)
        return to_out(out)  # (B, T, 192)
```

### FeedForward Layer

```python
class FeedForward:
    self.net = Sequential(
        LayerNorm(192),
        Linear(192, 2048),
        GELU(),
        Dropout(0.1),
        Linear(2048, 192),
        Dropout(0.1),
    )
```

## 2. SIGReg Loss -- Step by Step

### Initialization
```python
knots = 17
num_proj = 1024
t = linspace(0, 3, 17)                      # evaluation points on [0, 3]
dt = 3 / 16                                  # = 0.1875
weights = full(17, 2*dt)                      # trapezoidal rule interior weights = 0.375
weights[0] = weights[-1] = dt                # boundary weights = 0.1875
phi = exp(-t^2 / 2)                           # Gaussian characteristic function phi(t) = E[e^{itX}] for N(0,1)
                                              # = exp(-t^2/2) since E[cos(tX)] = exp(-t^2/2) for Gaussian
weights = weights * phi                       # combined: trapezoidal * Gaussian CF
```

### Forward Pass
```python
def forward(self, proj):
    # proj: (T, B, D=192) -- NOTE: transposed from training, time-first
    # T = sequence length, B = batch, D = embedding dim

    # Step 1: Sample random projection vectors
    A = randn(192, 1024)                       # random projection matrix
    A = A / A.norm(p=2, dim=0)                 # normalize each column to unit length

    # Step 2: Project embeddings onto random directions
    projected = proj @ A                        # (T, B, 1024)

    # Step 3: Compute empirical characteristic function
    x_t = projected.unsqueeze(-1) * self.t     # (T, B, 1024, 17) -- multiply each proj by each t value

    # Step 4: Compare empirical CF to theoretical Gaussian CF
    # E[cos(t*X)] for data vs exp(-t^2/2) for Gaussian
    cos_diff = (x_t.cos().mean(dim=0) - phi).square()  # mean over T (dim=-3) -> (B, 1024, 17)
    sin_diff = (x_t.sin().mean(dim=0)).square()         # sin component -> (B, 1024, 17)
    err = cos_diff + sin_diff                           # (B, 1024, 17)

    # Step 5: Weighted integration using trapezoidal rule
    statistic = (err @ weights) * T              # (B, 1024) * T -- scale by sample count

    # Step 6: Average over projections and batch
    return statistic.mean()                      # scalar loss
```

**Mathematical meaning:** The Epps-Pulley test statistic measures deviation of the empirical characteristic function from the Gaussian characteristic function. For a standard normal, E[e^{itX}] = exp(-t^2/2). The loss drives embeddings toward isotropic Gaussian distribution.

**Loss coefficient:** lambda = 0.09
**Total loss:** `pred_loss + 0.09 * sigreg_loss`

## 3. Training Loop

### Loss Computation
```python
def lejepa_forward(self, batch, stage, cfg):
    batch["action"] = nan_to_num(batch["action"], 0.0)  # NaN at sequence boundaries

    output = model.encode(batch)
    emb = output["emb"]       # (B, T=4, D=192)
    act_emb = output["act_emb"]  # (B, T=4, D=192)

    ctx_emb = emb[:, :3]       # (B, 3, 192)
    ctx_act = act_emb[:, :3]   # (B, 3, 192)

    tgt_emb = emb[:, 1:]       # (B, 3, 192) -- target shifted by num_preds=1
    pred_emb = model.predict(ctx_emb, ctx_act)  # (B, 3, 192)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()         # MSE prediction loss
    sigreg_loss = sigreg(emb.transpose(0, 1))               # (T, B, D) format
    loss = pred_loss + 0.09 * sigreg_loss
```

### Optimizer
```python
optimizer = AdamW(lr=5e-5, weight_decay=1e-3)
scheduler = LinearWarmupCosineAnnealingLR  # warmup + cosine annealing, per epoch
gradient_clip_val = 1.0
precision = bf16
```

### Training Config
```
max_epochs: 100
batch_size: 128
train_split: 0.9
seed: 3072
num_workers: 6
```

## 4. History Window Management

### During Training
- `num_steps = history_size + num_preds = 3 + 1 = 4` timesteps loaded per sample
- `ctx_emb = emb[:, :history_size]` -- first 3 frames as context
- `tgt_emb = emb[:, num_preds:]` -- frames from index 1 onward as targets
- This creates a teacher-forcing setup where the predictor sees 3 frames and predicts the next 3 (shifted)

### During Rollout (Inference)
```python
HS = history_size  # 3
for t in range(n_steps):
    act_emb = action_encoder(act)              # encode actions
    emb_trunc = emb[:, -HS:]                   # (BS, 3, 192) -- last 3 embeddings
    act_trunc = act_emb[:, -HS:]               # (BS, 3, 192) -- last 3 action embeddings
    pred_emb = predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, 192) -- take LAST prediction
    emb = cat([emb, pred_emb], dim=1)          # append to history
    # Actions are carried forward from act_future
```

**Key detail:** The predictor outputs T predictions (one per input position), but during rollout only the LAST one (`[:, -1:]`) is used as the next-step prediction. The history window slides by truncating to `emb[:, -HS:]`.

## 5. Surprise Detection

Surprise detection is implemented via the `AutoCostModel` in the `stable_worldmodel` library (external, not in the repo). Based on the paper and README:

- **Prediction error** is measured as MSE between predicted and actual embeddings
- **Surprise** = high prediction error indicates physically implausible events
- The model is used as a cost model in `WorldModelPolicy` with planning via `PlanConfig`
- **No explicit threshold in the codebase** -- surprise is a continuous signal used for planning cost
- The paper confirms: "Surprise evaluation confirms that the model reliably detects physically implausible events"
- History clearing: not explicitly in the codebase; during rollout, history is managed by the sliding window `emb[:, -HS:]`

## 6. Model Dimensions Summary

### Per-Component Parameters

| Component | Parameters | Details |
|-----------|-----------|---------|
| ViT-Tiny Encoder | ~5.5M | 12 layers, hidden=192, heads=3, mlp=768 |
| Projector MLP | ~0.8M | 192->2048->192, BatchNorm1d |
| Action Encoder (Embedder) | ~0.16M | Conv1d(10,10,k=1) + MLP(10->768->192) |
| ARPredictor (6 ConditionalBlocks) | ~10.8M | depth=6, hidden=192, heads=16, dim_head=64, mlp=2048 |
| pred_proj MLP | ~0.8M | 192->2048->192, BatchNorm1d |
| SIGReg | 0 | All buffers, no learnable params |
| **Total** | **~18M** | Paper states ~15M (may count differently) |

### Per ConditionalBlock Parameters

| Sub-component | Params | Breakdown |
|--------------|--------|-----------|
| Attention.to_qkv | 589,824 | Linear(192, 3072, bias=False) |
| Attention.to_out | 196,608 | Linear(1024, 192) |
| Attention.norm | 192 | LayerNorm(192) |
| FeedForward | 786,432 | LN(192) + Linear(192,2048) + Linear(2048,192) |
| AdaLN modulation | 222,336 | SiLU + Linear(192, 1152, bias=True) |
| **Per block** | **~1.80M** | |

### Key Dimensions Cheat Sheet

```
ViT encoder:
  - hidden_size: 192
  - num_heads: 3
  - head_dim: 64 (192/3)
  - mlp_dim: 768 (4x hidden)
  - depth: 12
  - patch_size: 14
  - image_size: 224
  - num_patches: 256
  - seq_len: 257 (with CLS)

Predictor (ARPredictor):
  - input_dim: 192 (embed_dim)
  - hidden_dim: 192 (same as ViT hidden)
  - output_dim: 192 (same as hidden)
  - depth: 6
  - heads: 16
  - dim_head: 64
  - inner_dim (attn): 1024 (16*64)
  - mlp_dim: 2048

Projection MLPs:
  - hidden_dim: 2048
  - norm: BatchNorm1d

Embedding space:
  - embed_dim: 192

Action encoder:
  - input: effective_act_dim = frameskip * action_dim = 5 * 2 = 10
  - Conv1d(10, 10, kernel_size=1)
  - MLP: 10 -> 768 -> 192

SIGReg:
  - knots: 17
  - num_proj: 1024
  - t range: [0, 3]
  - weight (lambda): 0.09
```

### Data Shapes (PushT example)

```
Input:
  pixels:  (B=128, T=4, 3, 224, 224)
  action:  (B=128, T=4, 10)  [frameskip=5 stacked, raw action_dim=2]

After encoding:
  emb:     (B=128, T=4, 192)
  act_emb: (B=128, T=4, 192)

Training slices:
  ctx_emb:  (128, 3, 192)
  ctx_act:  (128, 3, 192)
  tgt_emb:  (128, 3, 192)
  pred_emb: (128, 3, 192)

SIGReg input:
  emb.transpose(0,1): (4, 128, 192)

Rollout:
  emb_trunc:  (BS, 3, 192)
  act_trunc:  (BS, 3, 192)
  pred:       (BS, 1, 192)  [last timestep only]
```
