# JEPA-Mitsuba Integration: Comprehensive Technical Research Report

## 1. Mitsuba 3 Differentiable Rendering

### 1.1 AD Variants and Architecture

Mitsuba 3's differentiable rendering is powered by **Dr.Jit**, a just-in-time compiler that records computation graphs for automatic differentiation. The `*_ad_*` variants (e.g., `cuda_ad_rgb`, `llvm_ad_rgb`) enable both forward and reverse-mode AD through the entire rendering pipeline.

Key integrators for differentiable rendering:
- **`path`**: Standard path tracer (naive AD - records full computation graph, memory-hungry)
- **`rb`**: Radiative Backpropagation (specialized adjoint pass, more efficient)
- **`prb`**: Path Replay Backpropagation (constant memory, linear time - **recommended** for most inverse rendering)

### 1.2 Making Scene Parameters Differentiable

```python
import mitsuba as mi
import drjit as dr

mi.set_variant('llvm_ad_rgb')  # or 'cuda_ad_rgb'

# Load scene with custom integrator
scene = mi.load_file('scene.xml', integrator='prb', res=256, max_depth=6)

# Traverse scene to get all parameters as a flat dictionary
params = mi.traverse(scene)

# Method 1: Manual gradient tracking
key = 'green.reflectance.value'
dr.enable_grad(params[key])
params.update()

# Method 2: Via optimizer (recommended for inverse rendering)
opt = mi.ad.Adam(lr=0.05)
opt[key] = params[key]
params.update(opt)
```

The `mi.traverse(scene)` call returns a `SceneParameters` object — a flat dictionary mapping string keys (e.g., `'light1.to_world'`, `'sphere.bsdf.reflectance.value'`) to their current values. `params.keep([keys])` can be used to only track specific parameters.

**Differentiable parameter types:**
- Material properties: albedo colors, roughness, metallic values
- Light properties: position, intensity, emission spectra
- Geometry: vertex positions, shape transforms
- Camera: pose, intrinsics, field of view
- Volume properties: density grids, albedo grids

### 1.3 The Inverse Rendering Loop with Adam Optimizer

```python
import mitsuba as mi
import drjit as dr

mi.set_variant('llvm_ad_rgb')

scene = mi.load_file('cbox.xml', integrator='prb', res=128)
params = mi.traverse(scene)

# Reference image (target)
ref_image = mi.render(scene, spp=512)

# Select parameter to optimize
key = 'red.reflectance.value'
opt = mi.ad.Adam(lr=0.05)
opt[key] = params[key]
params.update(opt)

# Optimization loop
for it in range(200):
    # Render with current parameters (vary seed each iteration for variance)
    image = mi.render(scene, params, seed=it, spp=8)

    # Compute L2 loss against reference
    loss = dr.sum(dr.sqr(image - ref_image)) / len(image)

    # Backpropagate through rendering equation
    dr.backward(loss)

    # Adam step on parameter
    opt.step()

    # Copy updated parameter back to scene
    params[key] = opt[key]
    params.update()

    print(f"Iteration {it}: loss={loss}", end='\r')
```

**Critical details:**
- `mi.render()` records the full computation graph when AD variant is active
- `dr.backward(loss)` triggers reverse-mode AD through the rendering equation
- `dr.forward(param)` triggers forward-mode AD (useful for gradient visualization)
- The optimizer holds its own copy of parameters; `params.update(opt)` syncs back to scene
- `seed=it` varies the random number sequence per iteration to reduce variance
- `spp=8` (low) during optimization, `spp=512` for reference — common practice

### 1.4 Forward vs Reverse Mode

```python
# Forward mode: "How does changing this parameter affect the image?"
image = mi.render(scene, params, spp=128)
dr.forward(params[key])           # Push gradient from param through graph
grad_image = dr.grad(image)       # Result: per-pixel gradient image

# Reverse mode: "How should this parameter change to reduce image loss?"
image = mi.render(scene, params, spp=128)
loss = dr.sum(dr.sqr(image - ref_image))
dr.backward(loss)                 # Pull gradient from loss back to params
param_grad = dr.grad(params[key]) # Result: parameter gradient
```

Forward mode is efficient when you have few inputs and many outputs. Reverse mode is efficient for many inputs and few outputs (scalar loss). Mitsuba routes `dr.forward()` to `Integrator.render_forward()` and `dr.backward()` to `Integrator.render_backward()`.

### 1.5 PyTorch Interoperability via dr.wrap()

```python
@dr.wrap(source='torch', target='drjit')
def render_texture(texture, spp=256, seed=1):
    params[key] = texture
    params.update()
    return mi.render(scene, params, spp=spp, seed=seed, seed_grad=seed+1)

# Now usable in PyTorch pipeline:
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
for i in range(iterations):
    optimizer.zero_grad()
    rendered_img = render_texture(model(input_texture), spp=4, seed=i)
    loss = nn.L1Loss()(rendered_img, target.torch())
    loss.backward()       # Gradients flow through Mitsuba back to PyTorch
    optimizer.step()
```

This is the key bridge for JEPA integration: `dr.wrap()` allows Mitsuba rendering to be a differentiable layer in any PyTorch computation graph.

---

## 2. LeWorldModel (LeWM) Architecture Details

### 2.1 Overall Architecture

LeWM is the first JEPA that trains stably end-to-end from raw pixels using only **two loss terms**:
1. **Prediction loss** (MSE between predicted and actual next-frame embeddings)
2. **SIGReg** (Sketched Isotropic Gaussian Regularization — prevents collapse)

~15M parameters, trainable on a single GPU in a few hours. No stop-gradients, no EMAs, no pretrained encoders.

**Components:**
- **Encoder**: DeiT/ViT-Small (patch_size=16) → 192-dim CLS token per frame
- **Projector**: MLP (input_dim → hidden_dim → output_dim, with BatchNorm1d)
- **Action encoder**: `Embedder` (MLP mapping actions to embedding space)
- **Predictor**: `ARPredictor` (action-conditioned autoregressive transformer)
- **Prediction projector**: MLP mapping predictor outputs to projected space

```python
model = JEPA(
    encoder=encoder,           # ViT-Small, extracts 192-dim CLS token
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),    # Maps encoder output to latent space
    pred_proj=mlp("pred_proj"),    # Maps predictor output to same space
)
```

### 2.2 ARPredictor: Exact Architecture

```python
class ARPredictor(nn.Module):
    def __init__(self, *, num_frames, depth, heads, mlp_dim,
                 input_dim, hidden_dim, output_dim=None,
                 dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim, hidden_dim, output_dim or input_dim,
            depth, heads, dim_head, mlp_dim, dropout,
            block_class=ConditionalBlock,  # <-- KEY: uses AdaLN-zero blocks
        )

    def forward(self, x, c):
        """
        x: (B, T, d)     — latent embeddings (concatenated history)
        c: (B, T, act_dim) — action embeddings
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]  # Add positional encoding
        x = self.dropout(x)
        x = self.transformer(x, c)          # Conditioned transformer
        return x
```

**Key detail**: The predictor does NOT concatenate embeddings with actions before the transformer. Instead, actions serve as **conditioning signals** through AdaLN-zero inside each ConditionalBlock. The transformer processes the embedding sequence `x` while being modulated by action embeddings `c` at every layer.

### 2.3 ConditionalBlock: AdaLN-Zero Conditioning

This is the core mechanism. Each transformer block uses **Adaptive Layer Normalization with zero-initialization** (from DiT - Scalable Diffusion Models with Transformers):

```python
class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # The condition c produces 6 modulation parameters:
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)  # 6 = (shift, scale, gate) x 2
        )

    def forward(self, x, c):
        # c (action embedding) → 6 modulation vectors via SiLU + Linear
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # Modulate attention: x * (1 + scale) + shift, then gate the residual
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # Modulate MLP similarly
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift
```

**How conditioning works:**
1. The action embedding `c` passes through `SiLU → Linear(dim, 6*dim)` producing 6 vectors
2. These split into: `(shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)`
3. Before attention: LayerNorm(x) is modulated as `x * (1 + scale) + shift`
4. The attention output is **gated** by `gate_attn` (multiplied elementwise)
5. Same pattern for MLP path with separate shift/scale/gate
6. **Zero-initialization**: The Linear is initialized so gates start near zero, meaning the block initially acts as identity — training stability trick

### 2.4 SIGReg: Sketched Isotropic Gaussian Regularization

SIGReg prevents representation collapse by enforcing that latent embeddings follow an isotropic Gaussian distribution. It uses the **Cramer-Wold theorem**: a high-dimensional distribution is Gaussian iff every 1D projection is Gaussian.

```python
class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        # Quadrature points on [0, 3] for numerical integration
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Trapezoidal rule endpoints
        window = torch.exp(-t.square() / 2.0)  # Gaussian characteristic function
        self.register_buffer("t", t)
        self.register_buffer("phi", window)  # Target: phi(t) = exp(-t^2/2)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D) — latent embeddings
        """
        # Sample random projection directions
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))  # Normalize columns

        # Project latents onto random directions: (T, B, num_proj)
        projected = proj @ A

        # Epps-Pulley normality test via characteristic function
        # For Gaussian: E[cos(tX)] = exp(-t^2/2), E[sin(tX)] = 0
        x_t = projected.unsqueeze(-1) * self.t  # (T, B, num_proj, knots)
        err_cos = (x_t.cos().mean(dim=0) - self.phi).square()  # deviation from Gaussian
        err_sin = x_t.sin().mean(dim=0).square()                # should be ~0
        err = err_cos + err_sin
        statistic = (err @ self.weights) * projected.size(0)
        return statistic.mean()  # Average over projections and time
```

**How it works:**
1. Projects D-dimensional latents onto `num_proj=1024` random unit directions
2. For each 1D projection, checks if the empirical characteristic function matches Gaussian's: `E[cos(tX)] = exp(-t²/2)` and `E[sin(tX)] = 0`
3. Uses quadrature (trapezoidal rule with 17 knots on [0,3]) to compute the Epps-Pulley normality statistic
4. Minimizing this forces the embedding distribution toward isotropic Gaussian
5. **Default**: `knots=17`, `num_proj=1024`, `lambda=0.1` (only real hyperparameter)

### 2.5 Full Training Forward Pass

```python
# In JEPA class:
def encode(self, info):
    pixels = info['pixels'].float()                          # (B, T, C, H, W)
    pixels = rearrange(pixels, "b t ... -> (b t) ...")       # Flatten to (B*T, C, H, W)
    output = self.encoder(pixels, interpolate_pos_encoding=True)
    pixels_emb = output.last_hidden_state[:, 0]              # CLS token: (B*T, 192)
    emb = self.projector(pixels_emb)                          # Project: (B*T, D)
    info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)   # (B, T, D)
    info["act_emb"] = self.action_encoder(info["action"])     # (B, T, act_emb_dim)
    return info

def predict(self, emb, act_emb):
    preds = self.predictor(emb, act_emb)                      # ARPredictor forward
    preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
    return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

# Training loss:
# L_pred = MSE(predict(emb[:, :-1], act_emb[:, :-1]), emb[:, 1:])  # Teacher-forced
# L_sigreg = SIGReg(emb)
# L_total = L_pred + lambda * L_sigreg   (lambda=0.1)
```

### 2.6 Rollout with History Size Truncation

```python
def rollout(self, info, action_sequence, history_size=3):
    B, S, T = action_sequence.shape[:3]
    act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)

    # Encode initial frames
    _init = self.encode(_init)
    emb = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)

    HS = history_size
    for t in range(n_steps):
        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]       # Only last HS frames (history truncation)
        act_trunc = act_emb[:, -HS:]   # Only last HS actions
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # Predict next: (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)  # Append to history

        # Advance action window
        act = torch.cat([act[:, 1:], act_future[:, t:t+1]], dim=1)

    return rearrange(emb, "(b s) t d -> b s t d", b=B, s=S)
```

**History truncation** (`history_size=3`): During rollout, the predictor only sees the last `HS` frames rather than the full history. This:
- Keeps computational cost bounded (O(HS) not O(T))
- Prevents the transformer from attending to very old, potentially inaccurate predictions
- Acts as a fixed-size sliding window over the latent state sequence

---

## 3. JEPA for Rendering Acceleration

### 3.1 D-JEPA: Denoising with Joint-Embedding Predictive Architecture

D-JEPA (Chen et al., 2024) bridges JEPA with generative modeling by:
1. Reinterpreting JEPA as **masked image modeling** (masked token prediction)
2. Treating it as **generalized next-token prediction** for auto-regressive generation
3. Adding **diffusion loss** to model per-token probability distribution in continuous space
4. Also supporting **flow matching loss** as alternative

Key result: D-JEPA consistently achieves lower FID scores with fewer training epochs as GFLOPs increase, demonstrating strong scalability. The base, large, and huge models outperform all prior generative models on class-conditional ImageNet.

**Relevance to rendering**: D-JEPA proves JEPA can be adapted for pixel-space generation/denoising, not just latent prediction. The architecture can learn to denoise render outputs by predicting clean latent representations from noisy inputs.

### 3.2 Neural Rendering with Temporal Prediction

The analogy to FLIP simulation:
- **FLIP**: Given fluid state at timestep n, predict state at n+1 using physics
- **JEPA rendering**: Given render state at sample count n, predict converged result at sample count n+1 (or infinity)
- The JEPA predictor learns the "rendering dynamics" — how images evolve as more samples accumulate

**Temporal rendering prediction** scenarios:
1. **Sample prediction**: noisy_render(4spp) → predicted_clean_render (skip to ~256spp equivalent)
2. **Frame prediction**: render(camera_t) → predict render(camera_{t+1}) without re-rendering
3. **Parameter prediction**: render(params_t) → predict render(params_{t+1}) after parameter change

### 3.3 Differentiable Rendering as Training Signal

Differentiable rendering provides **ground-truth gradient signal** for neural networks:
- Forward pass: scene parameters → physical simulation → pixel values
- Backward pass: pixel loss → gradient through rendering equation → parameter gradients
- This is a **physically-grounded training signal** — the gradients encode real light transport

For JEPA training, this means:
- Mitsuba can generate infinite training data (random scenes, lighting, materials)
- The differentiable pipeline provides exact gradients for the rendering process
- JEPA can learn to predict the rendering equation's behavior from data

---

## 4. The Connection: Mitsuba as a JEPA Training "Gym"

### 4.1 The Core Idea

Mitsuba's differentiable rendering can serve as a **perfect simulation environment** for training a JEPA world model of rendering. The parallel:

| RL/Dynamics | Rendering |
|---|---|
| Environment state | Scene parameters (geometry, materials, lights, camera) |
| Action | Parameter perturbation (move light, change color, rotate camera) |
| Observation | Rendered image |
| Transition function | Rendering equation (Mitsuba) |
| Reward/Loss | Image quality metric (L2, LPIPS, SSIM) |

### 4.2 Forward Path: Scene → Render → Image

```
scene_params → mi.render() → image → encoder → latent_z
                                          ↓
action (param change) → action_encoder → act_emb
                                          ↓
                     predictor(z, act_emb) → predicted_z_next
                                          ↓
                     decoder(z_next) → predicted_image_next
```

The JEPA learns to predict what the **next render** will look like after a scene parameter change, entirely in latent space. Mitsuba provides the ground truth: actually render with the new parameters and encode that image.

### 4.3 What Mitsuba's Gradient Signal Teaches JEPA

When you backpropagate through Mitsuba:

1. **Light transport physics**: The gradient encodes how light bounces, reflects, refracts — the full path integral. A JEPA trained on Mitsuba data implicitly learns these physics.

2. **Scene parameter sensitivity**: `dr.grad(params[key])` tells you exactly which parameters matter for which pixels. JEPA learns this sensitivity map as a learned function.

3. **Discontinuity awareness**: Mitsuba's PRB integrator handles gradient computation at geometric discontinuities (silhouettes, shadows). JEPA can learn to predict these edge-aware gradient fields.

4. **Global illumination dynamics**: Changes to one part of the scene affect distant pixels via indirect illumination. JEPA must learn these long-range dependencies.

### 4.4 Can JEPA Learn to Invert the Rendering Equation?

**Partially, yes.** The rendering equation `L = f(geometry, materials, lights, camera)` is extremely high-dimensional, but:

- **Local inversions** are feasible: given an image, predict likely material properties or light positions in a known scene (this is exactly what Mitsuba's inverse rendering does with optimization)
- **JEPA as learned inverse renderer**: Instead of iterative optimization (200 Adam steps), a trained JEPA could predict scene parameters from images in a single forward pass
- **Hybrid approach**: Use JEPA for fast initial estimate, then Mitsuba optimization for refinement

### 4.5 Concrete Integration Architecture

```python
# Omen JEPA-Mitsuba Integration

import mitsuba as mi
import drjit as dr
import torch
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg

mi.set_variant('cuda_ad_rgb')

class MitsubaJEPAEnv:
    """Mitsuba as a gym environment for JEPA training."""

    def __init__(self, scene_path, param_keys):
        self.scene = mi.load_file(scene_path, integrator='prb')
        self.params = mi.traverse(self.scene)
        self.param_keys = param_keys

    def render_observation(self, params=None, spp=4):
        """Render current scene state → image observation."""
        return mi.render(self.scene, params or self.params, spp=spp)

    def step(self, param_updates):
        """Apply parameter changes and render next state."""
        for key, delta in param_updates.items():
            self.params[key] = self.params[key] + delta
        self.params.update()
        return self.render_observation()

    def compute_loss(self, predicted_image, spp_ref=256):
        """L2 loss against ground truth render."""
        ref = self.render_observation(spp=spp_ref)
        return dr.sum(dr.sqr(predicted_image - ref)) / len(ref)


# Training loop
env = MitsubaJEPAEnv('scene.xml', ['light.position', 'material.albedo.value'])
model = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
sigreg = SIGReg(knots=17, num_proj=1024)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    # Sample random action sequence (parameter perturbations)
    actions = sample_param_perturbations(env.param_keys, seq_len=T)

    # Encode initial frames
    info = {'pixels': initial_frames, 'action': actions}
    info = model.encode(info)  # → info["emb"], info["act_emb"]

    # Teacher-forced prediction
    preds = model.predict(info["emb"][:, :-1], info["act_emb"][:, :-1])

    # Ground truth: encode actual next-frame renders
    # (obtained by stepping the Mitsuba environment)
    with torch.no_grad():
        next_frames = mitsuba_rollout(env, actions)
        next_info = model.encode({'pixels': next_frames})

    # Losses
    L_pred = torch.nn.functional.mse_loss(preds, next_info["emb"][:, 1:])
    L_sigreg = sigreg(info["emb"])
    L_total = L_pred + 0.1 * L_sigreg

    opt.zero_grad()
    L_total.backward()
    opt.step()
```

### 4.6 Key Insights for Omen

1. **Mitsuba as data generator**: Infinite procedurally-generated scenes with known ground truth parameters. No need to collect real-world data.

2. **Two-stage training**:
   - **Stage 1 (JEPA pretraining)**: Train predictor on latent rendering dynamics — predict `z_{t+1}` from `(z_t, action)` where action = scene parameter delta
   - **Stage 2 (Fine-tuning)**: Connect decoder and train end-to-end with differentiable rendering loss

3. **Speed advantage**: Once trained, JEPA predictor can estimate render outcomes in ~1ms vs. ~100ms for a low-spp Mitsuba render, enabling real-time preview of parameter changes.

4. **Denoising application**: JEPA can learn the mapping `noisy_render(4spp) → clean_render(256spp)` as a special case where the "action" is increasing sample count. This is directly analogous to D-JEPA's denoising capability.

5. **Gradient distillation**: Mitsuba's exact gradients can be used as supervision signal for training JEPA to predict sensitivity maps — "which parameters should I change to achieve this target image?"

6. **The `dr.wrap()` bridge**: Enables seamless integration — Mitsuba rendering becomes a differentiable PyTorch layer, allowing JEPA training with standard PyTorch tools while getting exact rendering gradients.

---

## Sources

### Mitsuba 3
- [Mitsuba 3 Inverse Rendering Tutorials](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering_tutorials.html)
- [Gradient-based Optimization Tutorial](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/gradient_based_opt.html)
- [Forward & Inverse Rendering Tutorial](https://mitsuba.readthedocs.io/en/v3.5.1/src/inverse_rendering/forward_inverse_rendering.html)
- [PyTorch Interoperability Tutorial](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/pytorch_mitsuba_interoperability.html)
- [DrJit Paper (Jakob et al.)](https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Jakob2022DrJit.pdf)
- [Mitsuba 3 GitHub](https://github.com/mitsuba-renderer/mitsuba3)

### LeWorldModel
- [LeWM Paper (arXiv)](https://arxiv.org/abs/2603.19312)
- [LeWM GitHub](https://github.com/lucas-maes/le-wm)
- [LeWM module.py](https://github.com/lucas-maes/le-wm/blob/main/module.py)
- [LeWM jepa.py](https://github.com/lucas-maes/le-wm/blob/main/jepa.py)

### D-JEPA
- [D-JEPA Paper (arXiv)](https://arxiv.org/html/2410.03755v1)

### JEPA Foundations
- [A Path Towards Autonomous Machine Intelligence (LeCun)](https://openreview.net/pdf?id=BZ5a1r-kVsf)
