# Mitsuba 3 Differentiable Rendering Pipeline - Technical Reference

Complete API reference for building a JEPA training gym with Mitsuba 3 / Dr.Jit.

---

## 1. Differentiable Rendering Loop: mi.render()

### Function Signature

```python
mi.render(
    scene,           # mi.Scene - loaded scene
    params=None,     # mi.SceneParameters - from mi.traverse(scene), MUST pass for AD
    sensor=0,        # int or mi.Sensor - which camera
    integrator=None, # mi.Integrator - e.g. 'prb' for differentiable
    seed=0,          # int - RNG seed for primal render
    seed_grad=0,     # int - RNG seed for gradient pass
    spp=0,           # int - samples per pixel for primal (0 = use scene default)
    spp_grad=0       # int - samples per pixel for gradient pass
) -> drjit.llvm.ad.TensorXf  # Returns differentiable tensor (H, W, C)
```

### CRITICAL: params argument

You MUST pass params to mi.render() for differentiable rendering. Without it, the computation graph is NOT built and gradients will NOT flow:

```python
params = mi.traverse(scene)
# ... make params differentiable ...
image = mi.render(scene, params, spp=128)  # differentiable
image_no_ad = mi.render(scene, spp=128)    # NOT differentiable
```

### Computational Graph

When params is passed with AD-enabled variables:
1. The integrator traces the full path tracing computation
2. Every ray-sample-bsdf bounce is recorded in Dr.Jit's AD graph
3. mi.render() returns a TensorXf that is part of this graph
4. Calling dr.backward(loss) or dr.forward(param) traverses the graph

**Specialized integrators** (more memory-efficient than naive AD):
- `prb` - Path Replay Backpropagation: replays paths in adjoint pass, constant memory, O(n) time
- `rb` - Radiative Backpropagation: propagates radiance back along paths
- Default `path` with naive AD: stores full computation graph, OOM risk at high spp

```python
# Use 'prb' for differentiable rendering (recommended)
scene = mi.load_file('scene.xml', integrator='prb')
```

### Rendering at Different SPP in Same Loop

Yes, you can reuse the scene object. Just change spp:

```python
scene = mi.load_file('scene.xml', integrator='prb')
params = mi.traverse(scene)

# Low spp for gradient computation (fast, noisy)
image_low = mi.render(scene, params, spp=1)

# High spp for reference/target (slow, clean)
image_high = mi.render(scene, params, spp=256)

# Different spp in same optimization loop
for it in range(100):
    # Gradient step at low spp
    image = mi.render(scene, params, spp=4, seed=it)
    loss = dr.mean(dr.square(image - target))
    dr.backward(loss)
    opt.step()
    params.update(opt)

    # Validation at high spp (no gradient needed)
    if it % 10 == 0:
        val_image = mi.render(scene, spp=256, seed=9999)
```

**Note**: Use different seed values to decorrelate iterations. seed_grad controls the gradient pass seed independently.

---

## 2. Dr.Jit Autodiff API

### AD-Enabled Array Types

```python
# MUST use AD-enabled types (note the ".ad" suffix)
from drjit.cuda.ad import Float, Array3f, UInt, TensorXf
# OR for LLVM/CPU:
from drjit.llvm.ad import Float, Array3f, UInt, TensorXf
# OR the auto backend:
from drjit.auto.ad import Float, Array3f, UInt

# WRONG - these lack gradient tracking:
from drjit.auto import Float  # NO ".ad" suffix = no AD
```

### dr.enable_grad() - Mark Parameters as Differentiable

```python
import drjit as dr

x = Float(10)
dr.enable_grad(x)  # x is now tracked in the AD graph

# Works on PyTrees (dicts, lists, nested structures) recursively
params = mi.traverse(scene)
dr.enable_grad(params)  # enables grad on ALL array values in params

# Enable on specific parameter
dr.enable_grad(params['red.reflectance.value'])
```

### dr.backward() vs dr.forward() - When to Use Which

**Reverse mode (dr.backward)**: Use when you have FEW outputs and MANY inputs.
- Standard for optimization (one loss scalar -> many parameters)
- Computes d(loss)/d(all_params) in one pass

**Forward mode (dr.forward)**: Use when you have FEW inputs and MANY outputs.
- Useful for sensitivity analysis (one param -> full image gradient)
- Computes d(image)/d(param) for one param at a time

```python
# REVERSE MODE (typical optimization) - start from loss, propagate to params
image = mi.render(scene, params, spp=128)
loss = dr.mean(dr.square(image - target))
dr.backward(loss)
# Now params['red.reflectance.value'].grad contains d(loss)/d(param)

# FORWARD MODE - start from parameter, propagate to image
dr.enable_grad(params['red.reflectance.value'])
image = mi.render(scene, params, spp=128)
dr.forward(params['red.reflectance.value'])
# Now image.grad contains d(image)/d(param)
```

### Full AD Traversal API

```python
# Four directional traversals:
dr.forward_from(x)    # forward FROM x to everything downstream
dr.forward_to(y)      # forward TO y from everything with .grad set
dr.backward_from(y)   # reverse FROM y to everything upstream
dr.backward_to(x)     # reverse TO x (compute only x's gradient)

# Shorthands:
dr.forward(x)   == dr.forward_from(x)   # propagate from x
dr.backward(y)  == dr.backward_from(y)   # propagate from y
```

### dr.schedule() - Deferred Evaluation

```python
# dr.schedule() marks variables for inclusion in the NEXT kernel launch
# It does NOT evaluate immediately
dr.schedule(a, b)  # a and b will be computed in the next kernel

# dr.eval() triggers immediate compilation and execution
dr.eval(a, b)  # compiles and runs a kernel computing a and b together

# WHY USE schedule: avoids redundant kernel launches
# BAD: two separate kernels
print(a)  # kernel 1: computes a
print(b)  # kernel 2: computes b (re-does shared computation)

# GOOD: one merged kernel
dr.schedule(a, b)
dr.eval()  # single kernel computing both a and b
```

### Gradient Accumulation and Clearing

```python
# Read gradient
g = dr.grad(x)        # returns gradient tensor (zero if not computed)
g = x.grad            # shorthand

# Set gradient (for forward mode seeding)
dr.set_grad(x, Float(1.0))
x.grad = Float(1.0)   # shorthand

# Accumulate gradient (add to existing)
dr.accum_grad(x, new_grad)

# Clear gradient (IMPORTANT in loops)
dr.clear_grad(x)
dr.clear_grad(params)  # works on PyTrees

# Replace gradient (advanced - substitute AD-computed gradient)
y_custom = dr.replace_grad(y_ad, y_manual)

# Detach from AD graph (stop gradient flow)
x_detached = dr.detach(x)       # returns copy without AD tracking
x_detached = dr.detach(x, True) # preserve the value, just detach graph
```

---

## 3. Optimizers: mi.ad.Adam and drjit.opt.Adam

### Mitsuba mi.ad.Adam (for scene parameters)

```python
opt = mi.ad.Adam(lr=0.05)

# Attach a scene parameter - optimizer takes ownership
key = 'red.reflectance.value'
opt[key] = params[key]  # copies into optimizer, enables grad tracking

# Push optimizer values back to scene params
params.update(opt)  # MUST call this before rendering!

# Full optimization loop
for it in range(100):
    # 1. Render (must pass params!)
    image = mi.render(scene, params, spp=4, seed=it)

    # 2. Compute loss
    loss = dr.mean(dr.square(image - target))

    # 3. Backpropagate
    dr.backward(loss)

    # 4. Update optimizer state
    opt.step()

    # 5. Push updated values back to scene
    params.update(opt)
```

### Dr.Jit drjit.opt.Adam (for arbitrary tensors / neural networks)

```python
from drjit.opt import Adam

# Can optimize ANY Dr.Jit tensor, not just scene parameters
opt = Adam(lr=1e-3)

# Attach arbitrary tensors
my_tensor = dr.llvm.ad.TensorXf(dr.full(dr.llvm.ad.Float, 0.5, 1000))
opt['my_param'] = my_tensor

# Attach a drjit.nn.Module
net = MyDrJitNet()
opt.update(net)  # pulls ALL parameters from net into optimizer

# Training loop
for i in range(n_iter):
    net.update(opt)   # push optimizer state into network
    y = net(x)        # forward pass (differentiable)
    loss = ...
    dr.backward(loss)
    opt.step()
```

### Multiple Optimizers on One Module

```python
from drjit.opt import AdamW, Muon

# Muon for 2D weights, AdamW for biases/scalars
muon = Muon(lr=0.02)
muon.update(net)

adamw = AdamW(lr=1e-3)
adamw.update({k: net[k] for k in net if len(net[k].shape) == 1})

for i in range(n_iter):
    net.update(muon)
    net.update(adamw)
    y = net(x_tensor)
    loss = ...
    dr.backward(loss)
    muon.step()
    adamw.step()
```

---

## 4. Extracting Rendered Image as Tensor

### mi.render() Returns TensorXf Directly

```python
# mi.render() already returns a drjit TensorXf with shape (H, W, C)
image = mi.render(scene, params, spp=128)
# image is drjit.llvm.ad.TensorXf, shape e.g. (256, 256, 3)

# Convert to different formats:
image_np = image.numpy()          # NumPy array (evaluates if not already)
image_torch = image.torch()       # PyTorch tensor (zero-copy via DLPack)
image_jax = image.jax()           # JAX array
image_tf = image.tf()             # TensorFlow tensor

# Access the underlying array directly
image_array = image.array  # drjit Float (flattened)

# Convert to mi.Bitmap for saving
bitmap = mi.util.convert_to_bitmap(image)
bitmap.write('output.exr')
```

### Zero-Copy Conversion Between Frameworks

```python
import numpy as np
import torch

# Dr.Jit -> NumPy (zero-copy when possible via DLPack)
arr = image.numpy()

# Dr.Jit -> PyTorch (zero-copy)
tensor = image.torch()

# NumPy -> Dr.Jit
from drjit.llvm.ad import TensorXf
np_arr = np.random.randn(256, 256, 3).astype(np.float32)
dj_tensor = TensorXf(np_arr)

# PyTorch -> Dr.Jit (use .torch() reverse or constructor)
torch_tensor = torch.randn(256, 256, 3)
dj_tensor = TensorXf(torch_tensor)
```

### Differentiable Interop (gradients flow between frameworks)

```python
# For differentiable interop, use dr.wrap() (see section 6)
# Simple constructor conversion does NOT preserve gradients
from drjit.llvm.ad import Float
a_torch = torch.tensor([1.0], requires_grad=True)
b = Float(a_torch)  # NOT differentiable w.r.t. a_torch
```

---

## 5. Differentiable Scene Parameters

### Making Material Parameters Differentiable

```python
# Load scene with AD variant
mi.set_variant('llvm_ad_rgb')  # or 'cuda_ad_rgb'
scene = mi.load_file('scene.xml', integrator='prb')

# Get all parameters
params = mi.traverse(scene)

# Explore available parameters
print(params)  # dict-like: 'red.reflectance.value', 'light.emitter.radiance.value', etc.

# Make specific parameter differentiable
key = 'red.reflectance.value'
dr.enable_grad(params[key])

# Or enable on ALL parameters
dr.enable_grad(params)
```

### Common Differentiable Parameters

```python
# Material albedo/color
params['mesh.bsdf.reflectance.value']

# Light intensity
params['light.emitter.radiance.value']

# Vertex positions (for shape optimization)
params['mesh.vertex_positions']

# UV coordinates
params['mesh.vertex_texcoords']

# Volume density (for volumetric)
params['volume.density']

# Emitter transform
params['light.to_world']
```

### Updating Parameters and Scene State

```python
# After changing ANY parameter, you MUST call params.update()
params['red.reflectance.value'] = mi.Color3f(0.5, 0.1, 0.1)
params.update()  # notifies scene objects to rebuild internal state

# With optimizer (preferred in optimization loops)
params.update(opt)  # copies from optimizer + notifies scene
```

---

## 6. Gradient Flow FROM Neural Network THROUGH Renderer

### Pattern 1: PyTorch NN -> Mitsuba Renderer (via dr.wrap)

This is the key pattern for a JEPA gym: neural network outputs become scene parameters.

```python
import torch
import torch.nn as nn
import drjit as dr
import mitsuba as mi

# PyTorch network
net = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
    nn.Sigmoid()
)

# Dr.wrap makes a PyTorch-compatible function that runs Mitsuba inside
@dr.wrap(source='torch', target='drjit')
def render_with_params(texture, spp=256, seed=1):
    """PyTorch tensor in -> Dr.Jit rendering -> PyTorch tensor out.
    Gradients flow through automatically."""
    # Convert torch tensor to Dr.Jit and set as scene parameter
    params['mesh.bsdf.reflectance.value'] = mi.Color3f(texture)
    params.update()
    image = mi.render(scene, params, spp=spp, seed=seed)
    return image.torch()  # back to PyTorch

# Training loop - gradients flow: loss -> image -> texture -> net weights
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(100):
    latent = torch.randn(1, 128)
    texture = net(latent)             # PyTorch forward
    image = render_with_params(texture, spp=4)  # through renderer
    loss = ((image - target_image) ** 2).mean()  # PyTorch loss

    optimizer.zero_grad()
    loss.backward()                   # PyTorch autograd calls dr.backward() internally
    optimizer.step()
```

### Pattern 2: Pure Dr.Jit (no PyTorch, fully differentiable)

```python
from drjit.llvm.ad import Float, TensorXf
from drjit.opt import Adam
import drjit.nn as nn

# Dr.Jit native neural network (fused kernels, no framework boundary)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 3)

    def __call__(self, x):
        x = dr.relu(self.fc1(x))
        x = dr.sigmoid(self.fc2(x))
        return x

net = MLP()
opt_net = Adam(lr=1e-3)
opt_net.update(net)  # attach all net params to optimizer

# Scene optimizer
opt_scene = mi.ad.Adam(lr=0.05)
key = 'red.reflectance.value'
opt_scene[key] = params[key]

for it in range(100):
    # Push optimizer state to network and scene
    net.update(opt_net)
    params.update(opt_scene)

    # NN forward -> scene param -> render -> loss, all in one AD graph
    latent = dr.randn(dr.llvm.ad.Float, 128)
    texture = net(latent)
    params[key] = mi.Color3f(texture)
    params.update()

    image = mi.render(scene, params, spp=4, seed=it)
    loss = dr.mean(dr.square(image - target))
    dr.backward(loss)

    opt_net.step()
    opt_scene.step()
```

### Pattern 3: dr.wrap in Reverse (Dr.Jit -> PyTorch -> Dr.Jit)

```python
# Wrap a PyTorch computation to be called from Dr.Jit pipeline
@dr.wrap(source='drjit', target='torch')
def torch_feature_extractor(image_tensor):
    """Dr.Jit tensor in -> PyTorch processing -> Dr.Jit tensor out."""
    with torch.no_grad():
        features = pretrained_net(image_tensor)
    return features  # gradients still flow if needed
```

---

## 7. Multiple Renders in a Loop: Memory Management

### The Core Issue

Each mi.render() call with AD builds a computation graph. In a loop, this accumulates memory. You must manage this explicitly.

### Complete Memory-Safe Training Loop

```python
import drjit as dr
import mitsuba as mi

scene = mi.load_file('scene.xml', integrator='prb')
params = mi.traverse(scene)
target = mi.render(scene, spp=256)  # reference image (no AD needed)

opt = mi.ad.Adam(lr=0.05)
key = 'red.reflectance.value'
opt[key] = params[key]
params.update(opt)

for it in range(1000):
    # 1. Render with AD
    image = mi.render(scene, params, spp=4, seed=it)

    # 2. Compute loss
    loss = dr.mean(dr.square(image - target))

    # 3. Backpropagate (this traverses and frees the AD graph)
    dr.backward(loss)

    # 4. Optimizer step
    opt.step()

    # 5. Push back to scene
    params.update(opt)

    # 6. Clear any remaining gradients (prevent accumulation)
    dr.clear_grad(params)

    # Optional: force evaluation to free memory
    if it % 50 == 0:
        loss_val = float(loss.array[0])
        print(f"Iter {it}: loss = {loss_val:.6f}")
```

### Key Memory Management Rules

1. **Always call dr.backward() or dr.forward()** - this processes and releases the AD graph
2. **Call dr.clear_grad() between iterations** - prevents gradient accumulation
3. **Use prb integrator** - Path Replay Backpropagation uses constant memory (replays paths instead of storing them)
4. **Avoid dr.detach() on intermediate values you need gradients for**
5. **Use dr.freeze() decorator** for render functions in tight loops to skip re-tracing:

```python
@dr.freeze
def render_frozen(scene, params, spp, seed):
    return mi.render(scene, params, spp=spp, seed=seed)

for it in range(10000):
    image = render_frozen(scene, params, spp=4, seed=it)
    # First call: traces + compiles + caches
    # Subsequent calls: reuses cached kernel (skips tracing overhead)
    ...
```

### Using dr.freeze() for Performance

```python
import drjit as dr

@dr.freeze
def render_step(scene, params, spp, seed):
    """Cached render function - kernel is compiled once, reused thereafter."""
    return mi.render(scene, params, spp=spp, seed=seed)

# In the training loop, the tracing overhead is eliminated after first call
for it in range(10000):
    image = render_step(scene, params, spp=4, seed=it)
    ...
```

**Caveat**: dr.freeze requires that the computation graph structure stays the same across calls. Changing spp is fine (it is a kernel parameter), but changing the scene topology is not.

---

## 8. JEPA Gym Architecture Pattern

### Putting It All Together

```python
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')  # GPU-accelerated AD variant

import drjit as dr
from drjit.cuda.ad import Float, TensorXf
import torch
import torch.nn as nn

# === SCENE SETUP ===
scene = mi.load_file('scene.xml', integrator='prb')
params = mi.traverse(scene)

# === JEPA ENCODER (PyTorch) ===
encoder = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64 * 256 * 256, 512),
)

# === JEPA PREDICTOR (PyTorch) ===
predictor = nn.Sequential(
    nn.Linear(512 + 128, 256),  # latent + action
    nn.ReLU(),
    nn.Linear(256, 512),
)

# === RENDER WRAPPER ===
@dr.wrap(source='torch', target='drjit')
def render_scene(light_intensity, spp=4, seed=1):
    """Differentiable bridge: PyTorch -> Mitsuba -> PyTorch."""
    params['light1.emitter.radiance.value'] = mi.Color3f(light_intensity)
    params.update()
    image = mi.render(scene, params, spp=spp, seed=seed)
    return image.torch()

# === TRAINING LOOP ===
opt = torch.optim.Adam(
    list(encoder.parameters()) + list(predictor.parameters()),
    lr=1e-4
)

for epoch in range(num_epochs):
    # Sample light parameters (action/latent)
    action = torch.randn(batch_size, 128)

    # Render current scene state
    light_params = torch.sigmoid(action)
    rendered = render_scene(light_params, spp=1)  # low spp for speed

    # Encode rendered image
    latent = encoder(rendered.unsqueeze(0))

    # Predict next latent
    predicted_latent = predictor(torch.cat([latent, action], dim=-1))

    # Render next state and encode (target)
    with torch.no_grad():
        next_rendered = render_scene(next_light_params, spp=1)
        target_latent = encoder(next_rendered.unsqueeze(0))

    # JEPA loss (latent space, not pixel space)
    loss = ((predicted_latent - target_latent) ** 2).mean()

    opt.zero_grad()
    loss.backward()  # flows through: loss -> predictor -> latent -> encoder -> rendered -> render_scene -> action
    opt.step()
```

---

## 9. Quick Reference: Dr.Jit AD API Summary

| Function | Purpose |
|----------|---------|
| dr.enable_grad(x) | Mark variable for gradient tracking |
| dr.disable_grad(x) | Disable gradient tracking |
| dr.grad_enabled(x) | Check if grad tracking is on |
| dr.grad(x) | Read gradient value |
| dr.set_grad(x, g) | Set gradient value |
| dr.accum_grad(x, g) | Add to existing gradient |
| dr.clear_grad(x) | Zero out gradient |
| dr.detach(x) | Return copy without AD graph |
| dr.replace_grad(y, y_ad) | Swap primal value with AD-tracked version |
| dr.forward(x) | Forward-mode AD from x |
| dr.backward(y) | Reverse-mode AD from y |
| dr.forward_from(x) | Same as dr.forward(x) |
| dr.forward_to(y) | Forward to y from inputs with .grad set |
| dr.backward_from(y) | Same as dr.backward(y) |
| dr.backward_to(x) | Reverse to x only |
| dr.schedule(*args) | Defer evaluation to next kernel |
| dr.eval(*args) | Force immediate kernel compilation |
| dr.freeze | Decorator to cache traced kernels |
| dr.wrap(source, target) | Bridge PyTorch/JAX/TF <-> Dr.Jit AD |
