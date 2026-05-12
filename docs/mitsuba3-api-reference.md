# Mitsuba 3 API Reference - Key APIs for Omen Scene Extractor

## Rendering

### mi.render()
```python
# Top-level render function
image = mi.render(scene, sensor=0, seed=0, spp=0, develop=True, evaluate=True)
# Returns: drjit.llvm.ad.TensorXf (shape: [H, W, C])
# spp=0 uses sensor's sampler default
# Different sensors for multi-view: sensor=0, sensor=1, etc.
```

### Integrator.render()
```python
# Low-level: integrator.render(scene, sensor, seed, spp)
# Returns: TensorXf
# AD variants also have: render_forward(scene, params, sensor), render_backward(scene, params, grad_in, sensor)
```

## Scene Access (mi.Scene)

```python
scene.shapes()     # -> list of Shape/ShapePtr
scene.emitters()   # -> list of Emitter/EmitterPtr
scene.sensors()    # -> list of Sensor/SensorPtr
scene.bbox()       # -> ScalarBoundingBox3f
scene.integrator() # -> Integrator
```

## Geometry Extraction (mi.Mesh / ShapePtr)

```python
shape = scene.shapes()[0]

# Vertex data
shape.vertex_count()           # -> int
shape.vertex_normal(index)     # -> Vector3f
# vertex_positions accessed via traverse():
params = mi.traverse(shape)
vertices = params['vertex_positions']  # -> Float array [N*3]

# Face data
shape.face_count()             # -> int
shape.face_indices(index)      # -> Vector3u (triangle vertex indices)
shape.faces_buffer()           # -> UInt (raw buffer)

# Properties
shape.has_vertex_normals()     # -> Bool
shape.has_vertex_texcoords()   # -> Bool
shape.has_face_normals()       # -> Bool
shape.has_mesh_attributes()    # -> Bool
shape.surface_area()           # -> Float
shape.shape_type()             # -> ShapeType enum

# Material association
shape.bsdf()                   # -> BSDFPtr
shape.emitter()                # -> EmitterPtr (area lights)
shape.sensor()                 # -> SensorPtr (if area sensor)
```

## BSDF / Material (mi.BSDFPtr)

```python
bsdf = shape.bsdf()

# Access BSDF parameters via traverse():
params = mi.traverse(bsdf)
# Common parameters (depends on BSDF type):
# 'diffuse_reflectance'  -> Color3f / Texture
# 'specular_reflectance' -> Color3f / Texture
# 'roughness'            -> Float
# 'alpha'                -> Float (GGX)
# 'eta'                  -> Float (IOR for dielectric)
# 'metallic'             -> Float (conductor)
```

## Emitter / Light (mi.EmitterPtr)

```python
for emitter in scene.emitters():
    # Access emitter parameters via traverse():
    params = mi.traverse(emitter)
    # Point light: 'position' -> Point3f, 'intensity' -> Color3f
    # Area light: 'radiance' -> Color3f, associated with a Shape
    # Environment: 'radiance' -> Texture

    # Emitter type check
    emitter.is_environment()  # -> Bool
```

## Camera / Sensor (mi.Sensor / ProjectiveCamera)

```python
sensor = scene.sensors()[0]

# Camera properties (ProjectiveCamera)
sensor.fov()                    # -> float (field of view)
sensor.near_clip()              # -> float
sensor.far_clip()               # -> float
sensor.projection_transform()   # -> ProjectiveTransform4f

# Transform
sensor.to_world()               # -> Transform4f (camera-to-world)
# Use mi.traverse() for world_transform:
params = mi.traverse(sensor)
transform = params['to_world']  # Transform4f matrix

# Film
film = sensor.film()
film.size()                     # -> ScalarVector2i (width, height)
film.crop_size()                # -> ScalarVector2i
film.crop_offset()              # -> ScalarPoint2i
film.base_channels_count()      # -> int (typically 3 RGB + 1 alpha = 4)
film.bitmap(raw=False)          # -> Bitmap (developed image)

# Sampler
sampler = sensor.sampler()
sampler.sample_count()          # -> int (default spp)
```

## Scene Parameters (mi.traverse / mi.SceneParameters)

```python
# Get all scene parameters
params = mi.traverse(scene)
# params is dict-like: params['key'] = value
# Common keys:
#   'shape.vertex_positions'  (per-shape)
#   'shape.faces'            (per-shape)
#   'bsdf.diffuse_reflectance' (per-BSDF)
#   'emitter.position'       (per-emitter)
#   'sensor.to_world'        (per-sensor)

# Update scene after parameter changes
params['key'] = new_value
params.update()  # applies changes to scene

# For differentiable rendering:
dr.enable_grad(params['key'])   # enable gradient tracking
dr.set_grad(params['key'], grad_value)  # set gradient for forward mode
```

## Differentiable Rendering (AD variants: cuda_ad_rgb, etc.)

```python
import drjit as dr

# Render with AD tracking
image = mi.render(scene, params, spp=1)
# params must have dr.enable_grad() called on them first

# Reverse mode (backward)
loss = dr.mean(dr.square(image - gt))
dr.backward(loss)
grad = dr.grad(params['key'])  # read gradients

# Forward mode
dr.set_grad(params['key'], seed_gradient)
dr.forward(loss)
# produces gradient image showing how scene parameter changes affect render

# Optimizer
from drjit.opt import Adam
opt = Adam(lr=0.05)
opt[param_key] = initial_value
opt.update()  # Adam step
```

## OptixDenoiser (baseline comparison for Omen)

```python
denoiser = mi.OptixDenoiser(
    input_size=mi.ScalarVector2u(width, height),
    albedo=True,    # use albedo channel
    normals=True,   # use normals channel
    temporal=False, # temporal denoising
    denoise_alpha=False
)

# Apply denoiser to Bitmap
result = denoiser(
    noisy_bitmap,
    albedo_ch='albedo',     # channel name in multi-channel bitmap
    normals_ch='normal',    # channel name
    flow_ch='',             # optical flow for temporal
    previous_denoised_ch='', # previous frame for temporal
    noisy_ch='<root>'       # main noisy channel
)
# Returns: Bitmap (denoised)
```

## Bitmap I/O

```python
# Create bitmap
bmp = mi.Bitmap(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.Float32,
                mi.ScalarVector2u(width, height))

# From render result (TensorXf -> Bitmap)
bmp = mi.Bitmap(image)

# Read/Write
bmp.write('output.exr')  # OpenEXR
bmp.write('output.png')  # PNG

# Convert to numpy
np_array = np.array(bmp)  # shape: (H, W, C)
```

## Scene Creation

```python
# Load from dict
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {'type': 'path'},
    'sensor': {...},
    'shape': {...},
    'emitter': {...},
})

# Cornell box (built-in)
scene = mi.cornell_box()
```

## Variant Detection

```python
mi.set_variant('cuda_ad_rgb')  # NVIDIA GPU + AD
mi.set_variant('llvm_ad_rgb')  # CPU/ROCm + AD
mi.set_variant('scalar_rgb')   # CPU, no AD

variant = mi.variant()  # returns current variant string
```
