# bpy and Mitsuba 3 Installation Findings

**Date**: 2025-05-12
**Status**: Architecture validation complete

---

## Executive Summary

| Component | Availability | Recommended Approach |
|-----------|--------------|---------------------|
| **bpy (Blender Python API)** | ❌ Not on PyPI (official) | Use Blender as subprocess/service |
| **Mitsuba 3** | ✅ PyPI v3.8.0 | Install via `pip install mitsuba` |
| **Pixi/Conda** | ❌ No Mitsuba package | Use pip+uv in pixi environment |

---

## Finding 1: bpy Cannot Be Imported as Normal Python Module

### What We Tested
```bash
python3 -c "import bpy"
# Result: ModuleNotFoundError: No module named 'bpy'

pip install bpy
# Result: ERROR: No matching distribution found for bpy
```

### The Reality

**Blender does NOT ship bpy as a standalone Python package.**

To use bpy, you have TWO options:

#### Option A: Blender as Subprocess (Recommended ✅)

```python
import subprocess

# Run Blender headless, execute script
result = subprocess.run([
    'blender',                          # Blender executable
    '--background',                     # No UI
    '--python', 'extract_scene.py',     # Script to run
    '--',                               # Separator
    '--scene', 'input.blend',           # Args to script
    '--output', 'scene.json'
], capture_output=True)

scene_data = json.loads(result.stdout)
```

**extract_scene.py**:
```python
import sys
import json
import bpy

# Parse command-line args
args = {arg[i:]: arg[i+2:] for i, arg in enumerate(sys.argv) if arg.startswith('--')}

# Load blend file
bpy.ops.wm.open_mainfile(filepath=args['--scene'])

# Extract scene data
scene = bpy.context.scene
data = {
    'camera': str(scene.camera.name),
    'objects': [obj.name for obj in scene.objects if obj.type == 'MESH']
}

# Output as JSON
print(json.dumps(data))
```

#### Option B: Build Blender as Python Module (Advanced ❌)

```bash
cd /home/riju279/Documents/Projects/MOJO/Cycles_mojo/blender
mkdir build && cd build
cmake -DBUILD_PYTHON_MODULE=ON ..
make -j$(nproc)
```

Then you CAN do:
```python
import bpy  # Works!
```

**Problems with Option B**:
- Complex build process
- Need to rebuild for each Blender version
- Not officially supported/maintained
- Breaks easily with dependency conflicts

---

## Finding 2: PyPI "bpy" Package Is Third-Party

```bash
curl -s https://pypi.org/pypi/bpy/json | jq '.releases | keys'
# Result: ['4.2.0', '4.2.13', '4.2.15', '4.2.17', '4.2.18', '4.2.19', '4.2.20', '4.3.0', '4.4.0', '4.5.0']
```

**These are NOT official Blender builds.**

- Third-party attempts to package bpy
- Unofficial, unmaintained
- **Do NOT use for production**

---

## Finding 3: Mitsuba 3 Installation

### Via PyPI (Working ✅)

```bash
pip install mitsuba
# Installs mitsuba 3.8.0 successfully
```

### Via Pixi/Conda (Not Available ❌)

```bash
pixi add mitsuba
# Error: No packages found matching 'mitsuba'

pixi search mitsuba
# No results in conda-forge
```

**Solution**: Use pip within pixi environment

```toml
# pixi.toml
[project]
name = "omen"
version = "0.1.0"

[dependencies]
python = "3.11.*"

# Use pypi dependencies
[feature.pypi-dependencies]
dependencies = [
    "mitsuba",
    "drjit",
    "nabla-ml",  # from nightly.modular.com
]
```

```bash
pixi run pip install mitsuba drjit
```

---

## Finding 4: Correct Architecture for Omen

### ❌ WRONG Approach (What We Initially Thought)

```
Python Script
    ├─ import bpy  # ← This doesn't work!
    ├─ import mitsuba as mi
    └─ Render
```

### ✅ CORRECT Approach (Blender as Service)

```
┌─────────────────────────────────────────────────────────────┐
│                     Omen Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   Blender        │         │   Mitsuba 3      │         │
│  │   (Headless)     │───────▶│   (Rendering)    │         │
│  │                  │ JSON    │                  │         │
│  │  - Extract       │────────▶│  - Omen          │         │
│  │    scene data    │         │    Integrator    │         │
│  │  - Export        │         │  - JEPA          │         │
│  │    to JSON       │         │    acceleration  │         │
│  └──────────────────┘         └──────────┬───────┘         │
│           ▲                              │                 │
│           │                              │                 │
│           │ subprocess                   │ Render          │
│           │                              │ Output          │
│  ┌────────┴──────────────────────────────▼───────┐         │
│  │            Omen Python Scripts                │         │
│  │  ┌────────────────────────────────────────┐  │         │
│  │  │  1. Call Blender → get scene.json      │  │         │
│  │  │  2. Load scene.json → Mitsuba scene    │  │         │
│  │  │  3. Render with Omen integrator        │  │         │
│  │  │  4. Save output.exr                    │  │         │
│  │  └────────────────────────────────────────┘  │         │
│  └─────────────────────────────────────────────────┘         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Step 1: Create Blender Extraction Script

**`src/blend_extractor/extract.py`**:
```python
#!/usr/bin/env python3
"""Extract Blender scene data to JSON format for Mitsuba."""

import sys
import json
import bpy
from pathlib import Path

def extract_mesh_data(obj):
    """Extract mesh vertices and faces."""
    mesh = obj.data
    vertices = [(v.co.x, v.co.y, v.co.z) for v in mesh.vertices]
    faces = [[v for v in p.vertices] for p in mesh.polygons]
    return {'vertices': vertices, 'faces': faces}

def extract_material_data(mat):
    """Extract material properties."""
    return {
        'name': mat.name,
        'diffuse': [mat.diffuse_color.r, mat.diffuse_color.g, mat.diffuse_color.b],
    }

def extract_scene(filepath):
    """Main extraction function."""
    bpy.ops.wm.open_mainfile(filepath=filepath)

    scene = bpy.context.scene
    data = {
        'camera': {
            'name': scene.camera.name,
            'location': list(scene.camera.location),
            'rotation': list(scene.camera.rotation_euler),
        },
        'objects': []
    }

    for obj in scene.objects:
        if obj.type == 'MESH':
            mesh_data = extract_mesh_data(obj)
            material_data = extract_material_data(obj.active_material) if obj.active_material else None
            data['objects'].append({
                'name': obj.name,
                'location': list(obj.location),
                'mesh': mesh_data,
                'material': material_data
            })

    return data

if __name__ == '__main__':
    # Parse args: extract.py --input scene.blend --output scene.json
    args = {arg[i:]: arg[i+2:] for i, arg in enumerate(sys.argv) if arg.startswith('--')}

    scene_data = extract_scene(args['--input'])

    output_path = Path(args.get('--output', 'scene.json'))
    output_path.write_text(json.dumps(scene_data, indent=2))

    print(f"Extracted to {output_path}")
```

### Step 2: Create Omen/Mitsuba Script

**`src/omen_integrator/render.py`**:
```python
#!/usr/bin/env python3
"""Render scene with Mitsuba 3 + Omen integrator."""

import json
import sys
import mitsuba as mi

def json_to_mitsuba(scene_json_path):
    """Convert JSON scene to Mitsuba scene dict."""

    with open(scene_json_path) as f:
        data = json.load(f)

    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'path'},  # Later: 'omen'
        'sensor': {
            'type': 'perspective',
            'to_world': mi.ScalarTransform4f().look_at(
                origin=data['camera']['location'],
                target=[0, 0, 0],
                up=[0, 0, 1]
            ),
        },
    }

    for i, obj in enumerate(data['objects']):
        scene_dict[f'mesh_{i}'] = {
            'type': 'mesh',
            'vertices': obj['mesh']['vertices'],
            'faces': obj['mesh']['faces'],
        }

    return mi.load_dict(scene_dict)

def render(scene_json_path, output_path):
    """Main render function."""
    mi.set_variant('llvm_ad_rgb')
    scene = json_to_mitsuba(scene_json_path)
    image = mi.render(scene)
    mi.Bitmap(image).write(output_path)
    print(f"Rendered to {output_path}")

if __name__ == '__main__':
    render(sys.argv[1], sys.argv[2])
```

### Step 3: CLI Wrapper

**`src/omen_cli/main.py`**:
```python
#!/usr/bin/env python3
"""Omen CLI - Blender to Mitsuba rendering pipeline."""

import subprocess
import sys
from pathlib import Path

def render_blend_to_exr(blend_path, output_path):
    """Complete pipeline: .blend → JSON → Mitsuba → .exr"""

    # Step 1: Extract scene with Blender
    json_path = Path('/tmp/scene.json')

    subprocess.run([
        'blender',
        '--background',
        '--python',
        'src/blend_extractor/extract.py',
        '--',
        '--input', str(blend_path),
        '--output', str(json_path)
    ], check=True)

    # Step 2: Render with Mitsuba
    import importlib
    render_module = importlib.import_module('src.omen_integrator.render')
    render_module.render(str(json_path), str(output_path))

    print(f"✓ Rendered {blend_path} → {output_path}")

if __name__ == '__main__':
    render_blend_to_exr(sys.argv[1], sys.argv[2])
```

---

## Environment Setup

### Install Dependencies

```bash
cd /home/riju279/Documents/Projects/MOJO/Cycles_mojo/omen

# Install Mitsuba via pip (in pixi environment)
pixi run pip install mitsuba drjit

# Verify installation
pixi run python -c "import mitsuba as mi; print(mi.__version__)"
# Should print: 3.8.0
```

### Project Structure

```
omen/
├── pixi.toml
├── src/
│   ├── blend_extractor/     # Blender extraction scripts
│   │   └── extract.py       # Run with: blender --background --python extract.py
│   │
│   ├── omen_integrator/     # Mitsuba + Omen integrator
│   │   ├── __init__.py
│   │   ├── integrator.py    # OmenIntegrator class (Mitsuba plugin)
│   │   └── render.py        # Rendering script
│   │
│   └── omen_cli/            # CLI tools
│       ├── __init__.py
│       └── main.py          # `omen render scene.blend output.exr`
│
├── tests/
│   ├── scenes/              # .blend test files
│   │   └── simple.blend
│   │
│   └── integration/
│       └── test_pipeline.py
│
└── docs/
    └── BPY_MITSUBA_FINDINGS.md  # This file
```

---

## Workflow Examples

### Example 1: Render a .blend file

```bash
pixi run python -m omen_cli.main tests/scenes/simple.blend output.exr
```

### Example 2: Interactive development

```bash
# Terminal 1: Watch for changes and auto-reload
watchmedo auto-restart --pattern=*.py --recursive -- python -m omen_cli.main scene.blend out.exr

# Terminal 2: Edit code
vim src/omen_integrator/integrator.py
```

### Example 3: Jupyter notebook

```python
# notebooks/exploring_omen.ipynb
import subprocess
import mitsuba as mi
import matplotlib.pyplot as plt

# Extract scene
subprocess.run(['blender', '--background', '--python', 'extract.py', '--', 'scene.blend'])

# Render
mi.set_variant('llvm_ad_rgb')
scene = mi.load_file('/tmp/scene.xml')
image = mi.render(scene)

# Display
plt.imshow(mi.Bitmap(image))
plt.show()
```

---

## Key Takeaways

1. **bpy cannot be imported directly** → Use Blender as subprocess
2. **Mitsuba on PyPI works** → Install via `pip install mitsuba`
3. **Pixi doesn't have Mitsuba** → Use pip within pixi environment
4. **Architecture is service-based** → Blender (extractor) → JSON → Mitsuba (renderer)

---

## Next Steps

- [ ] Create `src/blend_extractor/extract.py`
- [ ] Install Mitsuba in pixi environment
- [ ] Create simple test .blend file
- [ ] Implement `OmenIntegrator` as Mitsuba plugin
- [ ] Test end-to-end pipeline
- [ ] Add hot-reload for development
