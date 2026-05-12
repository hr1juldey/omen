# Omen Render Engine - Quick Start

## One-Command Setup

### Linux / macOS

```bash
pixi install
```

That's it! Run this one command and everything is installed:
- ✅ Python 3.14
- ✅ Mojo / MAX
- ✅ Mitsuba 3 renderer
- ✅ All dependencies

## Verify Installation

```bash
pixi run python -c "import mitsuba as mi; print(f'Omen ready! Mitsuba {mi.__version__}')"
```

Expected output:
```
Omen ready! Mitsuba 3.8.0
```

## Usage

### Render a scene with Omen

```bash
pixi run python your_script.py
```

### Start a Python session with Omen

```bash
pixi run python
```

Then:
```python
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

# Load and render scene
scene = mi.load_dict(mi.cornell_box())
image = mi.render(scene)
mi.Bitmap(image).write('output.exr')
```

## Troubleshooting

### "pixi: command not found"

Install pixi:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### "ModuleNotFoundError: No module named 'mitsuba'"

Run setup again:
```bash
pixi install
```

### Python version issues

Pixi uses Python 3.14. Don't use system Python or other venvs.

## For Developers

```bash
# Install dev tools
pixi install --feature dev

# Run linting
pixi run ruff check .

# Run tests
pixi run pytest
```
