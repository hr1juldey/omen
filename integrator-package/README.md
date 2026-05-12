# Omen Mitsuba Integrator

JEPA-accelerated path tracing integrator for Mitsuba 3.

## Installation

```bash
pip install -e integrator-package/
```

## Usage

```python
import mitsuba as mi
from omen_integrator import register

register()  # Registers "omen" integrator

scene = mi.load_dict({
    'type': 'scene',
    'integrator': {'type': 'omen'}
})
```

## Parameters

- `max_depth`: Maximum path bounces (default: -1 for infinite)
- `rr_depth`: Russian roulette start depth (default: 5)
- `jepa_model`: Path to JEPA model (future)
- `use_gpu`: Enable GPU (future)
