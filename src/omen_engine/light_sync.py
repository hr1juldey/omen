"""Light sync — extract light data from Blender depsgraph.

Produces a list of light dicts (type, position/direction, color)
suitable for MitsubaBackend.build_scene().
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def extract_lights(depsgraph: Any) -> list[dict[str, Any]]:
    """Extract light sources from depsgraph objects.

    Falls back to a single point light at [5,5,5] if none found.
    """
    lights: list[dict[str, Any]] = []

    for obj in depsgraph.objects:
        if obj.type != "LIGHT":
            continue
        light = obj.data
        mat = np.array(obj.matrix_world, dtype=np.float32)
        pos = mat[:3, 3].tolist()
        color = list(light.color)
        energy = light.energy

        if light.type == "POINT":
            lights.append({
                "type": "point", "position": pos,
                "color": [c * energy for c in color],
            })
        elif light.type == "SUN":
            direction = mat[:3, :3] @ np.array([0, 0, -1], dtype=np.float32)
            lights.append({
                "type": "distant",
                "direction": direction.tolist(),
                "color": [c * energy for c in color],
            })
        elif light.type in ("AREA", "SPOT"):
            lights.append({
                "type": "point", "position": pos,
                "color": [c * energy for c in color],
            })

    if not lights:
        lights.append({
            "type": "point", "position": [5, 5, 5],
            "color": [100, 100, 100],
        })
        logger.warning("No lights found, added fallback point light")

    return lights
