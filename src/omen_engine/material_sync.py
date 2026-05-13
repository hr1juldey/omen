"""Material sync — extract BSDF parameters from Blender depsgraph.

Extracts base_color, roughness, metallic, emission from
Principled BSDF nodes for SceneGraphEncoder consumption.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MATERIAL = {
    "base_color": [0.8, 0.8, 0.8],
    "roughness": 0.5,
    "metallic": 0.0,
    "emission": [0.0, 0.0, 0.0],
}


def extract_materials(depsgraph: Any) -> list[dict[str, Any]]:
    """Extract material params from all mesh objects in depsgraph.

    Returns list of dicts: base_color (3), roughness (1),
    metallic (1), emission (3). 5 params for SceneGraphEncoder.
    """
    materials: list[dict[str, Any]] = []
    seen: set[int] = set()

    for obj in depsgraph.objects:
        if obj.type != "MESH":
            continue

        slots = getattr(obj, "material_slots", [])
        if not slots or slots[0].material is None:
            materials.append(dict(DEFAULT_MATERIAL))
            continue

        mat = slots[0].material
        mat_id = id(mat)
        if mat_id in seen:
            continue
        seen.add(mat_id)

        mat_data = dict(DEFAULT_MATERIAL)
        if mat.use_nodes and mat.node_tree:
            _extract_principled(mat.node_tree, mat_data)
        materials.append(mat_data)

    if not materials:
        materials.append(dict(DEFAULT_MATERIAL))

    return materials


def _extract_principled(node_tree: Any, out: dict[str, Any]) -> None:
    """Extract params from the first Principled BSDF node found."""
    for node in node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue

        inputs = node.inputs
        out["base_color"] = list(inputs["Base Color"].default_value[:3])
        out["roughness"] = float(inputs["Roughness"].default_value)
        out["metallic"] = float(inputs["Metallic"].default_value)

        emission_color = inputs.get("Emission Color")
        if emission_color is not None:
            out["emission"] = list(emission_color.default_value[:3])
        break
