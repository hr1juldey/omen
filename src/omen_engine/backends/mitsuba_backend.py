"""Mitsuba scene builder — converts numpy arrays to mi.Scene.

Thin conversion layer ONLY. No render logic, no AOV extraction.
That all lives in src/omen/modes/denoiser.py.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omen_engine.backends.bsdf_builder import material_to_bsdf

logger = logging.getLogger(__name__)


def build_scene(
    vertices: NDArray[np.float32],
    faces: NDArray[np.int32],
    camera_matrix: NDArray[np.float32],
    camera_fov: float,
    width: int,
    height: int,
    lights: list[dict[str, Any]],
    materials: list[dict[str, Any]],
) -> Any:
    """Build an mi.Scene from depsgraph numpy arrays."""
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")

    scene_dict: dict[str, Any] = {"type": "scene"}

    if vertices.shape[0] > 0 and faces.shape[0] > 0:
        scene_dict["mesh"] = _build_mesh(vertices, faces, materials)
    else:
        scene_dict["fallback_quad"] = _fallback_quad()
        logger.warning("Empty mesh, using fallback quad")

    scene_dict["sensor"] = _build_sensor(
        camera_matrix, camera_fov, width, height,
    )

    for i, lt in enumerate(lights):
        scene_dict[f"light_{i}"] = _build_light(lt)

    logger.info(
        "Built scene: %d verts, %d faces, %d lights",
        vertices.shape[0], faces.shape[0], len(lights),
    )
    return mi.load_dict(scene_dict)


def _build_sensor(
    cam_matrix: NDArray, fov: float, w: int, h: int,
) -> Any:
    import mitsuba as mi
    cam_np = np.linalg.inv(np.array(cam_matrix, dtype=np.float64).reshape(4, 4))
    return mi.load_dict({
        "type": "perspective",
        "fov": float(fov),
        "to_world": mi.ScalarTransform4f(cam_np.tolist()),
        "film": {
            "type": "hdrfilm", "width": w, "height": h,
            "pixel_format": "RGB", "component_format": "float32",
        },
        "sampler": {"type": "independent", "sample_count": 1},
    })


def _build_mesh(
    vertices: NDArray, faces: NDArray, materials: list[dict],
) -> Any:
    import mitsuba as mi
    mesh = mi.Mesh(
        "omen_mesh",
        vertex_count=vertices.shape[0],
        face_count=faces.shape[0],
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    mesh.vertex_positions_buffer()[:] = vertices.ravel()
    mesh.faces_buffer()[:] = faces.ravel()
    mi.traverse(mesh).update()

    if materials:
        mesh.set_bsdf(material_to_bsdf(materials[0]))
    return mesh


def _build_light(lt: dict[str, Any]) -> dict[str, Any]:
    import mitsuba as mi
    color = lt.get("color", [1, 1, 1])
    if lt["type"] == "point":
        return {
            "type": "point",
            "position": lt["position"],
            "intensity": mi.ScalarColor3f(*color),
        }
    # distant / directional
    return {
        "type": "directional",
        "direction": lt["direction"],
        "irradiance": mi.ScalarColor3f(*color),
    }


def _fallback_quad() -> Any:
    import mitsuba as mi
    verts = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]], dtype=np.float32)
    faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
    mesh = mi.Mesh("fallback", vertex_count=4, face_count=2,
                   has_vertex_normals=False, has_vertex_texcoords=False)
    mesh.vertex_positions_buffer()[:] = verts.ravel()
    mesh.faces_buffer()[:] = faces.ravel()
    mi.traverse(mesh).update()
    return mesh
