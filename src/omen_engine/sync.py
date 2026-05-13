"""OmenSync — Extract scene data from Blender's depsgraph.

Reads the evaluated depsgraph (handles geometry nodes) and
produces numpy arrays suitable for any Backend implementation.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class OmenSync:
    """Synchronises Blender depsgraph to backend-friendly numpy arrays."""

    def sync(
        self,
        depsgraph: Any,
    ) -> dict[str, Any]:
        """Full scene sync. Returns dict with vertices, faces,
        camera, and lights arrays."""
        vertices, faces = self._sync_meshes(depsgraph)
        camera = self._sync_camera(depsgraph)
        lights = self._sync_lights(depsgraph)

        return {
            "vertices": vertices,
            "faces": faces,
            **camera,
            "lights": lights,
        }

    def _sync_meshes(
        self, depsgraph: Any,
    ) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        all_verts: list[NDArray[np.float32]] = []
        all_faces: list[NDArray[np.int32]] = []
        offset = 0

        for obj in depsgraph.objects:
            if obj.type != "MESH":
                continue
            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()
            if mesh is None:
                continue

            mat = np.array(obj.matrix_world, dtype=np.float32)
            verts = np.array([mat[:3, :3] @ v.co + mat[:3, 3]
                              for v in mesh.vertices], dtype=np.float32)
            face_indices: list[list[int]] = []
            for poly in mesh.polygons:
                if len(poly.vertices) >= 3:
                    for i in range(len(poly.vertices) - 2):
                        face_indices.append([
                            poly.vertices[0] + offset,
                            poly.vertices[i + 1] + offset,
                            poly.vertices[i + 2] + offset,
                        ])
            eval_obj.to_mesh_clear()

            if len(verts) > 0 and len(face_indices) > 0:
                all_verts.append(verts)
                all_faces.append(np.array(face_indices, dtype=np.int32))
                offset += len(verts)

        if not all_verts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

        return np.concatenate(all_verts), np.concatenate(all_faces)

    def _sync_camera(self, depsgraph: Any) -> dict[str, Any]:
        scene = depsgraph.scene
        cam_obj = scene.camera
        if cam_obj is None:
            return {
                "camera_matrix": np.eye(4, dtype=np.float32),
                "camera_fov": 50.0,
                "width": 1920,
                "height": 1080,
            }

        cam = cam_obj.data
        mat = np.array(cam_obj.matrix_world, dtype=np.float32)
        aspect = scene.render.resolution_x / max(scene.render.resolution_y, 1)
        fov = cam.angle * (180.0 / 3.14159265)
        if aspect > 1.0:
            fov = 2.0 * np.arctan(np.tan(cam.angle / 2.0) * aspect) * (180.0 / np.pi)

        return {
            "camera_matrix": mat,
            "camera_fov": float(fov),
            "width": scene.render.resolution_x,
            "height": scene.render.resolution_y,
        }

    def _sync_lights(self, depsgraph: Any) -> list[dict[str, Any]]:
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
                direction = (mat[:3, :3] @ np.array([0, 0, -1], dtype=np.float32))
                lights.append({
                    "type": "distant",
                    "direction": direction.tolist(),
                    "color": [c * energy for c in color],
                })
            elif light.type == "AREA":
                lights.append({
                    "type": "point", "position": pos,
                    "color": [c * energy for c in color],
                })
            elif light.type == "SPOT":
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
