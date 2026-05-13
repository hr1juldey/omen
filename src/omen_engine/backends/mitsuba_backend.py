"""Mitsuba 3 backend — path tracing with omen_integrator."""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from omen_engine.backends import Backend

logger = logging.getLogger(__name__)


class MitsubaBackend(Backend):
    """Backend that delegates path tracing to Mitsuba 3."""

    def __init__(self) -> None:
        self._scene = None
        self._sensor = None
        self._aovs: dict[str, NDArray[np.float32]] = {}
        self._register_integrator()

    def _register_integrator(self) -> None:
        try:
            from omen_integrator import register
            register()
            logger.info("Omen integrator registered with Mitsuba")
        except ImportError:
            logger.warning("omen_integrator not found, using Mitsuba path tracer")

    def load_scene(
        self,
        vertices: NDArray[np.float32],
        faces: NDArray[np.int32],
        camera_matrix: NDArray[np.float32],
        camera_fov: float,
        width: int,
        height: int,
        lights: list[dict[str, Any]],
    ) -> None:
        import mitsuba as mi
        mi.set_variant("llvm_ad_rgb")

        mesh = mi.Mesh(
            "omen_mesh",
            vertex_count=vertices.shape[0],
            face_count=faces.shape[0],
            has_vertex_normals=False,
            has_vertex_texcoords=False,
        )
        mesh.vertex_positions_buffer()[:] = vertices.ravel()
        mesh.faces_buffer()[:] = faces.ravel()

        params = mi.traverse(mesh)
        params.update()

        cam_np = np.array(camera_matrix, dtype=np.float64).reshape(4, 4)
        cam_np = np.linalg.inv(cam_np)
        sensor = mi.load_dict({
            "type": "perspective",
            "fov": float(camera_fov),
            "to_world": mi.ScalarTransform4f(cam_np.tolist()),
            "film": {
                "type": "hdrfilm",
                "width": width,
                "height": height,
                "pixel_format": "RGB",
                "component_format": "float32",
            },
            "sampler": {"type": "independent", "sample_count": 1},
        })

        emitters = []
        for lt in lights:
            if lt["type"] == "point":
                emitters.append({
                    "type": "point",
                    "position": lt["position"],
                    "intensity": mi.ScalarColor3f(*lt.get("color", [1, 1, 1])),
                })
            elif lt["type"] == "distant":
                emitters.append({
                    "type": "directional",
                    "direction": lt["direction"],
                    "irradiance": mi.ScalarColor3f(*lt.get("color", [1, 1, 1])),
                })

        scene_dict: dict[str, Any] = {
            "type": "scene",
            "mesh": mesh,
            "sensor": sensor,
            "integrator": {"type": "omen"},
        }
        for i, em in enumerate(emitters):
            scene_dict[f"light_{i}"] = em

        self._scene = mi.load_dict(scene_dict)
        self._sensor = sensor
        logger.info(
            "Loaded scene: %d verts, %d faces, %d lights",
            vertices.shape[0], faces.shape[0], len(lights),
        )

    def render(self, spp: int, max_depth: int) -> dict[str, NDArray[np.float32]]:
        import mitsuba as mi
        if self._scene is None:
            raise RuntimeError("No scene loaded. Call load_scene() first.")

        result = mi.render(
            self._scene,
            sensor=self._sensor,
            spp=spp,
            max_depth=max_depth,
        )

        if isinstance(result, mi.Bitmap):
            buf = np.array(result, dtype=np.float32)
            self._aovs["color"] = buf[:, :, :3]
        else:
            for key in ["color", "albedo", "normal", "depth"]:
                if key in result:
                    self._aovs[key] = np.array(result[key], dtype=np.float32)
            if "color" not in self._aovs and len(result) > 0:
                first = next(iter(result.values()))
                self._aovs["color"] = np.array(first, dtype=np.float32)

        return self._aovs

    def get_aov_buffers(self) -> dict[str, NDArray[np.float32]]:
        return dict(self._aovs)
