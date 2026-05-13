"""Main Blender-to-Mitsuba scene converter.

Tasks 15.1-15.5: Load .blend via bpy, iterate objects,
extract geometry/materials/lights/camera into Mitsuba scene dict.
"""

import logging

import numpy as np

logger = logging.getLogger("omen.converter.blend_to_mitsuba")


def convert_scene(filepath: str | None = None) -> dict:
    """Convert a Blender scene to Mitsuba scene dict."""
    import bpy
    if filepath:
        bpy.ops.wm.open_mainfile(filepath=filepath)

    scene_dict = {"type": "scene", "integrator": {"type": "path", "max_depth": 8}}
    mesh_count = light_count = 0

    for obj in bpy.data.objects:
        if obj.type == "MESH":
            mesh_dict = _extract_mesh(obj)
            if mesh_dict:
                name = _safe_name(obj.name)
                scene_dict[name] = mesh_dict
                mesh_count += 1
                mat = _extract_material(obj)
                if mat:
                    scene_dict[f"{name}_bsdf"] = mat
                    mesh_dict["bsdf"] = {"type": "ref", "id": f"{name}_bsdf"}
        elif obj.type == "LIGHT":
            light_dict = _extract_light(obj)
            if light_dict:
                scene_dict[f"light_{light_count}"] = light_dict
                light_count += 1

    cam = _extract_camera()
    if cam:
        scene_dict["sensor"] = cam
    logger.info("Converted: %d meshes, %d lights, camera=%s",
                mesh_count, light_count, bool(cam))
    return scene_dict


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace(".", "_").lower()


def _extract_mesh(obj) -> dict | None:
    """Extract mesh: vertices, faces (triangulated), transform."""
    try:
        import bpy
        dg = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(dg)
        mesh = obj_eval.to_mesh()
        if not mesh or len(mesh.vertices) == 0:
            return None

        verts = np.array([v.co for v in mesh.vertices], dtype=np.float32)
        faces = []
        for poly in mesh.polygons:
            idx = list(poly.vertices)
            if len(idx) == 3:
                faces.append(idx)
            elif len(idx) == 4:
                faces.append([idx[0], idx[1], idx[2]])
                faces.append([idx[0], idx[2], idx[3]])
        faces = np.array(faces, dtype=np.uint32)

        result = {
            "type": "ply_mesh", "vertex_count": len(verts),
            "face_count": len(faces),
            "vertex_positions": verts.flatten().tolist(),
            "faces": faces.flatten().tolist(),
        }
        mat4 = np.array(obj.matrix_world, dtype=np.float32)
        if not np.allclose(mat4, np.eye(4)):
            result["to_world"] = _mat4_to_transform(mat4)
        obj_eval.to_mesh_clear()
        return result
    except Exception as exc:
        logger.warning("Mesh %s failed: %s", obj.name, exc)
        return None


def _extract_material(obj) -> dict | None:
    from omen.converter.material_converter import convert_material
    if obj.data.materials:
        mat = obj.data.materials[0]
        if mat:
            return convert_material(mat)
    return None


def _extract_light(obj) -> dict | None:
    """Convert Point/Area/Sun/Spot -> Mitsuba light."""
    light = obj.data
    color = list(light.color)
    energy = light.energy
    mat4 = np.array(obj.matrix_world, dtype=np.float32)
    pos = mat4[:3, 3].tolist()
    lt = light.type

    if lt == "POINT":
        return {"type": "point", "position": pos,
                "intensity": _scale(color, energy, 0.01)}
    elif lt == "SUN":
        return {"type": "directional", "direction": (-mat4[:3, 2]).tolist(),
                "irradiance": _scale(color, energy, 0.001)}
    elif lt == "SPOT":
        return {"type": "spot", "position": pos,
                "cutoff_angle": light.spot_size / 2,
                "intensity": _scale(color, energy, 0.01)}
    elif lt == "AREA":
        return {"type": "rectangle", "to_world": _mat4_to_transform(mat4),
                "radiance": _scale(color, energy, 0.01)}
    return None


def _extract_camera() -> dict | None:
    """Extract camera: fov, clip, transform."""
    import bpy
    cam_obj = bpy.context.scene.camera
    if not cam_obj:
        return None
    cam = cam_obj.data
    res_x = bpy.context.scene.render.resolution_x
    res_y = max(bpy.context.scene.render.resolution_y, 1)
    return {
        "type": "perspective", "fov": float(np.degrees(cam.angle)),
        "fov_axis": "x", "near_clip": cam.clip_start, "far_clip": cam.clip_end,
        "to_world": _mat4_to_transform(
            np.array(cam_obj.matrix_world, dtype=np.float32)),
        "film": {"type": "hdrfilm", "width": res_x, "height": res_y},
    }


def _scale(color, energy, factor):
    return [c * energy * factor for c in color]


def _mat4_to_transform(mat):
    return {"type": "matrix", "value": mat.T.flatten().tolist()}
