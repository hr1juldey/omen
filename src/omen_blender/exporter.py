"""Export Blender scene to JSON for Omen CLI consumption.

Extracts camera, lights, meshes, and materials from the depsgraph.
No mitsuba dependency - pure bpy API.
"""

import json
import math
import os
import tempfile

import bpy
import mathutils


def export_scene(depsgraph, output_dir):
    """Export full Blender scene to JSON dict.

    Returns path to the JSON file written in output_dir.
    """
    scene = depsgraph.scene
    scale = scene.render.resolution_percentage / 100.0
    width = int(scene.render.resolution_x * scale)
    height = int(scene.render.resolution_y * scale)

    data = {
        "resolution": [width, height],
        "camera": _export_camera(scene),
        "lights": _export_lights(depsgraph),
        "objects": _export_objects(depsgraph, output_dir),
    }

    path = os.path.join(output_dir, "scene.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _export_camera(scene):
    """Export active camera parameters."""
    cam = scene.camera
    if cam is None:
        return {}
    obj = cam.matrix_world
    pos = list(obj.translation)
    # Look direction: -Z in camera space
    forward = list(obj.to_quaternion() @ mathutils.Vector((0, 0, -1)))
    up = list(obj.to_quaternion() @ mathutils.Vector((0, 1, 0)))
    fov = math.degrees(cam.data.angle)
    return {
        "position": pos,
        "forward": forward,
        "up": up,
        "fov": fov,
        "near": cam.data.clip_start,
        "far": cam.data.clip_end,
        "type": "perspective",
    }


def _export_lights(depsgraph):
    """Export all light objects from evaluated depsgraph."""
    lights = []
    for obj in depsgraph.objects:
        if obj.type != "LIGHT":
            continue
        light = obj.data
        pos = list(obj.matrix_world.translation)
        color = list(light.color)
        energy = light.energy
        entry = {"name": obj.name, "position": pos, "color": color, "energy": energy}
        if light.type == "POINT":
            entry["type"] = "point"
        elif light.type == "SUN":
            direction = list(
                obj.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
            )
            entry["type"] = "directional"
            entry["direction"] = direction
        elif light.type == "SPOT":
            entry["type"] = "spot"
            entry["angle"] = math.degrees(light.spot_size)
            direction = list(
                obj.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
            )
            entry["direction"] = direction
        elif light.type == "AREA":
            entry["type"] = "area"
            entry["size"] = [light.size, light.size_y]
        lights.append(entry)
    return lights


def _export_objects(depsgraph, output_dir):
    """Export mesh objects as OBJ files + material metadata."""
    objects = []
    for obj in depsgraph.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.to_mesh()
        if mesh is None or len(mesh.vertices) == 0:
            continue
        # Write OBJ for geometry
        obj_path = os.path.join(output_dir, f"{obj.name}.obj")
        _write_obj(obj, mesh, obj_path)
        # Extract material info
        materials = _extract_materials(mesh)
        transform = [list(row) for row in obj.matrix_world.transposed()]
        objects.append({
            "name": obj.name,
            "geometry": obj_path,
            "transform": transform,
            "materials": materials,
        })
        obj.to_mesh_clear()
    return objects


def _write_obj(obj, mesh, path):
    """Write mesh to OBJ file (vertices + faces)."""
    mesh.calc_loop_triangles()
    with open(path, "w") as f:
        f.write(f"# Omen export: {obj.name}\n")
        for v in mesh.vertices:
            f.write(f"v {v.co.x} {v.co.y} {v.co.z}\n")
        for tri in mesh.loop_triangles:
            f.write(f"f {tri.vertices[0]+1} {tri.vertices[1]+1} {tri.vertices[2]+1}\n")


def _extract_materials(mesh):
    """Extract simplified material parameters from mesh material slots."""
    materials = []
    for mat in mesh.materials:
        if mat is None:
            continue
        entry = {"name": mat.name}
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                entry["type"] = "principled"
                entry["base_color"] = list(node.inputs["Base Color"].default_value)[:3]
                entry["roughness"] = node.inputs["Roughness"].default_value
                entry["metallic"] = node.inputs["Metallic"].default_value
                entry["ior"] = node.inputs["IOR"].default_value
                alpha = node.inputs["Alpha"].default_value
                if alpha < 1.0:
                    entry["alpha"] = alpha
                transmission = node.inputs["Transmission Weight"].default_value
                if transmission > 0.01:
                    entry["transmission"] = transmission
            elif node.type == "BSDF_GLASS":
                entry["type"] = "glass"
                entry["color"] = list(node.inputs["Color"].default_value)[:3]
                entry["roughness"] = node.inputs["Roughness"].default_value
                entry["ior"] = node.inputs["IOR"].default_value
            elif node.type == "EMISSION":
                entry["emission"] = list(node.inputs["Color"].default_value)[:3]
                entry["emission_strength"] = node.inputs["Strength"].default_value
        if "type" not in entry:
            entry["type"] = "diffuse"
            entry["base_color"] = [0.8, 0.8, 0.8]
        materials.append(entry)
    return materials
