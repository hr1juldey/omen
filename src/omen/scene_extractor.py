"""Extract scene graph from Mitsuba scene for JEPA encoding.

Returns a Python dict of numpy arrays:
    {geometry: np.array, materials: np.array, lights: np.array, camera: np.array}
"""

import logging
import numpy as np

logger = logging.getLogger("omen.scene_extractor")


def extract_scene_graph(scene) -> dict:
    """Extract complete scene graph from a Mitsuba scene.

    Args:
        scene: mi.Scene object from mi.load_file() or mi.load_dict()

    Returns:
        dict with keys: geometry, materials, lights, camera
        Each value is a numpy array or dict of numpy arrays.
    """
    geometry = _extract_geometry(scene)
    materials = _extract_materials(scene)
    lights = _extract_lights(scene)
    camera = _extract_camera(scene)

    return {
        "geometry": geometry,
        "materials": materials,
        "lights": lights,
        "camera": camera,
    }


def _extract_geometry(scene) -> dict:
    """Extract vertex positions, face indices, normals, and material IDs.

    Returns:
        dict with keys: vertices (Nx3), faces (Fx3), normals (Nx3), material_ids (F,)
    """
    all_vertices = []
    all_faces = []
    all_normals = []
    all_material_ids = []
    vertex_offset = 0

    bsdf_to_id = {}
    next_mat_id = 0

    for shape in scene.shapes():
        # Get BSDF and assign material ID
        bsdf = shape.bsdf()
        bsdf_key = id(bsdf)
        if bsdf_key not in bsdf_to_id:
            bsdf_to_id[bsdf_key] = next_mat_id
            next_mat_id += 1
        mat_id = bsdf_to_id[bsdf_key]

        # Extract vertices
        params = scene.parameters() if hasattr(scene, 'parameters') else {}
        vertex_positions = None
        try:
            import mitsuba as mi
            params = mi.traverse(shape)
            if 'vertex_positions' in params:
                vp = np.array(params['vertex_positions'])
                vertex_positions = vp.reshape(-1, 3)
        except Exception as e:
            logger.warning(f"Could not extract vertices: {e}")
            continue

        if vertex_positions is None:
            continue

        num_verts = vertex_positions.shape[0]
        all_vertices.append(vertex_positions)

        # Extract face indices
        face_indices = None
        try:
            import mitsuba as mi
            fi = shape.face_indices(0)
            if fi is not None:
                face_indices = np.array(fi).reshape(-1, 3) + vertex_offset
        except Exception:
            pass

        if face_indices is not None:
            all_faces.append(face_indices)
            all_material_ids.append(np.full(face_indices.shape[0], mat_id, dtype=np.uint32))

        # Extract normals
        normals = None
        try:
            if shape.has_vertex_normals():
                params = mi.traverse(shape)
                if 'vertex_normals' in params:
                    vn = np.array(params['vertex_normals'])
                    normals = vn.reshape(-1, 3)
        except Exception:
            pass

        if normals is not None:
            all_normals.append(normals)
        else:
            all_normals.append(np.zeros_like(vertex_positions))

        vertex_offset += num_verts

    result = {}
    if all_vertices:
        result["vertices"] = np.concatenate(all_vertices, axis=0).astype(np.float32)
    else:
        result["vertices"] = np.zeros((0, 3), dtype=np.float32)

    if all_faces:
        result["faces"] = np.concatenate(all_faces, axis=0).astype(np.uint32)
    else:
        result["faces"] = np.zeros((0, 3), dtype=np.uint32)

    if all_normals:
        result["normals"] = np.concatenate(all_normals, axis=0).astype(np.float32)
    else:
        result["normals"] = np.zeros_like(result["vertices"])

    if all_material_ids:
        result["material_ids"] = np.concatenate(all_material_ids, axis=0).astype(np.uint32)
    else:
        result["material_ids"] = np.zeros(0, dtype=np.uint32)

    return result


def _extract_materials(scene) -> dict:
    """Extract material parameters from scene BSDFs.

    Returns:
        dict with key 'params' (MxP array of material parameters)
        and 'type_ids' (M array of material type IDs).
        Material type IDs: 0=diffuse, 1=rough, 2=glossy, 3=glass, 4=metal, 5=unknown
    """
    seen_bsdfs = set()
    all_params = []
    all_type_ids = []

    for shape in scene.shapes():
        bsdf = shape.bsdf()
        bsdf_key = id(bsdf)
        if bsdf_key in seen_bsdfs:
            continue
        seen_bsdfs.add(bsdf_key)

        import mitsuba as mi
        params = mi.traverse(bsdf)
        bsdf_class = type(bsdf).__name__

        # Extract parameters based on BSDF type
        if 'PrincipledBSDF' in bsdf_class:
            mat_params = _extract_principled(params)
            type_id = 2  # glossy
        elif 'RoughBSDF' in bsdf_class or 'RoughDiffuse' in bsdf_class:
            mat_params = _extract_rough(params)
            type_id = 1  # rough
        elif 'DiffuseBSDF' in bsdf_class or 'Lambertian' in bsdf_class:
            mat_params = _extract_diffuse(params)
            type_id = 0  # diffuse
        else:
            mat_params = _extract_generic(params)
            type_id = 5  # unknown

        all_params.append(mat_params)
        all_type_ids.append(type_id)

    result = {}
    if all_params:
        result["params"] = np.array(all_params, dtype=np.float32)
        result["type_ids"] = np.array(all_type_ids, dtype=np.uint32)
    else:
        result["params"] = np.zeros((0, 5), dtype=np.float32)
        result["type_ids"] = np.zeros(0, dtype=np.uint32)

    return result


def _extract_principled(params) -> list:
    """Extract Principled BSDF parameters: diffuse_rgb(3), roughness(1), metallic(1)."""
    diffuse = np.array(params.get('diffuse_reflectance.value', [0.5, 0.5, 0.5]))
    if diffuse.ndim > 0 and len(diffuse) >= 3:
        diffuse = diffuse[:3]
    else:
        diffuse = np.array([0.5, 0.5, 0.5])
    roughness = float(params.get('alpha.value', 0.5))
    metallic = float(params.get('metallic.value', 0.0))
    return list(diffuse.flatten())[:3] + [roughness, metallic]


def _extract_rough(params) -> list:
    """Extract Rough BSDF parameters: diffuse_rgb(3), alpha(1), specular(1)."""
    diffuse = np.array(params.get('diffuse_reflectance.value', [0.5, 0.5, 0.5]))
    if diffuse.ndim > 0 and len(diffuse) >= 3:
        diffuse = diffuse[:3]
    else:
        diffuse = np.array([0.5, 0.5, 0.5])
    alpha = float(params.get('alpha.value', 0.5))
    specular = float(params.get('specular_reflectance.value', 0.5))
    return list(diffuse.flatten())[:3] + [alpha, specular]


def _extract_diffuse(params) -> list:
    """Extract Diffuse BSDF: reflectance_rgb(3), zeros(2)."""
    reflectance = np.array(params.get('reflectance.value', [0.5, 0.5, 0.5]))
    if reflectance.ndim > 0 and len(reflectance) >= 3:
        reflectance = reflectance[:3]
    else:
        reflectance = np.array([0.5, 0.5, 0.5])
    return list(reflectance.flatten())[:3] + [0.0, 0.0]


def _extract_generic(params) -> list:
    """Fallback: extract whatever params are available."""
    values = []
    for key in list(params.keys())[:5]:
        val = params[key]
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            values.append(0.0)
    while len(values) < 5:
        values.append(0.0)
    return values[:5]


def _extract_lights(scene) -> dict:
    """Extract light parameters.

    Returns:
        dict with key 'params' (Lx7 array: type_id(1), position/color(6))
        Light type IDs: 0=point, 1=area, 2=environment
    """
    all_lights = []

    for emitter in scene.emitters():
        import mitsuba as mi

        # Check if environment light
        try:
            is_env = emitter.is_environment()
        except Exception:
            is_env = False

        if is_env:
            # Environment light: type=2, radiance(3), zeros(3)
            params = mi.traverse(emitter)
            radiance = np.array(params.get('radiance.value', [1.0, 1.0, 1.0]))
            if radiance.ndim > 0 and len(radiance) >= 3:
                radiance = radiance[:3]
            else:
                radiance = np.array([1.0, 1.0, 1.0])
            light_data = [2] + list(radiance.flatten())[:3] + [0.0, 0.0, 0.0]
        else:
            # Check for point light vs area light
            emitter_class = type(emitter).__name__
            params = mi.traverse(emitter)

            if 'Point' in emitter_class:
                position = np.array(params.get('position', [0, 0, 0]))
                intensity = np.array(params.get('intensity.value', [1.0, 1.0, 1.0]))
                if position.ndim > 0:
                    position = position[:3]
                else:
                    position = np.array([0, 0, 0])
                if intensity.ndim > 0 and len(intensity) >= 3:
                    intensity = intensity[:3]
                else:
                    intensity = np.array([1.0, 1.0, 1.0])
                light_data = [0] + list(position.flatten())[:3] + list(intensity.flatten())[:3]
            else:
                # Area light or other: type=1, zeros(3), radiance(3)
                radiance = np.array(params.get('radiance.value', [1.0, 1.0, 1.0]))
                if radiance.ndim > 0 and len(radiance) >= 3:
                    radiance = radiance[:3]
                else:
                    radiance = np.array([1.0, 1.0, 1.0])
                light_data = [1, 0.0, 0.0, 0.0] + list(radiance.flatten())[:3]

        all_lights.append(light_data)

    result = {}
    if all_lights:
        result["params"] = np.array(all_lights, dtype=np.float32)
    else:
        result["params"] = np.zeros((0, 7), dtype=np.float32)

    return result


def _extract_camera(scene) -> dict:
    """Extract camera parameters: transform(16), fov(1), near(1), far(1), aspect(1), film_size(2).

    Returns:
        dict with key 'params' (22 array).
    """
    sensors = scene.sensors()
    if not sensors:
        return {"params": np.zeros(21, dtype=np.float32)}

    sensor = sensors[0]
    import mitsuba as mi
    params = mi.traverse(sensor)

    # Camera transform (4x4 = 16 floats)
    try:
        to_world = params.get('to_world')
        if to_world is not None:
            import drjit as dr
            transform_matrix = np.array(to_world).reshape(4, 4)
            transform_flat = transform_matrix.flatten()
        else:
            transform_flat = np.eye(4).flatten()
    except Exception:
        transform_flat = np.eye(4).flatten()

    # FOV
    fov = float(params.get('fov', 45.0))

    # Near/far clip
    near = float(params.get('near_clip', 0.01))
    far = float(params.get('far_clip', 10000.0))

    # Film size
    try:
        film_size = np.array(params.get('film.size', [256, 256]))
        aspect = film_size[0] / max(film_size[1], 1)
    except Exception:
        film_size = np.array([256, 256])
        aspect = 1.0

    camera_data = np.concatenate([
        transform_flat[:16],
        [fov, near, far, aspect],
        film_size[:2].astype(np.float32)
    ]).astype(np.float32)

    return {"params": camera_data}
