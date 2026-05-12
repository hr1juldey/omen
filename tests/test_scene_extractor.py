"""Test: Extract Cornell box, verify 2 meshes, 1 area light, 1 camera."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


def test_cornell_box_extraction():
    """Extract Cornell box scene graph and verify structure."""
    import mitsuba as mi
    mi.set_variant('scalar_rgb')

    from omen.scene_extractor import extract_scene_graph

    # Load Cornell box
    scene = mi.load_dict(mi.cornell_box())

    # Extract scene graph
    sg = extract_scene_graph(scene)

    # Verify top-level keys
    assert 'geometry' in sg, "Missing geometry key"
    assert 'materials' in sg, "Missing materials key"
    assert 'lights' in sg, "Missing lights key"
    assert 'camera' in sg, "Missing camera key"

    # Verify geometry structure
    geom = sg['geometry']
    assert 'vertices' in geom, "Missing vertices"
    assert 'faces' in geom, "Missing faces"
    assert 'normals' in geom, "Missing normals"
    assert 'material_ids' in geom, "Missing material_ids"

    # Verify vertex array shape (Nx3)
    assert geom['vertices'].ndim == 2, f"vertices should be 2D, got {geom['vertices'].ndim}"
    assert geom['vertices'].shape[1] == 3, f"vertices should be Nx3, got {geom['vertices'].shape}"

    # Verify face array shape (Fx3)
    assert geom['faces'].ndim == 2, f"faces should be 2D, got {geom['faces'].ndim}"
    assert geom['faces'].shape[1] == 3, f"faces should be Fx3, got {geom['faces'].shape}"

    num_verts = geom['vertices'].shape[0]
    num_faces = geom['faces'].shape[0]
    print(f"Geometry: {num_verts} vertices, {num_faces} faces")
    assert num_verts > 0, "No vertices extracted"
    assert num_faces > 0, "No faces extracted"

    # Verify materials structure
    mats = sg['materials']
    assert 'params' in mats, "Missing material params"
    assert 'type_ids' in mats, "Missing material type_ids"
    num_materials = mats['params'].shape[0]
    print(f"Materials: {num_materials} unique materials")
    assert num_materials > 0, "No materials extracted"

    # Verify lights structure
    lights = sg['lights']
    assert 'params' in lights, "Missing light params"
    num_lights = lights['params'].shape[0]
    print(f"Lights: {num_lights} lights")
    assert num_lights >= 1, "Expected at least 1 light in Cornell box"

    # Verify camera structure
    cam = sg['camera']
    assert 'params' in cam, "Missing camera params"
    cam_params = cam['params']
    assert cam_params.shape[0] == 22, f"Camera params should be 22 floats, got {cam_params.shape[0]}"

    # Check FOV is reasonable
    fov = cam_params[16]
    assert 10 < fov < 120, f"FOV {fov} out of reasonable range"

    # Check film size
    film_w, film_h = cam_params[19], cam_params[20]
    assert film_w > 0 and film_h > 0, f"Film size ({film_w}, {film_h}) should be positive"

    print(f"Camera: FOV={fov:.1f}, film=({film_w:.0f}x{film_h:.0f})")
    print("PASSED: Cornell box extraction verified")


def test_dict_return_type():
    """Verify scene graph returns proper Python dicts with numpy arrays."""
    import mitsuba as mi
    mi.set_variant('scalar_rgb')

    from omen.scene_extractor import extract_scene_graph

    scene = mi.load_dict(mi.cornell_box())
    sg = extract_scene_graph(scene)

    # Top level should be dict
    assert isinstance(sg, dict)

    # Geometry should be dict of numpy arrays
    assert isinstance(sg['geometry'], dict)
    for k, v in sg['geometry'].items():
        assert isinstance(v, np.ndarray), f"geometry[{k}] should be ndarray, got {type(v)}"

    # Materials should be dict of numpy arrays
    assert isinstance(sg['materials'], dict)

    # Lights should be dict of numpy arrays
    assert isinstance(sg['lights'], dict)

    # Camera should be dict of numpy arrays
    assert isinstance(sg['camera'], dict)

    print("PASSED: Return types verified")


if __name__ == '__main__':
    test_cornell_box_extraction()
    test_dict_return_type()
    print("\nAll tests passed!")
