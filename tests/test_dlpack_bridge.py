"""Test: Transfer tensors to Nabla and back, verify values match."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np


def test_numpy_roundtrip():
    """Test numpy -> bridge.to_nabla -> bridge.to_numpy round-trip."""
    from omen.jepa_bridge import JEPABridge

    # Create bridge without model (just testing tensor transfer)
    bridge = JEPABridge(model_path=None)

    # Test data
    original = np.random.randn(4, 256, 256, 4).astype(np.float32)

    # Transfer to Nabla
    nb_tensor = bridge.to_nabla(original)
    assert nb_tensor is not None, "to_nabla returned None"

    # Transfer back to numpy
    result = bridge.to_numpy(nb_tensor)
    assert result is not None, "to_numpy returned None"

    # Verify values match
    assert np.allclose(original, result, atol=1e-6), \
        f"Values don't match after round-trip: max diff = {np.max(np.abs(original - result))}"

    print(f"PASSED: numpy round-trip (shape {original.shape}, max diff = {np.max(np.abs(original - result)):.2e})")


def test_alpha_channel():
    """Test alpha channel addition to RGB render."""
    from omen.jepa_bridge import JEPABridge

    bridge = JEPABridge(model_path=None)

    # RGB input
    rgb = np.random.rand(256, 256, 3).astype(np.float32)
    rgba = bridge.add_alpha(rgb)

    assert rgba.shape == (256, 256, 4), f"Expected (256,256,4), got {rgba.shape}"
    assert np.allclose(rgba[:, :, 3], 1.0), "Alpha channel should be all 1.0"
    assert np.allclose(rgba[:, :, :3], rgb), "RGB channels should be unchanged"

    # Already RGBA input
    rgba_input = np.random.rand(256, 256, 4).astype(np.float32)
    rgba_output = bridge.add_alpha(rgba_input)
    assert rgba_output.shape == (256, 256, 4), "Should pass through RGBA unchanged"

    print("PASSED: alpha channel addition")


def test_scene_graph_conversion():
    """Test converting scene graph dict to Nabla tensors."""
    from omen.jepa_bridge import JEPABridge

    bridge = JEPABridge(model_path=None)

    scene_graph = {
        "geometry": np.random.randn(100, 3).astype(np.float32),
        "materials": np.random.randn(5, 7).astype(np.float32),
        "camera": np.random.randn(22).astype(np.float32),
    }

    # Convert each value
    nb_scene = {k: bridge.to_nabla(v) for k, v in scene_graph.items()}

    # Convert back and verify
    for key in scene_graph:
        original = scene_graph[key]
        recovered = bridge.to_numpy(nb_scene[key])
        assert np.allclose(original, recovered, atol=1e-6), \
            f"Scene graph key '{key}' doesn't match after round-trip"

    print("PASSED: scene graph conversion")


def test_graceful_degradation():
    """Test bridge works even when Nabla model is unavailable."""
    from omen.jepa_bridge import JEPABridge

    # Should not crash even with invalid path
    bridge = JEPABridge(model_path="/nonexistent/model.omen")
    # available may be True or False depending on Nabla installation
    # but it should not raise

    # Passthrough mode
    test_input = np.random.rand(256, 256, 4).astype(np.float32)
    result = bridge.denoise({}, test_input, 256, 256)
    assert result.shape == test_input.shape, "Denoise passthrough should return same shape"

    print(f"PASSED: graceful degradation (available={bridge.available})")


if __name__ == '__main__':
    test_numpy_roundtrip()
    test_alpha_channel()
    test_scene_graph_conversion()
    test_graceful_degradation()
    print("\nAll DLPack bridge tests passed!")
