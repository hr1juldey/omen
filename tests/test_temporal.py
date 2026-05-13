"""Tests for Section 14: Temporal Coherence & JEPA World Model.

Task 14.3: Surprise detection MSE z-score.
Task 14.5: Jump cut detection.
Task 14.4: Auto-surprise for structural changes.
Task 14.2: Scene delta computation.
Task 14.10: Animation renderer end-to-end.
"""

import numpy as np


def test_14_2_scene_delta():
    """Task 14.2: Scene graph diff computation."""
    from omen.temporal import compute_scene_delta

    prev = {
        "geometry": np.random.randn(100, 3).astype(np.float32),
        "materials": np.random.randn(5, 10).astype(np.float32),
        "lights": np.random.randn(3, 6).astype(np.float32),
        "camera": np.array([0, 0, 5, 0, 0, 0, 1], dtype=np.float32),
    }
    # Small camera movement, same topology
    curr = prev.copy()
    curr["camera"] = np.array([0.1, 0, 5, 0, 0, 0, 1], dtype=np.float32)

    delta = compute_scene_delta(prev, curr)
    assert not delta["has_structural_change"]
    assert delta["camera_translation"] > 0
    assert delta["camera_translation"] < 1.0

    # Different topology -> structural change
    curr2 = dict(curr, geometry=np.random.randn(200, 3).astype(np.float32))
    delta2 = compute_scene_delta(prev, curr2)
    assert delta2["has_structural_change"]
    print("14.2 PASS: scene delta computation verified")


def test_14_3_surprise_detection():
    """Task 14.3: MSE z-score > 2 sigma surprise detection."""
    from omen.temporal import detect_surprise

    latent = np.random.randn(1, 64).astype(np.float32)
    # Small diff -> no surprise
    is_s, mse, z, _, _ = detect_surprise(latent, latent + 0.01, 0.0, 1.0)
    assert not is_s, "Small diff should not be surprise"

    # Large diff -> surprise
    is_s2, mse2, z2, _, _ = detect_surprise(latent, latent + 10.0, 0.0, 1.0)
    assert is_s2, "Large diff should be surprise"
    assert z2 > 2.0
    print(f"14.3 PASS: surprise detection (z={z2:.1f})")


def test_14_4_auto_surprise():
    """Task 14.4: Auto-surprise for structural changes."""
    from omen.temporal import detect_auto_surprise

    # No structural change
    delta = {"has_structural_change": False, "geometry_delta": 0.01}
    assert not detect_auto_surprise(delta)

    # Structural change -> auto surprise
    delta2 = {"has_structural_change": True}
    assert detect_auto_surprise(delta2)

    # Large geometry delta
    delta3 = {"has_structural_change": False, "geometry_delta": 1.0}
    assert detect_auto_surprise(delta3)
    print("14.4 PASS: auto-surprise detection verified")


def test_14_5_jump_cut():
    """Task 14.5: Jump cut detection."""
    from omen.temporal import detect_jump_cut

    assert not detect_jump_cut({"camera_translation": 0.5, "camera_rotation": 0.1})
    assert detect_jump_cut({"camera_translation": 2.0, "camera_rotation": 0.1})
    assert detect_jump_cut({"camera_translation": 0.1, "camera_rotation": 1.0})
    print("14.5 PASS: jump cut detection verified")


def test_14_10_animation_renderer():
    """Task 14.10: AnimationRenderer API validation."""
    from omen.modes.animation import AnimationRenderer

    # Test with mock bridge (model unavailable)
    class MockBridge:
        available = False

    renderer = AnimationRenderer(None, MockBridge())
    assert renderer.frame_count == 0
    assert len(renderer.history) == 0
    assert renderer.surprise_mean == 0.0
    print("14.10 PASS: AnimationRenderer initialized correctly")


if __name__ == "__main__":
    test_14_2_scene_delta()
    test_14_3_surprise_detection()
    test_14_4_auto_surprise()
    test_14_5_jump_cut()
    test_14_10_animation_renderer()
    print("All Section 14 tests passed")
