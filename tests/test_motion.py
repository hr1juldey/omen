"""Tests for Section 18: Motion Blur & Temporal Reprojection."""

import numpy as np


def test_18_2_read_motion_vectors():
    """Task 18.2: Motion vector reading with fallback."""
    from omen.motion import read_motion_vectors
    # No motion vectors -> zeros
    mv = read_motion_vectors({}, 64, 64)
    assert mv.shape == (64, 64, 2)
    assert np.allclose(mv, 0)

    # With motion key
    fake = {"motion": np.random.randn(32, 32, 2).astype(np.float32)}
    mv2 = read_motion_vectors(fake, 32, 32)
    assert mv2.shape == (32, 32, 2)
    print("18.2 PASS: motion vector reading")


def test_18_3_temporal_reproject():
    """Task 18.3: Bilinear warp."""
    from omen.motion import temporal_reproject
    prev = np.random.rand(32, 32, 4).astype(np.float32)
    # Zero motion -> identical output
    motion = np.zeros((32, 32, 2), dtype=np.float32)
    result = temporal_reproject(prev, motion)
    np.testing.assert_allclose(result, prev, atol=1e-5)
    print("18.3 PASS: temporal reprojection")


def test_18_4_motion_coherence():
    """Task 18.4: Motion coherence."""
    from omen.motion import compute_motion_coherence
    # Zero motion -> coherence = 1
    mv = np.zeros((16, 16, 2), dtype=np.float32)
    c = compute_motion_coherence(mv)
    assert np.allclose(c, 1.0)

    # Large motion -> low coherence
    mv2 = np.full((16, 16, 2), 40.0, dtype=np.float32)
    c2 = compute_motion_coherence(mv2)
    assert c2.mean() < 0.5
    print("18.4 PASS: motion coherence")


def test_18_5_occlusion_mask():
    """Task 18.5: Occlusion detection."""
    from omen.motion import compute_occlusion_mask
    # Uniform motion -> no occlusion
    mv = np.full((16, 16, 2), 2.0, dtype=np.float32)
    occ = compute_occlusion_mask(mv)
    assert occ.sum() == 0

    # Sharp velocity discontinuity -> occlusion
    mv2 = np.zeros((16, 16, 2), dtype=np.float32)
    mv2[:8, :, 0] = 50.0
    occ2 = compute_occlusion_mask(mv2, threshold=5.0)
    assert occ2.sum() > 0
    print("18.5 PASS: occlusion mask")


def test_18_7_merge_reprojected():
    """Task 18.7: Merge reprojected + current."""
    from omen.motion import merge_reprojected
    reproj = np.ones((8, 8, 4), dtype=np.float32)
    noisy = np.zeros((8, 8, 4), dtype=np.float32)
    weight = np.full((8, 8), 0.5, dtype=np.float32)
    merged = merge_reprojected(reproj, noisy, weight)
    np.testing.assert_allclose(merged, 0.5, atol=1e-5)
    print("18.7 PASS: merge reprojected")


def test_18_12_motion_expert_routing():
    """Task 18.12: Motion expert routing."""
    from omen.motion import route_motion_expert
    fp = np.zeros((4, 4, 23), dtype=np.float32)
    scores = route_motion_expert(fp)
    assert scores.shape == (4, 4, 4)
    # Static scene -> expert 0 (static) should dominate
    assert scores[:, :, 0].mean() > 0.3
    print("18.12 PASS: motion expert routing")


if __name__ == "__main__":
    test_18_2_read_motion_vectors()
    test_18_3_temporal_reproject()
    test_18_4_motion_coherence()
    test_18_5_occlusion_mask()
    test_18_7_merge_reprojected()
    test_18_12_motion_expert_routing()
    print("All Section 18 tests passed")
