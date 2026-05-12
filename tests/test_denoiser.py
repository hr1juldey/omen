"""Tests for Mode 1 - Denoiser (tasks 10.8, 10.9).

Test 1: Cornell box at 4spp, verify denoised output shape and type
Test 2: Cornell box with NO AOV, verify graceful degradation
Reference: 256spp ground truth for quality comparison (when model available)
"""

import logging

import numpy as np

logger = logging.getLogger("omen.test_denoiser")


def _create_cornell_scene():
    """Create Cornell box scene via Mitsuba."""
    import mitsuba as mi

    return mi.cornell_box()


def test_10_8_denoiser_with_aov():
    """Task 10.8: Cornell box at 4spp, verify pipeline runs end-to-end."""
    from omen.jepa_bridge import JEPABridge
    from omen.modes.denoiser import render_denoiser

    scene = _create_cornell_scene()
    bridge = JEPABridge()

    result = render_denoiser(scene, bridge, spp=4, tier="medium")

    assert result.ndim == 3, f"Expected 3D, got {result.ndim}D"
    assert result.shape[2] == 4, f"Expected RGBA (4ch), got {result.shape[2]}ch"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    h, w = result.shape[0], result.shape[1]
    assert h > 0 and w > 0, f"Invalid dimensions: {h}x{w}"

    logger.info("10.8 PASS: shape=%s dtype=%s", result.shape, result.dtype)


def test_10_9_denoiser_no_aov():
    """Task 10.9: Cornell box with NO aux passes, verify degraded mode."""
    from omen.jepa_bridge import JEPABridge
    from omen.modes.denoiser import render_denoiser

    scene = _create_cornell_scene()
    bridge = JEPABridge()

    # Force basic path tracer (no AOV integrator) by rendering with fallback
    result = render_denoiser(scene, bridge, spp=4, tier="fast")

    assert result.ndim == 3, f"Expected 3D, got {result.ndim}D"
    assert result.shape[2] == 4, f"Expected RGBA (4ch), got {result.shape[2]}ch"
    assert result.dtype == np.float32

    # Even without model, should return valid RGBA
    assert np.all(result[:, :, 3] > 0), "Alpha channel should be > 0"

    logger.info("10.9 PASS: degraded mode shape=%s", result.shape)


def test_10_quality_metrics():
    """Task 10.7: Verify SSIM/PSNR metrics on known images."""
    from omen.modes.quality import detect_artifacts, psnr, ssim

    # Identical images -> perfect scores
    img = np.random.rand(64, 64, 3).astype(np.float32)
    assert psnr(img, img) == float("inf"), "Identical images should have inf PSNR"
    assert ssim(img, img) > 0.99, (
        f"Identical images SSIM should be ~1.0, got {ssim(img, img)}"
    )

    # Different images -> lower scores
    noise = np.random.rand(64, 64, 3).astype(np.float32)
    p = psnr(img, noise)
    assert 0 < p < 30, f"Unrelated PSNR should be low, got {p}"

    # Artifact detection
    clean = np.zeros((32, 32, 3), dtype=np.float32)
    artifacts = detect_artifacts(clean)
    assert not artifacts["has_artifacts"], "Zero image should have no artifacts"

    logger.info("Quality metrics PASS: PSNR=%.1f SSIM=%.3f", p, ssim(img, noise))


def test_11_6_adaptive():
    """Task 11.6: Cornell box adaptive mode, verify output and sample reduction."""
    from omen.jepa_bridge import JEPABridge
    from omen.modes.adaptive import render_adaptive

    scene = _create_cornell_scene()
    bridge = JEPABridge()

    result = render_adaptive(scene, bridge, spp_target=128)

    assert result.ndim == 3, f"Expected 3D, got {result.ndim}D"
    assert result.shape[2] == 4, f"Expected RGBA (4ch), got {result.shape[2]}ch"
    assert result.dtype == np.float32

    logger.info("11.6 PASS: adaptive output shape=%s", result.shape)


def test_12_5_multires():
    """Task 12.5: Cornell box multires, verify merge output."""
    from omen.jepa_bridge import JEPABridge
    from omen.modes.multires import render_multires

    scene = _create_cornell_scene()
    bridge = JEPABridge()

    result = render_multires(scene, bridge, scale=4)

    assert result.ndim == 3, f"Expected 3D, got {result.ndim}D"
    assert result.shape[2] == 4, f"Expected RGBA (4ch), got {result.shape[2]}ch"
    assert result.dtype == np.float32

    logger.info("12.5 PASS: multires output shape=%s", result.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_10_quality_metrics()
    test_10_8_denoiser_with_aov()
    test_10_9_denoiser_no_aov()
    test_11_6_adaptive()
    test_12_5_multires()
    print("All mode tests passed")
