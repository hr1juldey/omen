"""Quality metrics for denoiser validation.

Computes SSIM, PSNR, and basic artifact detection
for comparing denoised output against reference renders.
"""

import logging

import numpy as np

logger = logging.getLogger("omen.modes.quality")


def psnr(clean: np.ndarray, reference: np.ndarray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        clean: (H, W, C) denoised image
        reference: (H, W, C) ground truth image
        max_val: maximum pixel value (1.0 for float, 255 for uint8)

    Returns:
        PSNR in dB (higher is better, inf if identical)
    """
    mse = np.mean((clean - reference) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(max_val**2 / mse))


def _uniform_filter_2d(img: np.ndarray, size: int) -> np.ndarray:
    """Fast 2D uniform mean filter via cumulative sums."""
    pad = size // 2
    padded = np.pad(img.astype(np.float64), pad, mode="reflect")
    cum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    h, w = img.shape
    s = size
    result = (
        cum[s : s + h, s : s + w]
        - cum[:h, s : s + w]
        - cum[s : s + h, :w]
        + cum[:h, :w]
    ) / (s * s)
    return result.astype(np.float32)


def ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 7,
) -> float:
    """Compute mean Structural Similarity Index (SSIM).

    Args:
        img1: (H, W) or (H, W, C) image
        img2: same shape as img1
        window_size: sliding window size (default 7)

    Returns:
        Mean SSIM (1.0 = identical, 0.0 = unrelated)
    """
    c1 = 0.01**2
    c2 = 0.03**2

    if img1.ndim == 3:
        scores = [
            _ssim_channel(img1[:, :, c], img2[:, :, c], window_size, c1, c2)
            for c in range(img1.shape[2])
        ]
        return float(np.mean(scores))

    return _ssim_channel(img1, img2, window_size, c1, c2)


def _ssim_channel(a, b, ws, c1, c2):
    """SSIM for a single channel using uniform window."""
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    mu_a = _uniform_filter_2d(a_f, ws)
    mu_b = _uniform_filter_2d(b_f, ws)
    sig_a = _uniform_filter_2d(a_f**2, ws) - mu_a**2
    sig_b = _uniform_filter_2d(b_f**2, ws) - mu_b**2
    sig_ab = _uniform_filter_2d(a_f * b_f, ws) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2)
    return float(np.mean(num / den))


def detect_artifacts(image: np.ndarray, threshold: float = 0.5) -> dict:
    """Detect common denoising artifacts: NaN/Inf, clamping, ringing.

    Args:
        image: (H, W, C) float32 image
        threshold: clamp detection threshold

    Returns:
        dict with artifact counts and severity
    """
    nan_count = int(np.sum(np.isnan(image)))
    inf_count = int(np.sum(np.isinf(image)))
    clamped = int(np.sum((image < 0) | (image > threshold)))
    return {
        "nan_pixels": nan_count,
        "inf_pixels": inf_count,
        "clamped_pixels": clamped,
        "has_artifacts": (nan_count + inf_count + clamped) > 0,
    }
