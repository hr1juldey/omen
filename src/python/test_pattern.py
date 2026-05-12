"""Test pattern generation for Omen render engine validation."""

from typing import List


def generate_gradient(width: int, height: int) -> List[List[float]]:
    """Generate horizontal red-to-blue gradient.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Flat list of [r, g, b, a] values for each pixel (row-major order)
    """
    pixels = []
    for y in range(height):
        for x in range(width):
            t = x / max(1, width - 1)
            r = 1.0 - t
            g = 0.0
            b = t
            a = 1.0
            pixels.append([r, g, b, a])
    return pixels
