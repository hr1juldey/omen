"""Tile extraction and reconstruction for VRAM-safe training.

Splits full-resolution images into non-overlapping tiles for
GPU training, then reconstructs full images from tile predictions.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Tile:
    """A single image tile with its position metadata."""

    data: np.ndarray  # (tile_h, tile_w, C)
    coords: tuple[int, int, int, int]  # (y_start, y_end, x_start, x_end)
    is_edge: bool


def extract_tiles(
    image: np.ndarray, tile_size: int = 512
) -> list[Tile]:
    """Extract non-overlapping tiles from an image.

    Edge tiles at image boundaries may be smaller than tile_size.

    Args:
        image: (H, W, C) or (H, W) image array.
        tile_size: Target tile dimension in pixels.

    Returns:
        List of Tile objects with position metadata.
    """
    h, w = image.shape[:2]
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile_data = image[y:y_end, x:x_end]
            is_edge = (y_end - y < tile_size) or (x_end - x < tile_size)
            tiles.append(Tile(tile_data, (y, y_end, x, x_end), is_edge))
    return tiles


def tile_to_full(
    tiles: list[Tile], original_shape: tuple[int, ...]
) -> np.ndarray:
    """Reconstruct full image from tile predictions.

    Args:
        tiles: List of Tile objects with data and coordinates.
        original_shape: Target shape (H, W, ...) for the output.

    Returns:
        Reconstructed image array.
    """
    out = np.zeros(original_shape, dtype=np.float32)
    for tile in tiles:
        y0, y1, x0, x1 = tile.coords
        out[y0:y1, x0:x1] = tile.data
    return out
