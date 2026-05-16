## ADDED Requirements

### Requirement: Tile extraction from full-resolution images
The system SHALL extract non-overlapping 512x512 tiles from full-resolution images. The `TileExtractor` class in `src/omen/training/tile.py` SHALL accept an image array and tile_size parameter, returning a list of `Tile` dataclass instances containing the tile data, coordinates, and an edge flag.

#### Scenario: 4K image produces correct tiles
- **WHEN** a 3840x2160 image is tiled with tile_size=512
- **THEN** the system SHALL produce 40 tiles (8 columns x 5 rows)
- **AND** edge tiles at right boundary SHALL be 256 pixels wide
- **AND** edge tiles at bottom boundary SHALL be 112 pixels tall
- **AND** corner tiles SHALL be 256x112 pixels

#### Scenario: Image smaller than tile size
- **WHEN** a 480x270 image is tiled with tile_size=512
- **THEN** the system SHALL produce 1 tile containing the entire image
- **AND** the tile SHALL be marked as an edge tile

#### Scenario: Exact multiple tile size
- **WHEN** a 1024x1024 image is tiled with tile_size=512
- **THEN** the system SHALL produce 4 tiles (2x2), all 512x512
- **AND** no tiles SHALL be marked as edge tiles

### Requirement: Tile-to-full reconstruction
The system SHALL reconstruct a full-resolution image from a list of tile predictions by placing each tile at its original coordinates. Overlapping regions SHALL use the last-written value.

#### Scenario: Reconstruct 4K from 40 tiles
- **WHEN** 40 tile predictions are provided for a 3840x2160 image
- **THEN** the reconstructed image SHALL have shape (3840, 2160, C)
- **AND** tile data SHALL be placed at the correct coordinates
- **AND** edge tile data SHALL fit within the image boundary without padding

### Requirement: Tile coordinate consistency
GT and noisy images from the same render MUST produce tiles with identical coordinates. The `extract_tiles` function SHALL use deterministic grid positions based solely on image shape and tile_size.

#### Scenario: GT and noisy tiles align
- **WHEN** GT (3840, 2160, 3) and noisy (3840, 2160, 3) are tiled with same tile_size
- **THEN** both SHALL produce the same number of tiles
- **AND** corresponding tiles SHALL have identical (y_start, y_end, x_start, x_end) coordinates
