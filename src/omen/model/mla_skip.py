"""MLA-inspired skip connection compression (from DeepSeek-V2/V3).

Compresses U-Net skip connections 16× via low-rank projections:
  down: Linear(C, C//16)  — encode skip features
  up:   Linear(C//16, C)  — reconstruct before decoder concat

Reduces 4K skip memory from ~6GB to ~375MB.
Projections W_down and W_up are learnable (end-to-end trained).
"""

import logging

try:
    import nabla as nb
    from nabla import nn

    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False

logger = logging.getLogger("omen.model.mla_skip")

COMPRESS_RATIO = 16


class MLASkipCompress(nn.Module):
    """Compress skip features: (B, H, W, C) -> (B, H, W, C//16)."""

    def __init__(self, channels: int, ratio: int = COMPRESS_RATIO):
        super().__init__()
        self.latent_dim = max(channels // ratio, 4)
        self.down = nn.Linear(channels, self.latent_dim)

    def forward(self, skip_features):
        """Compress skip connection features.

        Args:
            skip_features: (B, H, W, C) encoder features

        Returns:
            compressed: (B, H, W, C//16) compressed skip latent
        """
        return nb.silu(self.down(skip_features))


class MLASkipReconstruct(nn.Module):
    """Reconstruct skip features: (B, H, W, C//16) -> (B, H, W, C)."""

    def __init__(self, channels: int, ratio: int = COMPRESS_RATIO):
        super().__init__()
        self.latent_dim = max(channels // ratio, 4)
        self.up = nn.Linear(self.latent_dim, channels)

    def forward(self, compressed):
        """Reconstruct skip features from compressed latent.

        Args:
            compressed: (B, H, W, C//16) compressed skip latent

        Returns:
            reconstructed: (B, H, W, C) reconstructed features
        """
        return self.up(compressed)


class MLASkipPair(nn.Module):
    """Paired compress/reconstruct for one U-Net skip level.

    Usage in U-Net:
        encoder: skip_latent = mla_pair.compress(features)
        decoder: features = mla_pair.reconstruct(skip_latent)
    """

    def __init__(self, channels: int, ratio: int = COMPRESS_RATIO):
        super().__init__()
        self.compress = MLASkipCompress(channels, ratio)
        self.reconstruct = MLASkipReconstruct(channels, ratio)
        self.channels = channels
        self.latent_dim = max(channels // ratio, 4)

    def forward_compress(self, features):
        """Compress encoder features for skip storage."""
        return self.compress(features)

    def forward_reconstruct(self, compressed):
        """Reconstruct skip features for decoder concatenation."""
        return self.reconstruct(compressed)


def memory_savings(
    height: int, width: int, channels: int, ratio: int = COMPRESS_RATIO
) -> dict:
    """Calculate memory savings from MLA compression.

    Args:
        height: feature map height
        width: feature map width
        channels: feature channels
        ratio: compression ratio

    Returns:
        dict with original_mb, compressed_mb, savings_pct
    """
    elements = height * width * channels
    original_mb = elements * 4 / (1024 * 1024)  # float32
    compressed_mb = height * width * (channels // ratio) * 4 / (1024 * 1024)
    savings_pct = (1 - compressed_mb / original_mb) * 100 if original_mb > 0 else 0
    return {
        "original_mb": round(original_mb, 1),
        "compressed_mb": round(compressed_mb, 1),
        "savings_pct": round(savings_pct, 1),
    }
