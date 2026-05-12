"""Decoder and ConfidenceHead modules.

Decoder: latent -> RGBA image via Conv2dTranspose stack (~1M params)
ConfidenceHead: latent -> per-pixel confidence map via MLP + Sigmoid
"""

import logging

try:
    import nabla as nb
    from nabla import nn
    import nabla.nn.functional as F
    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False

logger = logging.getLogger("omen.model.decoder")

LATENT_DIM = 192


class Decoder(nn.Module):
    """Decode latent vector to RGBA image.

    Architecture:
    Linear(latent_dim, 128 * h_base * w_base)
    Reshape -> (128, h_base, w_base)
    Conv2dTranspose(128, 64, 3, stride=2) -> SiLU
    Conv2dTranspose(64, 32, 3, stride=2) -> SiLU
    Conv2dTranspose(32, 4, 3, stride=2) -> Sigmoid
    """

    def __init__(self, latent_dim: int = LATENT_DIM, base_size: int = 8):
        super().__init__()
        self.base_size = base_size
        self.latent_dim = latent_dim

        # Project latent to spatial feature map
        self.proj = nn.Linear(latent_dim, 128 * base_size * base_size)

        # Upsampling via transposed convolutions
        self.deconv1 = nn.Conv2dTranspose(128, 64, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.Conv2dTranspose(64, 32, kernel_size=3, stride=2, padding=1)
        self.deconv3 = nn.Conv2dTranspose(32, 4, kernel_size=3, stride=2, padding=1)

    def forward(self, latent, height, width):
        """Decode latent to RGBA image.

        Args:
            latent: (batch, latent_dim)
            height: target image height
            width: target image width

        Returns:
            rgba: (batch, height, width, 4) in [0, 1]
        """
        batch = latent.shape[0]

        # Project to spatial features
        x = self.proj(latent)
        x = x.reshape(batch, 128, self.base_size, self.base_size)

        # Upsample
        x = F.silu(self.deconv1(x))
        x = F.silu(self.deconv2(x))
        x = F.sigmoid(self.deconv3(x))

        # Convert NCHW -> NHWC
        x = x.transpose(0, 2, 3, 1)

        # Resize to target if needed
        if x.shape[1] != height or x.shape[2] != width:
            x = F.interpolate(x, size=(height, width))

        return x


class ConfidenceHead(nn.Module):
    """Predict per-pixel confidence from latent.

    Architecture:
    Linear(192, 96) -> SiLU -> Linear(96, 48) -> SiLU -> Linear(48, 1) -> Sigmoid

    Output: confidence map in [0, 1]
    High confidence = JEPA prediction reliable
    Low confidence = needs more path tracing
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
            nn.Sigmoid(),
        )

    def forward(self, latent, height, width):
        """Predict per-pixel confidence map.

        Args:
            latent: (batch, latent_dim)
            height: image height
            width: image width

        Returns:
            confidence: (batch, height, width, 1) in [0, 1]
        """
        # Apply MLP to get scalar confidence
        conf = self.net(latent)  # (batch, 1)

        # Broadcast to spatial map
        conf = conf.expand(conf.shape[0], height * width)
        conf = conf.reshape(conf.shape[0], height, width, 1)

        return conf
