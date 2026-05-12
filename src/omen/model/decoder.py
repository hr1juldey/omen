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

    Uses nb.conv2d_transpose() functional API (NHWC format).
    Filter layout for conv2d_transpose: (K_h, K_w, C_out, C_in).
    Architecture:
    Linear(latent_dim, 128 * h_base * w_base)
    Reshape -> (batch, h_base, w_base, 128)  [NHWC]
    conv2d_transpose(128, 64, 3, stride=2) -> SiLU
    conv2d_transpose(64, 32, 3, stride=2) -> SiLU
    conv2d_transpose(32, 4, 3, stride=2) -> Sigmoid
    """

    def __init__(self, latent_dim: int = LATENT_DIM, base_size: int = 8):
        super().__init__()
        self.base_size = base_size
        self.latent_dim = latent_dim

        # Project latent to spatial feature map
        self.proj = nn.Linear(latent_dim, 128 * base_size * base_size)

        # Transposed conv filters: layout is (K_h, K_w, C_out, C_in)
        self.deconv1_filter = F.he_normal((3, 3, 64, 128))
        self.deconv1_filter.requires_grad = True
        self.deconv1_bias = nb.zeros(64)
        self.deconv1_bias.requires_grad = True
        self.deconv2_filter = F.he_normal((3, 3, 32, 64))
        self.deconv2_filter.requires_grad = True
        self.deconv2_bias = nb.zeros(32)
        self.deconv2_bias.requires_grad = True
        self.deconv3_filter = F.he_normal((3, 3, 4, 32))
        self.deconv3_filter.requires_grad = True
        self.deconv3_bias = nb.zeros(4)
        self.deconv3_bias.requires_grad = True

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

        # Project to spatial features and reshape to NHWC
        x = self.proj(latent)
        x = x.reshape(batch, self.base_size, self.base_size, 128)

        # Upsample via transposed convolutions (NHWC throughout)
        x = nb.silu(nb.conv2d_transpose(x, self.deconv1_filter, stride=2, padding=1, bias=self.deconv1_bias))
        x = nb.silu(nb.conv2d_transpose(x, self.deconv2_filter, stride=2, padding=1, bias=self.deconv2_bias))
        x = nb.sigmoid(nb.conv2d_transpose(x, self.deconv3_filter, stride=2, padding=1, bias=self.deconv3_bias))

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
