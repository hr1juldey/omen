"""U-Net residual noise predictor + ConfidenceHead.

Decoder: (jepa_latent, noisy_image) -> noise/residual map
  clean = noisy - predicted_noise
ConfidenceHead: latent -> per-pixel confidence map
"""

import logging

try:
    import nabla as nb
    from nabla import nn
    import nabla.nn.functional as F
    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False

from omen.kernels.conv2d import conv2d_safe
from omen.model.mla_skip import MLASkipPair

logger = logging.getLogger("omen.model.decoder")

LATENT_DIM = 1024


def _pixel_shuffle(x, r=2):
    """Pixel shuffle: (B,H,W,C*r*r) -> (B,H*r,W*r,C)."""
    B, H, W, C = (int(d) for d in x.shape)
    y = nb.reshape(x, (B, H, r, W, r, C // (r * r)))
    y = nb.permute(y, (0, 2, 1, 4, 3, 5))
    return nb.reshape(y, (B, H * r, W * r, C // (r * r)))


class Decoder(nn.Module):
    """U-Net residual noise predictor.

    Encoder: 4-stage strided conv (3->64->128->256->256)
    Bottleneck: JEPA latent injected via gated projection
    Decoder: 3-stage Pixel Shuffle upsample + skip connections
    Skip compression: MLA for high-res stages 1-2
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder filters (HWIO layout)
        self.e1 = F.he_normal((3, 3, 3, 64))
        self.e1.requires_grad = True
        self.e2 = F.he_normal((3, 3, 64, 128))
        self.e2.requires_grad = True
        self.e3 = F.he_normal((3, 3, 128, 256))
        self.e3.requires_grad = True
        self.e4 = F.he_normal((3, 3, 256, 256))
        self.e4.requires_grad = True

        # JEPA latent injection at bottleneck
        self.lat_proj = nn.Linear(latent_dim, 256)
        self.lat_gate = nn.Linear(latent_dim, 256)

        # MLA skip compression for high-res levels
        self.mla1 = MLASkipPair(64)
        self.mla2 = MLASkipPair(128)

        # Upsample projections (Linear -> pixel shuffle 2x)
        self.up4 = nn.Linear(256, 256 * 4)
        self.up3 = nn.Linear(256, 128 * 4)
        self.up2 = nn.Linear(128, 64 * 4)

        # Decoder filters (after skip concat)
        self.d4 = F.he_normal((3, 3, 512, 256))
        self.d4.requires_grad = True
        self.d3 = F.he_normal((3, 3, 256, 128))
        self.d3.requires_grad = True
        self.d2 = F.he_normal((3, 3, 128, 64))
        self.d2.requires_grad = True
        self.d1 = F.he_normal((3, 3, 64, 3))
        self.d1.requires_grad = True

    def forward(self, latent, noisy_image):
        """Predict noise/residual.

        Args:
            latent: (B, latent_dim) JEPA representation
            noisy_image: (B, H, W, 3) noisy render

        Returns:
            residual: (B, H, W, 3) predicted noise map
        """
        # Encoder path
        s1 = nb.silu(conv2d_safe(noisy_image, self.e1, padding=(1, 1)))
        s2 = nb.silu(conv2d_safe(s1, self.e2, stride=(2, 2), padding=(1, 1)))
        s3 = nb.silu(conv2d_safe(s2, self.e3, stride=(2, 2), padding=(1, 1)))
        e4 = nb.silu(conv2d_safe(s3, self.e4, stride=(2, 2), padding=(1, 1)))

        # Bottleneck: gated JEPA latent injection
        gate = nb.sigmoid(self.lat_gate(latent))
        l_feat = gate * self.lat_proj(latent)
        bn = e4 * nb.reshape(l_feat, (int(latent.shape[0]), 1, 1, 256))

        # MLA compress high-res skips (save memory)
        c1 = self.mla1.forward_compress(s1)
        c2 = self.mla2.forward_compress(s2)

        # Decoder: pixel shuffle up + skip concat + conv
        d4 = _pixel_shuffle(self.up4(bn))
        d4 = nb.silu(conv2d_safe(
            nb.concatenate([d4, s3], axis=-1), self.d4, padding=(1, 1)))

        d3 = _pixel_shuffle(self.up3(d4))
        r2 = self.mla2.forward_reconstruct(c2)
        d3 = nb.silu(conv2d_safe(
            nb.concatenate([d3, r2], axis=-1), self.d3, padding=(1, 1)))

        d2 = _pixel_shuffle(self.up2(d3))
        r1 = self.mla1.forward_reconstruct(c1)
        d2 = nb.silu(conv2d_safe(
            nb.concatenate([d2, r1], axis=-1), self.d2, padding=(1, 1)))

        # Output: 3-channel residual (no activation — noise can be +/-)
        out = conv2d_safe(d2, self.d1, padding=(1, 1))

        # Handle non-divisible resolutions
        H, W = int(noisy_image.shape[1]), int(noisy_image.shape[2])
        if int(out.shape[1]) != H or int(out.shape[2]) != W:
            out = F.interpolate(out, size=(H, W))

        return out


class ConfidenceHead(nn.Module):
    """Predict per-pixel confidence from latent."""

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
        """Predict per-pixel confidence map."""
        conf = self.net(latent)
        conf = conf.expand(conf.shape[0], height * width)
        return nb.reshape(conf, (int(conf.shape[0]), height, width, 1))
