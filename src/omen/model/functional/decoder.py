"""Functional U-Net decoder — JAX-style forward pass.

Mirrors Decoder.forward but takes params dict directly.
"""

import nabla as nb

from omen.kernels.activations import sigmoid_gpu, silu_gpu
from omen.kernels.conv2d import conv2d_safe


def _linear(x, weight, bias):
    return x @ weight + bias


def _pixel_shuffle(x, r=2):
    """(B,H,W,C*r*r) -> (B,H*r,W*r,C)."""
    B, H, W, C = (int(d) for d in x.shape)
    y = nb.reshape(x, (B, H, r, W, r, C // (r * r)))
    y = nb.permute(y, (0, 2, 1, 4, 3, 5))
    return nb.reshape(y, (B, H * r, W * r, C // (r * r)))


def _mla_compress(features, down_w, down_b):
    return silu_gpu(features @ down_w + down_b)


def _mla_reconstruct(compressed, up_w, up_b):
    return compressed @ up_w + up_b


def decoder_fn(p, latent, noisy_image):
    """U-Net residual noise prediction using params dict.

    Args:
        p: prefix-stripped params with e1-e4, lat_gate/proj, mla1/2, up4/3/2, d4-d1.
        latent: (batch, latent_dim) JEPA representation.
        noisy_image: (batch, H, W, 3) noisy render.

    Returns:
        (batch, H, W, 3) predicted noise map.
    """
    # Encoder path
    s1 = silu_gpu(conv2d_safe(noisy_image, p["e1"], padding=(1, 1)))
    s2 = silu_gpu(conv2d_safe(s1, p["e2"], stride=(2, 2), padding=(1, 1)))
    s3 = silu_gpu(conv2d_safe(s2, p["e3"], stride=(2, 2), padding=(1, 1)))
    e4 = silu_gpu(conv2d_safe(s3, p["e4"], stride=(2, 2), padding=(1, 1)))

    # Bottleneck: gated JEPA latent injection
    gate = sigmoid_gpu(_linear(latent, p["lat_gate.weight"], p["lat_gate.bias"]))
    l_feat = gate * _linear(latent, p["lat_proj.weight"], p["lat_proj.bias"])
    bn = e4 * nb.reshape(l_feat, (int(latent.shape[0]), 1, 1, 256))

    # MLA compress high-res skips
    c1 = _mla_compress(s1, p["mla1.compress.down.weight"], p["mla1.compress.down.bias"])
    c2 = _mla_compress(s2, p["mla2.compress.down.weight"], p["mla2.compress.down.bias"])

    # Decoder: pixel shuffle up + skip concat + conv
    d4 = _pixel_shuffle(_linear(bn, p["up4.weight"], p["up4.bias"]))
    d4 = silu_gpu(
        conv2d_safe(nb.concatenate([d4, s3], axis=-1), p["d4"], padding=(1, 1))
    )

    d3 = _pixel_shuffle(_linear(d4, p["up3.weight"], p["up3.bias"]))
    r2 = _mla_reconstruct(
        c2, p["mla2.reconstruct.up.weight"], p["mla2.reconstruct.up.bias"]
    )
    d3 = silu_gpu(
        conv2d_safe(nb.concatenate([d3, r2], axis=-1), p["d3"], padding=(1, 1))
    )

    d2 = _pixel_shuffle(_linear(d3, p["up2.weight"], p["up2.bias"]))
    r1 = _mla_reconstruct(
        c1, p["mla1.reconstruct.up.weight"], p["mla1.reconstruct.up.bias"]
    )
    d2 = silu_gpu(
        conv2d_safe(nb.concatenate([d2, r1], axis=-1), p["d2"], padding=(1, 1))
    )

    out = conv2d_safe(d2, p["d1"], padding=(1, 1))

    H, W = int(noisy_image.shape[1]), int(noisy_image.shape[2])
    if int(out.shape[1]) != H or int(out.shape[2]) != W:
        import nabla.nn.functional as F

        out = F.interpolate(out, size=(H, W))

    return out
