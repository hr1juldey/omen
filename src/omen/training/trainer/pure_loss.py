"""Pure loss function for compiled functional training.

Chains functional sub-modules from omen.model.functional.
No model.load_state_dict — takes flat params dict directly.
"""

import nabla.nn.functional as F

from omen.model.functional import (
    _extract_prefix,
    cross_attn_fn,
    decoder_fn,
    render_encoder_fn,
    sigreg_fn,
)
from omen.model.jepa import SIGREG_LAMBDA


def pure_loss_fn(params, noisy, gt, scene_latent, config):
    """Pure functional loss — no Python side effects.

    Args:
        params: flat ``{name: Tensor}`` state dict to differentiate w.r.t.
        noisy: ``(B, H, W, 4)`` noisy RGBA render.
        gt: ``(B, H, W, 4)`` ground-truth RGBA render.
        scene_latent: ``(B, latent_dim)`` pre-encoded scene latent tensor.
        config: OmenConfig (for SIGReg switches).

    Returns:
        scalar total loss tensor.
    """
    p_re = _extract_prefix(params, "render_encoder.")
    p_fu = _extract_prefix(params, "fusion.")
    p_de = _extract_prefix(params, "decoder.")

    # Encode noisy render + fuse -> predicted latent
    render_lat_noisy = render_encoder_fn(p_re, noisy)
    predicted_latent = cross_attn_fn(p_fu, render_lat_noisy, scene_latent)

    # Encode gt render + fuse -> target latent
    render_lat_gt = render_encoder_fn(p_re, gt)
    target_latent = cross_attn_fn(p_fu, render_lat_gt, scene_latent)

    # Decode noise prediction (decoder takes RGB only)
    noisy_rgb = noisy[:, :, :, :3]
    predicted_noise = decoder_fn(p_de, predicted_latent, noisy_rgb)
    gt_residual = gt[:, :, :, :3] - noisy_rgb

    # Loss: latent prediction + noise prediction + SIGReg
    pred_loss = F.mse_loss(predicted_latent, target_latent)
    pred_loss = pred_loss + F.mse_loss(predicted_noise, gt_residual)
    reg_loss = sigreg_fn(predicted_latent, config)

    return pred_loss + SIGREG_LAMBDA * reg_loss
