"""Compiled forward+backward step — @nb.compile decorated.

Only does value_and_grad (no optimizer update).
AdamW update runs outside @nb.compile to avoid CPU scalar device mismatch
in nabla's adamw_step (bias_correction creates CPU scalars from Python floats).
"""

import nabla as nb

from omen.training.trainer.pure_loss import pure_loss_fn


@nb.compile
def compiled_loss_and_grads(params, noisy, gt, scene_latent):
    """Compiled forward + backward only.

    Args:
        params: flat ``{name: Tensor}`` state dict.
        noisy: ``(B, H, W, 4)`` noisy RGBA render.
        gt: ``(B, H, W, 4)`` ground-truth RGBA render.
        scene_latent: ``(B, latent_dim)`` pre-encoded scene latent.

    Returns:
        (loss, grads) — scalar loss + gradient dict.
    """
    return nb.value_and_grad(pure_loss_fn, argnums=0)(
        params, noisy, gt, scene_latent, None
    )
