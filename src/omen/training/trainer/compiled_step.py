"""Compiled train step — @nb.compile decorated, no Python side effects.

Wraps value_and_grad(pure_loss_fn) + per-component adamw_update
inside a single compiled graph entry. Cache hits on same tensor shapes.
"""

import nabla as nb
from nabla.nn.optim import adamw_update

from omen.training.trainer.optimizers import COMPONENT_LRS, COMPONENT_PREFIXES
from omen.training.trainer.pure_loss import pure_loss_fn


@nb.compile
def compiled_train_step(params, noisy, gt, scene_latent, opt_states, weight_decay):
    """Compiled forward + backward + per-component optimizer update.

    Args:
        params: flat ``{name: Tensor}`` state dict.
        noisy: ``(B, H, W, 4)`` noisy RGBA render.
        gt: ``(B, H, W, 4)`` ground-truth RGBA render.
        scene_latent: ``(B, latent_dim)`` pre-encoded scene latent.
        opt_states: ``{name: {"m": pytree, "v": pytree, "step": int}}``.
        weight_decay: float, AdamW weight decay (constant).

    Returns:
        (new_params, new_states, loss) — no side effects.
    """
    loss, grads = nb.value_and_grad(pure_loss_fn, argnums=0)(
        params, noisy, gt, scene_latent, None
    )

    new_params = dict(params)
    new_states = {}

    for name in sorted(COMPONENT_LRS.keys()):
        if name not in opt_states:
            continue

        prefixes = COMPONENT_PREFIXES[name]
        subset_p = {
            k: new_params[k] for k in params if any(k.startswith(p) for p in prefixes)
        }
        subset_g = {
            k: grads[k] for k in grads if any(k.startswith(p) for p in prefixes)
        }

        if not subset_p:
            continue

        updated_p, updated_state = adamw_update(
            subset_p,
            subset_g,
            opt_states[name],
            lr=COMPONENT_LRS[name],
            weight_decay=weight_decay,
        )

        new_params.update(updated_p)
        new_states[name] = updated_state

    return new_params, new_states, loss
