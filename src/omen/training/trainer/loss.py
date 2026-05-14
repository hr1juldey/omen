"""Pure loss function for functional value_and_grad differentiation."""


def compute_training_loss(params, model, noisy, gt, scene_graph, config):
    """Pure loss function compatible with ``nb.value_and_grad``.

    Args:
        params: Flat ``{name: Tensor}`` state dict to differentiate w.r.t.
        model: OmenJEPA instance (used for encode/decode/loss).
        noisy: Noisy render input ``(B, H, W, C)``.
        gt: Ground-truth render ``(B, H, W, C)``.
        scene_graph: Scene metadata tensor.
        config: OmenConfig instance.

    Returns:
        Scalar total loss tensor.
    """
    model.load_state_dict(params)

    predicted_latent, _ = model.encode(scene_graph, noisy)
    target_latent, _ = model.encode(scene_graph, gt)

    noisy_rgb = noisy[:, :, :, :3]
    predicted_noise = model.decode(predicted_latent, noisy_rgb)
    gt_residual = gt[:, :, :, :3] - noisy_rgb

    total_loss, _, _ = model.compute_loss(
        predicted_latent,
        target_latent,
        config=config,
        predicted_noise=predicted_noise,
        gt_residual=gt_residual,
    )
    return total_loss
