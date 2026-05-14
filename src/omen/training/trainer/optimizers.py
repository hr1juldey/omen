"""Per-component optimizer creation for Nabla multi-optimizer training."""

import logging

from nabla import nn

from omen.config import OmenConfig

logger = logging.getLogger("omen.training.trainer.optimizers")

# Per-component base learning rates
COMPONENT_LRS = {
    "encoder": 5e-5,
    "decoder": 5e-5,
    "shared_expert": 5e-5,
    "material_experts": 5e-5,
    "light_experts": 3e-5,
    "geometry_experts": 4e-5,
    "motion_experts": 5e-5,
    "ar_predictor": 5e-5,
    "episodic_correction": 2e-2,
}


def create_optimizers(model, config: OmenConfig, weight_decay: float):
    """Create separate AdamW optimizer for each component group.

    Returns (optimizers_dict, component_params_dict).
    """
    try:
        import nabla as nb
        NABLA_AVAILABLE = True
    except ImportError:
        NABLA_AVAILABLE = False

    if not NABLA_AVAILABLE:
        raise ImportError("Nabla required for training")

    c = config.components
    optimizers = {}
    component_params = {}

    def collect(prefix: str) -> list:
        return [p for n, p in model.named_parameters() if n.startswith(prefix)]

    def create_opt(name: str, params: list, lr: float):
        if params:
            component_params[name] = params
            optimizers[name] = nn.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Core encoder group
    encoder_params = (
        collect("scene_encoder.") +
        collect("render_encoder.") +
        collect("cross_attn.") +
        collect("confidence_head.")
    )
    create_opt("encoder", encoder_params, COMPONENT_LRS["encoder"])
    create_opt("decoder", collect("decoder."), COMPONENT_LRS["decoder"])

    # MoE components
    if c.moe:
        create_opt("shared_expert", collect("moe.shared"), COMPONENT_LRS["shared_expert"])
        if c.moe_materials:
            create_opt("material_experts", collect("moe.materials"), COMPONENT_LRS["material_experts"])
        if c.moe_lights:
            create_opt("light_experts", collect("moe.lights"), COMPONENT_LRS["light_experts"])
        if c.moe_geometry:
            create_opt("geometry_experts", collect("moe.geometry"), COMPONENT_LRS["geometry_experts"])
        if c.moe_motion:
            create_opt("motion_experts", collect("moe.motion"), COMPONENT_LRS["motion_experts"])

    # ARPredictor
    if c.ar_predictor:
        ar_params = collect("ar_predictor.") + collect("delta_encoder.")
        create_opt("ar_predictor", ar_params, COMPONENT_LRS["ar_predictor"])

    # Episodic correction
    if c.episodic_correction:
        create_opt("episodic_correction", collect("episodic."), COMPONENT_LRS["episodic_correction"])

    return optimizers, component_params
