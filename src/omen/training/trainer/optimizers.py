"""Per-component functional optimizer creation for Nabla training."""

import logging

from nabla.nn.optim import adamw_init


logger = logging.getLogger("omen.training.trainer.optimizers")

# Per-component base learning rates
COMPONENT_LRS = {
    "encoder": 5e-6,
    "decoder": 5e-6,
    "shared_expert": 5e-6,
    "material_experts": 5e-6,
    "light_experts": 3e-6,
    "geometry_experts": 4e-6,
    "motion_experts": 5e-6,
    "ar_predictor": 5e-6,
    "episodic_correction": 2e-3,
}

# Per-component state_dict key prefixes (model attribute names)
COMPONENT_PREFIXES = {
    "encoder": ["scene_encoder.", "render_encoder.", "fusion.", "confidence_head."],
    "decoder": ["decoder."],
    "shared_expert": ["moe.shared"],
    "material_experts": ["moe.materials"],
    "light_experts": ["moe.lights"],
    "geometry_experts": ["moe.geometry"],
    "motion_experts": ["moe.motion"],
    "ar_predictor": ["ar_predictor.", "delta_encoder."],
    "episodic_correction": ["episodic."],
}


def _collect_param_names(model, prefix):
    """Return list of param names starting with *prefix*."""
    return [n for n, _ in model.named_parameters() if n.startswith(prefix)]


def create_functional_optimizers(model, config, weight_decay):
    """Create functional AdamW optimizer state for each component group.

    Returns:
        dict mapping component name to ``{param_names, state, lr, weight_decay}``.
    """
    c = config.components
    components = {}
    state_dict = model.state_dict()

    def add_component(name, prefixes):
        names = []
        for pfx in prefixes:
            names.extend(_collect_param_names(model, pfx))
        if not names:
            return
        subset = {n: state_dict[n] for n in names}
        components[name] = {
            "param_names": names,
            "state": adamw_init(subset),
            "lr": COMPONENT_LRS.get(name, 5e-5),
            "weight_decay": weight_decay,
        }

    # Core encoder group
    add_component(
        "encoder",
        COMPONENT_PREFIXES["encoder"],
    )
    add_component("decoder", ["decoder."])

    # MoE components
    if c.moe:
        add_component("shared_expert", ["moe.shared"])
        if c.moe_materials:
            add_component("material_experts", ["moe.materials"])
        if c.moe_lights:
            add_component("light_experts", ["moe.lights"])
        if c.moe_geometry:
            add_component("geometry_experts", ["moe.geometry"])
        if c.moe_motion:
            add_component("motion_experts", ["moe.motion"])

    # ARPredictor
    if c.ar_predictor:
        add_component("ar_predictor", ["ar_predictor.", "delta_encoder."])

    # Episodic correction
    if c.episodic_correction:
        add_component("episodic_correction", ["episodic."])

    logger.info("Created %d functional optimizer components", len(components))
    return components
