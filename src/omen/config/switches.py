"""Component, training, and mode switch dataclasses."""

from dataclasses import dataclass, field


@dataclass
class ComponentSwitches:
    """Toggle individual model components ON/OFF.

    All components are initialized regardless of switch state.
    Disabled components use identity passthrough in forward pass.
    """

    # Always-on core (V1)
    scene_encoder: bool = True
    render_encoder: bool = True
    cross_attention: bool = True
    decoder: bool = True
    confidence_head: bool = True

    # MoE system
    moe: bool = False
    moe_materials: bool = False  # Only relevant when moe=True
    moe_lights: bool = False
    moe_geometry: bool = False
    moe_motion: bool = False
    scene_graph_routing: bool = False

    # Temporal
    ar_predictor: bool = False
    scene_delta_encoder: bool = False

    # Regularization
    sigreg: bool = False
    simple_var_reg: bool = True

    # Adaptation
    episodic_correction: bool = True
    lora: bool = False

    # Other
    mla_skip: bool = False


@dataclass
class TrainingSwitches:
    """Training-specific switches."""

    per_component_lr: bool = True
    surprise_lr_modulation: bool = True
    stratified_replay: bool = True
    replay_size: int = 500
    replay_ratio: float = 0.5
    surprise_lr_scale: float = 2.0


@dataclass
class ModeSwitches:
    """Mode pipeline switches."""

    denoiser: bool = True
    adaptive: bool = False
    multires: bool = False
    temporal: bool = False
