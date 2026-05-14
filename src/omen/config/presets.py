"""Preset configurations for different training stages."""

from omen.config.core import OmenConfig


def v1_dense() -> OmenConfig:
    """V1: Dense denoiser, no MoE, no AR, no SIGReg."""
    return OmenConfig()


def v1_moe() -> OmenConfig:
    """After V1 validated: unlock MoE with scene-graph routing."""
    cfg = OmenConfig()
    cfg.components.moe = True
    cfg.components.scene_graph_routing = True
    return cfg


def v1_animation() -> OmenConfig:
    """After MoE validated: unlock temporal prediction."""
    cfg = v1_moe()
    cfg.components.ar_predictor = True
    cfg.components.scene_delta_encoder = True
    cfg.modes.temporal = True
    return cfg


def full() -> OmenConfig:
    """Everything enabled."""
    cfg = OmenConfig()
    cfg.components.moe = True
    cfg.components.scene_graph_routing = True
    cfg.components.ar_predictor = True
    cfg.components.scene_delta_encoder = True
    cfg.components.sigreg = True
    cfg.components.simple_var_reg = False
    cfg.modes.adaptive = True
    cfg.modes.multires = True
    cfg.modes.temporal = True
    return cfg
