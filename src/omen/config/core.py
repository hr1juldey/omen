"""OmenConfig core with serialization, validation, and factory methods."""

from dataclasses import dataclass, field

from omen.config.switches import ComponentSwitches, ModeSwitches, TrainingSwitches


@dataclass
class OmenConfig:
    """Serializable configuration controlling all component switches.

    Changing a switch at runtime does not require model re-initialization.
    All parameters exist regardless of switch state.
    Disabled components contribute zero to forward pass and receive zero gradients.
    """

    components: ComponentSwitches = field(default_factory=ComponentSwitches)
    training: TrainingSwitches = field(default_factory=TrainingSwitches)
    modes: ModeSwitches = field(default_factory=ModeSwitches)
    tier: str = "fast"

    @staticmethod
    def v1_dense():
        """V1: Dense denoiser, no MoE, no AR, no SIGReg."""
        return OmenConfig()

    @staticmethod
    def v1_moe():
        """After V1 validated: unlock MoE with scene-graph routing."""
        cfg = OmenConfig()
        cfg.components.moe = True
        cfg.components.scene_graph_routing = True
        return cfg

    @staticmethod
    def v1_animation():
        """After MoE validated: unlock temporal prediction."""
        cfg = OmenConfig.v1_moe()
        cfg.components.ar_predictor = True
        cfg.components.scene_delta_encoder = True
        cfg.modes.temporal = True
        return cfg

    @staticmethod
    def full():
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

    def to_dict(self) -> dict:
        """Serialize config to dictionary for checkpoint saving."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """Deserialize config from dictionary for checkpoint loading."""
        components = ComponentSwitches(**data.get("components", {}))
        training = TrainingSwitches(**data.get("training", {}))
        modes = ModeSwitches(**data.get("modes", {}))
        return cls(
            components=components,
            training=training,
            modes=modes,
            tier=data.get("tier", "fast"),
        )

    def validate(self) -> list[str]:
        """Validate config constraints. Returns list of error messages."""
        errors = []
        c = self.components

        if c.sigreg and c.simple_var_reg:
            errors.append("Cannot enable both sigreg and simple_var_reg")

        if c.ar_predictor and not c.scene_delta_encoder:
            errors.append("ARPredictor requires scene_delta_encoder")

        if not c.moe:
            if any([c.moe_materials, c.moe_lights, c.moe_geometry, c.moe_motion]):
                errors.append("MoE sub-switches require moe=True")

        if c.scene_graph_routing and not c.moe:
            errors.append("Scene-graph routing requires moe=True")

        if self.training.stratified_replay and self.training.replay_ratio <= 0:
            errors.append("replay_ratio must be > 0")
        if self.training.stratified_replay and self.training.replay_ratio >= 1:
            errors.append("replay_ratio must be < 1")

        if self.tier not in ("fast", "medium", "beast"):
            errors.append(f"Invalid tier: {self.tier}")

        return errors
