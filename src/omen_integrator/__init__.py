"""Omen integrator - JEPA-accelerated path tracing for Mitsuba 3."""

import mitsuba as mi


class OmenIntegrator:
    """Path tracing integrator with JEPA scene analysis acceleration."""

    def __init__(self, props=mi.Properties()):
        """Initialize Omen integrator with parameters.

        Args:
            props: Mitsuba Properties object containing plugin parameters
        """
        self.max_depth = props.get("max_depth", -1)
        self.rr_depth = props.get("rr_depth", 5)
        self.jepa_model_path = props.get("jepa_model", "")
        self.use_gpu = props.get("use_gpu", True)

    def render(self, scene, sensor, seed=0, spp=0, develop=True, evaluate=True):
        """Render scene with JEPA-accelerated path tracing.

        Args:
            scene: Mitsuba Scene object
            sensor: Mitsuba Sensor object
            seed: Random seed for reproducibility
            spp: Samples per pixel override
            develop: Whether to develop the film into a TensorXf
            evaluate: Whether to evaluate the rendering task

        Returns:
            TensorXf: Rendered image tensor if develop=True, empty tensor otherwise
        """
        from omen_integrator.core import render_path_tracer

        return render_path_tracer(scene, sensor, self, seed, spp, develop, evaluate)


def register() -> None:
    """Register Omen integrator with Mitsuba plugin system."""
    mi.register_integrator("omen", lambda props: OmenIntegrator(props))
