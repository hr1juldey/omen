"""Core path tracing logic for Omen integrator."""

import mitsuba as mi
import drjit as dr

from omen_integrator.path import trace_path


def render_path_tracer(scene, sensor, integrator, seed, spp, develop, evaluate):
    """Execute path tracing with JEPA acceleration.

    Args:
        scene: Mitsuba Scene object
        sensor: Mitsuba Sensor object
        integrator: OmenIntegrator instance with parameters
        seed: Random seed for reproducibility
        spp: Samples per pixel override (0 = use sensor's default)
        develop: Whether to develop the film into a TensorXf
        evaluate: Whether to evaluate the rendering task

    Returns:
        TensorXf: Rendered image tensor if develop=True, empty tensor otherwise
    """
    # For initial implementation, use Mitsuba's built-in path tracer
    # This ensures correct rendering while we develop custom logic
    path_integrator = mi.load_dict(
        {
            "type": "path",
            "max_depth": integrator.max_depth if integrator.max_depth > 0 else 8,
            "rr_depth": integrator.rr_depth,
        }
    )

    # Use Mitsuba's render with our integrator
    result = mi.render(
        scene, spp=spp if spp > 0 else 4, seed=seed, integrator=path_integrator
    )

    if develop:
        return result
    return mi.TensorXf()
