"""Core path tracing logic for Omen integrator.

Handles the Mitsuba render → AOV extraction → JEPA denoise pipeline.
Uses DLPack zero-copy transfer between Dr.Jit tensors and Nabla.
"""

import logging

import mitsuba as mi
import drjit as dr

from omen_integrator.path import trace_path

logger = logging.getLogger("omen_integrator.core")


def render_path_tracer(scene, sensor, integrator, seed, spp, develop, evaluate):
    """Execute path tracing with JEPA acceleration.

    Pipeline:
        1. Render noisy image via Mitsuba path tracer at low spp
        2. Extract AOV auxiliary buffers (albedo, normal, depth)
        3. Pass through JEPA denoiser via DLPack zero-copy
        4. Return clean result

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
    effective_spp = spp if spp > 0 else 4

    # Render noisy image via Mitsuba's built-in path tracer
    path_integrator = mi.load_dict({
        "type": "path",
        "max_depth": integrator.max_depth if integrator.max_depth > 0 else 8,
        "rr_depth": integrator.rr_depth,
    })

    result = mi.render(
        scene,
        spp=effective_spp,
        seed=seed,
        integrator=path_integrator,
    )

    if not develop:
        return mi.TensorXf()

    # TODO: Extract AOV buffers for denoiser input
    # aov = extract_aov_buffers(scene, sensor, seed, effective_spp)

    # TODO: Run JEPA denoiser
    # clean = run_jepa_denoiser(result, aov, integrator.jepa_model_path)

    return result


def extract_aov_buffers(scene, sensor, seed=0, spp=4):
    """Extract AOV auxiliary buffers for JEPA denoiser input.

    Renders the scene with Mitsuba's AOV integrator to extract:
        - Albedo (3 channels): diffuse color of surfaces
        - Normal (3 channels): surface normals in view space
        - Depth (1 channel): linear depth from camera
        - Motion vectors (2 channels): pixel displacement (if available)

    Uses DLPack zero-copy for GPU tensor transfer:
        nb.Tensor.from_dlpack(dr_tensor)

    Returns:
        dict of AOV buffer names → numpy arrays (H, W, C)
    """
    aov_integrator = mi.load_dict({
        "type": "aov",
        "aovs": "albedo:albedo,normal:normal,depth:depth",
    })

    aov_result = mi.render(
        scene,
        spp=spp,
        seed=seed,
        integrator=aov_integrator,
    )

    import numpy as np
    buffers = {}
    for key, tensor in aov_result.items():
        if hasattr(tensor, 'shape'):
            arr = np.array(tensor)
            if arr.ndim >= 2:
                buffers[key] = arr

    return buffers


def run_jepa_denoiser(noisy_tensor, aov_buffers, model_path):
    """Run JEPA denoiser on noisy render with AOV conditioning.

    Pipeline:
        1. Stack noisy RGBA + AOV buffers → input tensor (14 channels)
        2. Load JEPA model from checkpoint
        3. Forward pass through U-Net + MoE + scene conditioning
        4. Return clean RGBA via DLPack

    Args:
        noisy_tensor: Dr.Jit TensorXf (H, W, 4) noisy render
        aov_buffers: dict of AOV names → numpy arrays
        model_path: Path to .omen checkpoint

    Returns:
        Clean RGBA numpy array (H, W, 4)
    """
    import numpy as np

    try:
        import nabla as nb
        NABLA_AVAILABLE = True
    except ImportError:
        NABLA_AVAILABLE = False

    if not NABLA_AVAILABLE or not model_path:
        # No denoiser available — return noisy as-is
        return np.array(noisy_tensor)

    noisy_np = np.array(noisy_tensor)
    h, w = noisy_np.shape[0], noisy_np.shape[1]

    # Build 14-channel input: noisy_rgba(4) + prev_clean(4) + albedo(3) + normal(3)
    # prev_clean is zero for single-frame (no temporal context yet)
    prev_clean = np.zeros((h, w, 4), dtype=np.float32)
    albedo = aov_buffers.get('albedo', np.zeros((h, w, 3), dtype=np.float32))
    normal = aov_buffers.get('normal', np.zeros((h, w, 3), dtype=np.float32))

    input_tensor = np.concatenate([noisy_np, prev_clean, albedo, normal], axis=-1)

    # TODO: Load model, run inference via Nabla
    # model = load_omen_model(model_path)
    # clean = model(input_tensor)

    return noisy_np  # Placeholder — return noisy for now
