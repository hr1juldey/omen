"""Python bridge for moe_dispatch Mojo GPU kernel.

Fuses expert routing + weighted combination into single GPU kernel.
Replaces Python for-loops in omen.model.moe ExpertGroup.forward().
"""

import logging
from pathlib import Path

import numpy as np

try:
    from nabla.ops import call_custom_kernel

    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False

logger = logging.getLogger("omen.kernels.moe_dispatch")

MAX_EXPERTS = 8
KERNEL_DIR = Path(__file__).parent


def compute_moe_dispatch_gpu(
    expert_outputs: np.ndarray,
    routing_weights: np.ndarray,
) -> np.ndarray:
    """Dispatch MoE expert combination via Nabla + Mojo GPU kernel.

    Args:
        expert_outputs: (tokens, channels, num_experts) stacked expert results
        routing_weights: (tokens, num_experts) sparse top-k weights

    Returns:
        (tokens, channels) weighted combination
    """
    if not NABLA_AVAILABLE:
        return compute_moe_dispatch_numpy(expert_outputs, routing_weights)

    try:
        import nabla as nb
        from nabla.ops import UnaryOperation

        class MoEDispatchOp(UnaryOperation):
            name = "moe_dispatch"

            def compute_physical_shape(self, args, kwargs, output_sharding=None):
                eo = args[0]
                t, c, _ = eo.shape
                return [(t, c)], [eo.dtype], [eo.device]

            def kernel(self, expert_out, routing_w, **kwargs):
                return call_custom_kernel(
                    "moe_dispatch", str(KERNEL_DIR), expert_out, routing_w
                )

        eo_tensor = nb.Tensor.from_dlpack(expert_outputs.astype(np.float32))
        rw_tensor = nb.Tensor.from_dlpack(routing_weights.astype(np.float32))
        op = MoEDispatchOp()
        result = op(eo_tensor, rw_tensor)
        return result.to_numpy()
    except Exception as exc:
        logger.warning("MoE dispatch Mojo failed (%s) — numpy fallback", exc)
        return compute_moe_dispatch_numpy(expert_outputs, routing_weights)


def compute_moe_dispatch_numpy(
    expert_outputs: np.ndarray,
    routing_weights: np.ndarray,
) -> np.ndarray:
    """Pure numpy fallback for MoE dispatch."""
    # expert_outputs: (T, C, E), routing_weights: (T, E)
    # output: (T, C) = sum_e(weight[t,e] * expert_out[t,:,e])
    return np.einsum("tce,te->tc", expert_outputs, routing_weights)
