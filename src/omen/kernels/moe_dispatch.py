"""Python bridge for moe_dispatch Mojo GPU kernel.

Fuses expert routing + weighted combination into single GPU kernel.
Packs expert_outputs + routing_weights into single flat tensor to avoid
the MAX framework multi-input custom kernel data transfer bug.
"""

import logging
from pathlib import Path

import numpy as np

try:
    from nabla.ops import UnaryOperation, call_custom_kernel

    NABLA_AVAILABLE = True
except ImportError:
    UnaryOperation = object
    NABLA_AVAILABLE = False

logger = logging.getLogger("omen.kernels.moe_dispatch")

MAX_EXPERTS = 8
KERNEL_DIR = Path(__file__).parent


def _pack_moe(expert_outputs: np.ndarray, routing_weights: np.ndarray) -> np.ndarray:
    """Pack expert_outputs + routing_weights into single flat tensor.

    Layout: [T, C, E, eo_flat, rw_flat]
    """
    t, c, e = expert_outputs.shape
    header = np.array([t, c, e], dtype=np.float32)
    return np.concatenate([header, expert_outputs.flatten(), routing_weights.flatten()])


class MoEDispatchOp(UnaryOperation):
    """Nabla op wrapping Mojo moe_dispatch kernel (single packed input)."""

    @property
    def name(self) -> str:
        return "moe_dispatch"

    def __init__(self, tokens: int, channels: int):
        self.tokens = tokens
        self.channels = channels

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        return (
            [(self.tokens, self.channels)],
            [args[0].dtype],
            [args[0].device],
        )

    def kernel(self, args, kwargs):
        from max.graph import TensorType

        source = args[0]
        out_type = TensorType(
            dtype=source.dtype,
            shape=(self.tokens, self.channels),
            device=source.device,
        )
        result = call_custom_kernel("moe_dispatch", str(KERNEL_DIR), source, out_type)
        return [result]


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

        eo32 = expert_outputs.astype(np.float32)
        rw32 = routing_weights.astype(np.float32)
        t, c, _ = eo32.shape

        packed = _pack_moe(eo32, rw32)
        tensor = nb.Tensor.from_dlpack(packed)

        op = MoEDispatchOp(tokens=t, channels=c)
        result = op([tensor], {})[0]
        return result.to_numpy()
    except Exception as exc:
        logger.warning("MoE dispatch Mojo failed (%s) — numpy fallback", exc)
        return compute_moe_dispatch_numpy(expert_outputs, routing_weights)


def compute_moe_dispatch_numpy(
    expert_outputs: np.ndarray,
    routing_weights: np.ndarray,
) -> np.ndarray:
    """Pure numpy fallback for MoE dispatch."""
    return np.einsum("tce,te->tc", expert_outputs, routing_weights)
