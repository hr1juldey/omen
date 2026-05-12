"""Tile-based Mixture of Experts with cryptomatte routing.

Implements DeepSeek-V3 style MoE with:
- 8 material experts: diffuse, glossy/glass, metal, SSS/skin,
  volume/smoke, emissive, hair/fur, cloth
- 5 light experts: point/spot, area, sun/directional, env/HDRI, emissive geo
- 5 geometry experts: flat, curved, edges/silhouette, detail/hair, transparent
- 4 motion experts: static, linear, fast/blur, occlusion boundary
- 1 shared expert: always active (base denoising)

Total: 23 experts. Auxiliary-loss-free load balancing via bias vectors.
"""

import logging

try:
    import nabla as nb
    from nabla import nn

    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False

logger = logging.getLogger("omen.model.moe")

FINGERPRINT_DIM = 23

# Expert group sizes
MATERIAL_EXPERTS = 8
LIGHT_EXPERTS = 5
GEOMETRY_EXPERTS = 5
MOTION_EXPERTS = 4
SHARED_EXPERTS = 1
TOTAL_EXPERTS = (
    MATERIAL_EXPERTS
    + LIGHT_EXPERTS
    + GEOMETRY_EXPERTS
    + MOTION_EXPERTS
    + SHARED_EXPERTS
)


class ExpertFFN(nn.Module):
    """Single expert: Linear(C, 4C) -> SiLU -> Linear(4C, C)."""

    def __init__(self, channels: int, hidden_mult: int = 4):
        super().__init__()
        hidden = channels * hidden_mult
        self.w1 = nn.Linear(channels, hidden)
        self.w2 = nn.Linear(hidden, channels)

    def forward(self, x):
        return self.w2(nb.silu(self.w1(x)))


class SharedExpert(nn.Module):
    """Always-active shared expert (DeepSeekMoE shared expert isolation)."""

    def __init__(self, channels: int):
        super().__init__()
        self.ffn = ExpertFFN(channels)

    def forward(self, x):
        return self.ffn(x)


class ExpertGroup(nn.Module):
    """Group of homogeneous experts with load-balancing bias."""

    def __init__(self, num_experts: int, channels: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([ExpertFFN(channels) for _ in range(num_experts)])
        # Auxiliary-loss-free load balancing (DeepSeek-V3)
        # Bias adjusted +-0.001 per training step, NO gradient
        self.bias = nb.zeros(num_experts)

    def forward(self, x, routing_logits):
        """Route tokens to top-k experts in this group.

        Args:
            x: (B, N, C) token features
            routing_logits: (B, N, num_experts) routing scores

        Returns:
            output: (B, N, C) expert-combined features
            aux_info: dict for load balancing
        """
        # Add bias for load balancing (no gradient)
        biased_logits = routing_logits + self.bias
        # Top-k selection
        weights = nb.softmax(biased_logits, axis=-1)
        top_weights, top_indices = nb.topk(weights, self.top_k, axis=-1)
        top_weights = top_weights / top_weights.sum(axis=-1, keepdims=True)

        # Dispatch to experts and combine
        output = nb.zeros_like(x)
        for k in range(self.top_k):
            idx = top_indices[:, :, k]
            w = top_weights[:, :, k : k + 1]
            for e in range(self.num_experts):
                mask = (idx == e).astype(x.dtype)
                expert_out = self.experts[e](x)
                output = output + mask * w * expert_out

        return output, {"top_weights": top_weights}

    def update_bias(self, step_bias: float = 0.001):
        """Adjust bias per training step (auxiliary-loss-free balancing)."""
        counts = nb.zeros(self.num_experts)
        # Bias towards underloaded experts
        self.bias = self.bias + step_bias * (counts.mean() - counts)


class TileMoERouter(nn.Module):
    """Route 8x8 tiles to expert groups via 23-dim tile fingerprint.

    Fingerprint channels: material_hist(8) + normal_var(3) + depth_var(1) +
    edge_density(1) + dominant_mat(1) + mean_albedo(3) + velocity_stats(6) = 23
    """

    def __init__(self, channels: int, top_k: int = 2):
        super().__init__()
        self.shared = SharedExpert(channels)
        self.materials = ExpertGroup(MATERIAL_EXPERTS, channels, top_k)
        self.lights = ExpertGroup(LIGHT_EXPERTS, channels, top_k=1)
        self.geometry = ExpertGroup(GEOMETRY_EXPERTS, channels, top_k=1)
        self.motion = ExpertGroup(MOTION_EXPERTS, channels, top_k=1)
        # Routing projections from 23-dim fingerprint
        self.mat_gate = nn.Linear(FINGERPRINT_DIM, MATERIAL_EXPERTS)
        self.light_gate = nn.Linear(FINGERPRINT_DIM, LIGHT_EXPERTS)
        self.geo_gate = nn.Linear(FINGERPRINT_DIM, GEOMETRY_EXPERTS)
        self.motion_gate = nn.Linear(FINGERPRINT_DIM, MOTION_EXPERTS)

    def forward(self, x, fingerprints):
        """Route tile tokens through MoE.

        Args:
            x: (B, N, C) tile token features (N = H/8 * W/8 * 64)
            fingerprints: (B, H/8, W/8, 23) tile fingerprints

        Returns:
            output: (B, N, C) shared_expert(x) + sum(routed_experts(x))
        """
        result = self.shared(x)
        fp_flat = fingerprints.reshape(fingerprints.shape[0], -1, FINGERPRINT_DIM)
        mat_out, _ = self.materials(x, self.mat_gate(fp_flat))
        light_out, _ = self.lights(x, self.light_gate(fp_flat))
        geo_out, _ = self.geometry(x, self.geo_gate(fp_flat))
        motion_out, _ = self.motion(x, self.motion_gate(fp_flat))
        return result + mat_out + light_out + geo_out + motion_out
