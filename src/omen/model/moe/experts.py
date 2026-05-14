"""Expert modules for MoE system."""

try:
    import nabla as nb
    from nabla import nn
    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False


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
        for i in range(num_experts):
            setattr(self, f"expert_{i}", ExpertFFN(channels))
        self.experts = [getattr(self, f"expert_{i}") for i in range(num_experts)]
        self.bias = nb.zeros((num_experts,))

    def forward(self, x, routing_logits):
        """Route tokens to top-k experts in this group."""
        biased_logits = routing_logits + self.bias
        weights = nb.softmax(biased_logits, axis=-1)
        top_weights, top_indices = nb.topk(weights, self.top_k, axis=-1)
        top_weights = top_weights / top_weights.sum(axis=-1, keepdims=True)

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
        counts = nb.zeros((self.num_experts,))
        self.bias = self.bias + step_bias * (counts.mean() - counts)
