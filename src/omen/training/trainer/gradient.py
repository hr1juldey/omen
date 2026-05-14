"""Gradient clipping utility for Nabla training."""


def _clip_grad_norm(parameters, max_norm):
    """Clip gradient norm to prevent explosion."""
    try:
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm += (p.grad ** 2).sum().to_numpy().item()
        total_norm = total_norm ** 0.5

        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad = p.grad * scale
    except Exception:
        pass
