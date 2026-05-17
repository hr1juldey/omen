"""Functional AR predictor — JAX-style forward pass.

Mirrors ARPredictor + ConditionalBlock + MultiHeadAttention.forward
but takes params dict directly.
"""

import nabla as nb

NUM_HEADS = 8
NUM_LAYERS = 4
HISTORY_SIZE = 3


def _linear(x, weight, bias):
    return x @ weight + bias


def _layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / nb.sqrt(var + eps) * weight + bias


def _mha_fn(attn_p, x, num_heads):
    """Functional multi-head self-attention."""
    B, S, D = (int(d) for d in x.shape)
    head_dim = D // num_heads
    scale = head_dim**-0.5

    q = _linear(x, attn_p["q_proj.weight"], attn_p["q_proj.bias"])
    k = _linear(x, attn_p["k_proj.weight"], attn_p["k_proj.bias"])
    v = _linear(x, attn_p["v_proj.weight"], attn_p["v_proj.bias"])

    # (B, S, D) -> (B, H, S, d)
    q = nb.permute(nb.reshape(q, (B, S, num_heads, head_dim)), (0, 2, 1, 3))
    k = nb.permute(nb.reshape(k, (B, S, num_heads, head_dim)), (0, 2, 1, 3))
    v = nb.permute(nb.reshape(v, (B, S, num_heads, head_dim)), (0, 2, 1, 3))

    scores = nb.matmul(q, nb.permute(k, (0, 1, 3, 2))) * scale
    weights = nb.softmax(scores, axis=-1)
    out = nb.matmul(weights, v)

    out = nb.permute(out, (0, 2, 1, 3))
    out = nb.reshape(out, (B, S, D))
    return _linear(out, attn_p["out_proj.weight"], attn_p["out_proj.bias"])


def _block_fn(p, x, delta_emb, num_heads):
    """Functional ConditionalBlock with AdaLN-zero."""
    dim = int(x.shape[-1])
    ada_out = _linear(nb.silu(delta_emb), p["adaLN.1.weight"], p["adaLN.1.bias"])

    s1 = ada_out[:, 0:dim]
    sc1 = ada_out[:, dim : 2 * dim]
    g1 = ada_out[:, 2 * dim : 3 * dim]
    s2 = ada_out[:, 3 * dim : 4 * dim]
    sc2 = ada_out[:, 4 * dim : 5 * dim]
    g2 = ada_out[:, 5 * dim : 6 * dim]

    # Attention with modulation
    h = _layer_norm(x, p["norm1.weight"], p["norm1.bias"])
    h = h * (1 + nb.reshape(sc1, (-1, 1, dim))) + nb.reshape(s1, (-1, 1, dim))
    x = x + nb.reshape(g1, (-1, 1, dim)) * _mha_fn(p["attn"], h, num_heads)

    # MLP with modulation
    h = _layer_norm(x, p["norm2.weight"], p["norm2.bias"])
    h = h * (1 + nb.reshape(sc2, (-1, 1, dim))) + nb.reshape(s2, (-1, 1, dim))
    mlp_h = nb.silu(_linear(h, p["mlp.0.weight"], p["mlp.0.bias"]))
    mlp_out = _linear(mlp_h, p["mlp.2.weight"], p["mlp.2.bias"])
    x = x + nb.reshape(g2, (-1, 1, dim)) * mlp_out

    return x


def ar_predictor_fn(p, history, current_latent, delta_emb):
    """Predict next frame latent using params dict.

    Args:
        p: prefix-stripped params with pos_embed, block_0-3, norm, proj.
        history: list of (batch, dim) tensors.
        current_latent: (batch, dim) current frame latent.
        delta_emb: (batch, dim) scene delta embedding.

    Returns:
        (batch, dim) predicted next latent.
    """
    seq = list(history[-HISTORY_SIZE:]) + [current_latent]
    x = nb.stack(seq, axis=1)

    positions = nb.arange(int(x.shape[1]))
    pos_emb = p["pos_embed.weight"][positions]
    x = x + pos_emb

    for i in range(NUM_LAYERS):
        bp = {
            k[len(f"block_{i}.") :]: v
            for k, v in p.items()
            if k.startswith(f"block_{i}.")
        }
        x = _block_fn(bp, x, delta_emb, NUM_HEADS)

    x = _layer_norm(x, p["norm.weight"], p["norm.bias"])
    return _linear(x[:, -1, :], p["proj.weight"], p["proj.bias"])
