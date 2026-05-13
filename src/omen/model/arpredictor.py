"""ARPredictor - Autoregressive JEPA world model predictor.

Architecture (based on LeWorldModel):
- ConditionalBlock with AdaLN-zero conditioning (4 layers)
- SceneDeltaEncoder for animation changes
- Input: history window (H=3) + current latent -> predicted next latent

~4M params total
"""

import logging

try:
    import nabla as nb
    from nabla import nn
    import nabla.nn.functional as F
    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False

logger = logging.getLogger("omen.model.arpredictor")

LATENT_DIM = 192
NUM_HEADS = 8
NUM_LAYERS = 4
HISTORY_SIZE = 3


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning from delta embedding.

    Architecture:
    - AdaLN-zero: SiLU(delta_emb) -> Linear(dim, 6*dim) -> 6 modulation params
    - Modulation: x * (1 + scale) + shift for each of 6 params
    - Multi-head self-attention + MLP with gated residuals
    """

    def __init__(self, dim: int = LATENT_DIM, num_heads: int = NUM_HEADS):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # AdaLN-zero conditioning: SiLU + Linear(dim, 6*dim)
        # 6 params: shift1, scale1, gate1, shift2, scale2, gate2
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )

        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(d_model=dim, num_heads=num_heads)

        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, dim),
        )

    def _modulate(self, x, shift, scale):
        """AdaLN modulation: x * (1 + scale) + shift"""
        return x * (1 + scale) + shift

    def forward(self, x, delta_emb):
        """Forward pass with AdaLN-zero conditioning.

        Args:
            x: (batch, seq_len, dim) input sequence
            delta_emb: (batch, dim) scene delta conditioning

        Returns:
            (batch, seq_len, dim) transformed sequence
        """
        # Get 6 modulation parameters from delta
        mod_params = self.adaLN(delta_emb)
        s1, sc1, g1, s2, sc2, g2 = mod_params.chunk(6, axis=-1)

        # Attention block with modulation
        h = self._modulate(self.norm1(x), s1.unsqueeze(1), sc1.unsqueeze(1))
        attn_out = self.attn(h, h, h)
        x = x + g1.unsqueeze(1) * attn_out

        # MLP block with modulation
        h = self._modulate(self.norm2(x), s2.unsqueeze(1), sc2.unsqueeze(1))
        mlp_out = self.mlp(h)
        x = x + g2.unsqueeze(1) * mlp_out

        return x


class SceneDeltaEncoder(nn.Module):
    """Encode per-frame scene changes (camera, objects, lights, etc).

    Architecture: Linear smoothing layer -> MLP(smoothed -> 768 -> 192)
    ~155K params

    Note: Nabla has no nn.Conv1d, so we use Linear for the smoothing step.
    A Conv1d with kernel_size=1 is equivalent to a Linear layer anyway.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, delta_dim: int = 50):
        super().__init__()
        self.smooth = nn.Linear(delta_dim, delta_dim)
        self.mlp = nn.Sequential(
            nn.Linear(delta_dim, 768),
            nn.SiLU(),
            nn.Linear(768, latent_dim),
        )

    def forward(self, delta_tensor):
        """Encode scene delta.

        Args:
            delta_tensor: (batch, delta_dim) flattened delta vector

        Returns:
            (batch, latent_dim) delta embedding
        """
        # Linear smoothing (Conv1d k=1 == Linear)
        x = self.smooth(delta_tensor)
        return self.mlp(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor using ConditionalBlock transformer.

    Input: history window [latent_{N-3}, ..., latent_{N-1}, current_latent] -> (batch, 4, dim)
    Conditioning: delta embedding from SceneDeltaEncoder
    Output: predicted next latent (batch, dim)
    """

    def __init__(self, dim: int = LATENT_DIM, num_heads: int = NUM_HEADS,
                 num_layers: int = NUM_LAYERS, history_size: int = HISTORY_SIZE):
        super().__init__()
        self.dim = dim
        self.history_size = history_size

        # Positional embedding for sequence
        self.pos_embed = nn.Embedding(history_size + 1, dim)

        # Conditional transformer blocks
        # Plain list — Nabla has no ModuleList; modules self-register params
        self.blocks = [
            ConditionalBlock(dim, num_heads) for _ in range(num_layers)
        ]

        # Final norm and projection
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, history, current_latent, delta_emb):
        """Predict next frame latent.

        Args:
            history: list of (batch, dim) tensors, length <= history_size
            current_latent: (batch, dim) current frame latent
            delta_emb: (batch, dim) scene delta embedding

        Returns:
            predicted_latent: (batch, dim)
        """
        # Build sequence: [history[-3:], current_latent]
        seq = []
        for i, h in enumerate(history[-self.history_size:]):
            seq.append(h)
        seq.append(current_latent)

        # Stack into (batch, seq_len, dim)
        x = nb.stack(seq, axis=1)

        # Add positional embeddings
        positions = nb.arange(x.shape[1])
        pos_emb = self.pos_embed(positions)
        x = x + pos_emb

        # Apply conditional blocks
        for block in self.blocks:
            x = block(x, delta_emb)

        x = self.norm(x)

        # Take last position as prediction
        predicted = self.proj(x[:, -1, :])

        return predicted
