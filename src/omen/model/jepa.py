"""OmenJEPA - Top-level JEPA model with component switches.

Architecture (~8M params total):
- SceneGraphEncoder + RenderFeatureEncoder + CrossAttention (~3M)
- ARPredictor with ConditionalBlock layers (~4M) [optional]
- Decoder Conv2dTranspose (~1M)
- ConfidenceHead MLP
- EpisodicCorrection (~100K params) [optional]
- SIGReg loss (0 learnable params) [optional]
"""

import logging

try:
    import nabla as nb
    from nabla import nn
    import nabla.nn.functional as F
    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False

from omen.config import OmenConfig
from omen.model.scene_encoder import SceneGraphEncoder, RenderFeatureEncoder, CrossAttentionFusion
from omen.model.arpredictor import ARPredictor, SceneDeltaEncoder
from omen.model.decoder import Decoder, ConfidenceHead
from omen.model.sigreg import SIGRegLoss
from omen.model.episodic import EpisodicCorrection

logger = logging.getLogger("omen.model.jepa")

LATENT_DIM = 192
SIGREG_LAMBDA = 0.09


class OmenJEPA(nn.Module):
    """Omen JEPA model with component switches.

    All components always exist (parameters initialized).
    Config switches control forward behavior and gradient flow.
    Disabled components use identity passthrough.
    """

    def __init__(self, config: OmenConfig = None, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.config = config or OmenConfig()
        self.latent_dim = latent_dim
        c = self.config.components

        # Core encoders (always-on for V1)
        self.scene_encoder = SceneGraphEncoder(latent_dim)
        self.render_encoder = RenderFeatureEncoder(latent_dim)
        self.fusion = CrossAttentionFusion(latent_dim)

        # Temporal prediction (optional)
        self.delta_encoder = SceneDeltaEncoder(latent_dim)
        self.ar_predictor = ARPredictor(latent_dim)

        # Decoders (always-on for V1)
        self.decoder = Decoder(latent_dim)
        self.confidence_head = ConfidenceHead(latent_dim)

        # Episodic correction (optional, but ON by default in V1)
        if c.episodic_correction:
            self.episodic = EpisodicCorrection(latent_dim)

        # Loss (SIGReg optional, simple_var_reg ON by default)
        self.sigreg = SIGRegLoss()

    def encode(self, scene_graph, rgba):
        """Encode scene graph + render into fused latent."""
        scene_latent = self.scene_encoder(scene_graph)
        render_latent = self.render_encoder(rgba)
        fused = self.fusion(render_latent, scene_latent)
        return fused, scene_latent

    def decode(self, latent, height, width):
        """Decode latent to RGBA image."""
        return self.decoder(latent, height, width)

    def predict_confidence(self, latent, height, width):
        """Predict per-pixel confidence map."""
        if not self.config.components.confidence_head:
            return None
        return self.confidence_head(latent, height, width)

    def predict_temporal(self, history, current_latent, delta_emb):
        """Predict next frame latent using ARPredictor.

        Returns current_latent unchanged (identity) when ARPredictor disabled.
        """
        c = self.config.components
        if not c.ar_predictor:
            return current_latent  # Identity passthrough

        return self.ar_predictor(history, current_latent, delta_emb)

    def compute_loss(self, predicted_latent, target_latent, config=None):
        """Compute JEPA loss with config switches.

        Args:
            predicted_latent: (batch, latent_dim) predicted latent
            target_latent: (batch, latent_dim) ground truth latent
            config: OmenConfig (uses self.config if None)

        Returns:
            total_loss, pred_loss, reg_loss
        """
        cfg = config or self.config

        # JEPA prediction loss
        pred_loss = F.mse_loss(predicted_latent, target_latent)

        # Regularization (respects config switches)
        reg_loss = self.sigreg(predicted_latent, config=cfg)
        total = pred_loss + SIGREG_LAMBDA * reg_loss

        return total, pred_loss, reg_loss
