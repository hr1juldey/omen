"""OmenJEPA - Top-level JEPA model combining all components.

Architecture (~8M params total):
- SceneGraphEncoder + RenderFeatureEncoder + CrossAttention (~3M)
- ARPredictor with ConditionalBlock layers (~4M)
- Decoder Conv2dTranspose (~1M)
- ConfidenceHead MLP
- SIGReg loss (0 learnable params)
"""

import logging

try:
    import nabla as nb
    from nabla import nn
    import nabla.nn.functional as F
    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False

from omen.model.scene_encoder import SceneGraphEncoder, RenderFeatureEncoder, CrossAttentionFusion
from omen.model.arpredictor import ARPredictor, SceneDeltaEncoder
from omen.model.decoder import Decoder, ConfidenceHead
from omen.model.sigreg import SIGRegLoss

logger = logging.getLogger("omen.model.jepa")

LATENT_DIM = 192


class OmenJEPA(nn.Module):
    """Omen JEPA model for scene-aware rendering acceleration.

    Modes:
    1. Denoise: encode noisy render + scene -> decode clean RGBA
    2. Confidence: encode + predict per-pixel confidence
    3. Multires merge: dual encode + merge
    4. Temporal predict: ARPredictor with history + delta
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoders
        self.scene_encoder = SceneGraphEncoder(latent_dim)
        self.render_encoder = RenderFeatureEncoder(latent_dim)
        self.fusion = CrossAttentionFusion(latent_dim)

        # Temporal prediction
        self.delta_encoder = SceneDeltaEncoder(latent_dim)
        self.ar_predictor = ARPredictor(latent_dim)

        # Decoders
        self.decoder = Decoder(latent_dim)
        self.confidence_head = ConfidenceHead(latent_dim)

        # Loss
        self.sigreg = SIGRegLoss()

    def encode(self, scene_graph, rgba):
        """Encode scene graph + render into fused latent.

        Args:
            scene_graph: dict of tensors from scene_extractor
            rgba: (batch, H, W, 4) render

        Returns:
            latent: (batch, latent_dim) fused latent
        """
        scene_latent = self.scene_encoder(scene_graph)
        render_latent = self.render_encoder(rgba)
        fused = self.fusion(render_latent, scene_latent)
        return fused

    def decode(self, latent, height, width):
        """Decode latent to RGBA image.

        Args:
            latent: (batch, latent_dim)
            height: target height
            width: target width

        Returns:
            rgba: (batch, height, width, 4) in [0, 1]
        """
        return self.decoder(latent, height, width)

    def predict_confidence(self, latent, height, width):
        """Predict per-pixel confidence map.

        Args:
            latent: (batch, latent_dim)
            height: image height
            width: image width

        Returns:
            confidence: (batch, height, width, 1) in [0, 1]
        """
        return self.confidence_head(latent, height, width)

    def predict_temporal(self, history, current_latent, delta_emb):
        """Predict next frame latent using ARPredictor.

        Args:
            history: list of (batch, latent_dim) past latents
            current_latent: (batch, latent_dim) current frame
            delta_emb: (batch, latent_dim) scene delta

        Returns:
            predicted: (batch, latent_dim)
        """
        return self.ar_predictor(history, current_latent, delta_emb)

    def merge(self, scene_graph, low_res, high_res, scale):
        """Merge low-res clean + high-res noisy renders.

        Args:
            scene_graph: dict of tensors
            low_res: (batch, H//scale, W//scale, 4) clean low-res
            high_res: (batch, H, W, 4) noisy high-res
            scale: downscale factor

        Returns:
            merged: (batch, H, W, 4)
        """
        # Encode both
        low_latent = self.encode(scene_graph, low_res)
        high_latent = self.encode(scene_graph, high_res)

        # Average latents (simple merge - can be made smarter)
        merged_latent = (low_latent + high_latent) / 2.0

        # Decode at full resolution
        h = high_res.shape[1]
        w = high_res.shape[2]
        return self.decode(merged_latent, h, w)

    def compute_loss(self, predicted_latent, target_latent, embeddings=None, lambda_sigreg=0.09):
        """Compute JEPA loss: L_pred(latent) + lambda * L_sigreg.

        This is the core JEPA training loss — it operates in LATENT space,
        NOT pixel space. The model learns to predict latent representations,
        not pixels. The decoder is only used at inference time to render output.

        Args:
            predicted_latent: (batch, latent_dim) predicted latent
            target_latent: (batch, latent_dim) ground truth latent (encoded from clean render)
            embeddings: (batch, dim) latent embeddings for SIGReg collapse prevention
            lambda_sigreg: SIGReg weight (default 0.09 from lewm.yaml)

        Returns:
            total_loss, pred_loss, sigreg_loss
        """
        # JEPA: prediction loss in LATENT space
        pred_loss = F.mse_loss(predicted_latent, target_latent)

        # SIGReg: prevent representation collapse
        if embeddings is not None:
            sigreg_loss = self.sigreg(embeddings)
        else:
            sigreg_loss = nb.constant(0.0)

        total = pred_loss + lambda_sigreg * sigreg_loss
        return total, pred_loss, sigreg_loss
