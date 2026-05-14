"""Scene-Aware Dual Encoder for 3D scene encoding.

Replaces ViT-Tiny with two parallel encoders:
1. SceneGraphEncoder - encodes geometry, materials, lights (~1M params)
2. RenderFeatureEncoder - encodes render image features via Conv2d (~1.5M params)
3. Cross-attention fusion merges both into unified latent (~0.5M params)

Total: ~3M params (vs 5.5M ViT-Tiny)
Output: latent vector of shape (batch, 192)
"""

import logging
import numpy as np

try:
    import nabla as nb
    from nabla import nn
    import nabla.nn.functional as F
    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False

from omen.kernels.conv2d import conv2d_safe

logger = logging.getLogger("omen.model.scene_encoder")

LATENT_DIM = 1024
NUM_HEADS = 8


class SceneGraphEncoder(nn.Module):
    """Encode scene graph (geometry, materials, lights) into latent vector.

    Architecture:
    - Geometry: Linear(6, 64) -> MHA attention over face features
    - Materials: Embedding + Linear(params_dim, 64) per material
    - Lights: Linear(7, 64) per light
    - Cross-attention fusion -> Linear(64, LATENT_DIM)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        # Geometry: input is face center (3) + normal (3) = 6 features
        self.geom_linear = nn.Linear(6, 64)
        # Materials: 5 params per material
        self.mat_linear = nn.Linear(5, 64)
        # Lights: 7 params per light
        self.light_linear = nn.Linear(7, 64)
        # Project to latent dim (mean pooling replaces self-attn for 3 features)
        self.proj = nn.Linear(64, latent_dim)

    def forward(self, scene_graph: dict):
        """Encode scene graph components.

        Args:
            scene_graph: dict with geometry, materials, lights tensors

        Returns:
            latent: (batch, latent_dim)
        """
        features = []

        # Geometry features
        geom = scene_graph.get("geometry", {})
        if isinstance(geom, dict):
            verts = geom.get("vertices", None)
            faces = geom.get("faces", None)
            if verts is not None:
                # Compute face centers and normals as features
                # verts: (batch, num_verts, 3) -> centroid + spread = 6 features
                if len(verts.shape) >= 2:
                    # Centroid: mean over vertex dim
                    if len(verts.shape) == 3:
                        centroid = verts.mean(axis=1)  # (B, 3)
                        B = int(centroid.shape[0])
                        D = int(centroid.shape[1])
                        spread = nb.mean(
                            (verts - nb.reshape(centroid, (B, 1, D))) ** 2, axis=1
                        )  # (B, 3) variance per axis
                        face_feats = nb.concatenate([centroid, spread], axis=-1)
                    else:
                        face_feats = nb.reshape(verts, (1, int(verts.shape[-1])))
                        n = int(face_feats.shape[-1])
                        if n < 6:
                            face_feats = nb.pad(face_feats, ((0, 0), (0, 6 - n)))
                        face_feats = face_feats[:, :6]
                    geom_emb = self.geom_linear(face_feats)
                    features.append(geom_emb)

        # Material features
        mats = scene_graph.get("materials", {})
        if isinstance(mats, dict):
            params = mats.get("params", None)
            if params is not None and len(params.shape) >= 2:
                mat_emb = self.mat_linear(params)
                features.append(nb.mean(mat_emb, axis=1))

        # Light features
        lights = scene_graph.get("lights", {})
        if isinstance(lights, dict):
            params = lights.get("params", None)
            if params is not None and len(params.shape) >= 2:
                light_emb = self.light_linear(params)
                features.append(nb.mean(light_emb, axis=1))

        if not features:
            # Empty scene - return zeros
            return nb.zeros((1, self.latent_dim))

        # Concatenate all features and mean pool
        all_feats = nb.concatenate(features, axis=0)
        pooled = nb.mean(all_feats, axis=0)
        pooled = nb.reshape(pooled, (1, int(pooled.shape[0])))

        # Project to latent dim
        return self.proj(pooled)


class RenderFeatureEncoder(nn.Module):
    """Encode render image features via Conv2d functional stack.

    Uses nb.conv2d() functional API (NHWC format, HWIO filter layout).
    Architecture:
    Conv2d(4, 32, 3, stride=2) -> SiLU
    Conv2d(32, 64, 3, stride=2) -> SiLU
    Conv2d(64, 128, 3, stride=2) -> SiLU
    Global average pool
    Linear(128, LATENT_DIM)
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        # Store conv filters as learnable parameters (HWIO layout for nb.conv2d)
        # conv1: (3, 3, 4, 32) - 4 input channels, 32 output
        self.conv1_filter = F.he_normal((3, 3, 4, 32))
        self.conv1_filter.requires_grad = True
        self.conv1_bias = nb.zeros((32,))
        self.conv1_bias.requires_grad = True
        # conv2: (3, 3, 32, 64)
        self.conv2_filter = F.he_normal((3, 3, 32, 64))
        self.conv2_filter.requires_grad = True
        self.conv2_bias = nb.zeros((64,))
        self.conv2_bias.requires_grad = True
        # conv3: (3, 3, 64, 128)
        self.conv3_filter = F.he_normal((3, 3, 64, 128))
        self.conv3_filter.requires_grad = True
        self.conv3_bias = nb.zeros((128,))
        self.conv3_bias.requires_grad = True
        self.proj = nn.Linear(128, latent_dim)

    def forward(self, rgba: "nb.Tensor"):
        """Encode RGBA render image to latent vector.

        Args:
            rgba: (batch, H, W, 4) RGBA render (already NHWC)

        Returns:
            latent: (batch, latent_dim)
        """
        # conv2d_safe: NHWC input, HWIO filter
        x = nb.silu(conv2d_safe(rgba, self.conv1_filter, stride=2, padding=1, bias=self.conv1_bias))
        x = nb.silu(conv2d_safe(x, self.conv2_filter, stride=2, padding=1, bias=self.conv2_bias))
        x = nb.silu(conv2d_safe(x, self.conv3_filter, stride=2, padding=1, bias=self.conv3_bias))

        # Global average pool over spatial dims (H, W) -> (B, 128)
        x = x.mean(axis=(1, 2))

        return self.proj(x)


class CrossAttentionFusion(nn.Module):
    """Fuse render and scene latents via gated addition.

    Replaces cross-attention with a learnable gate (nabla backward compatible).
    """

    def __init__(self, latent_dim: int = LATENT_DIM, num_heads: int = NUM_HEADS):
        super().__init__()
        self.latent_dim = latent_dim
        self.gate = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, render_latent: "nb.Tensor", scene_latent: "nb.Tensor"):
        """Gated fusion.

        Args:
            render_latent: (batch, latent_dim) from RenderFeatureEncoder
            scene_latent: (batch, latent_dim) from SceneGraphEncoder

        Returns:
            fused: (batch, latent_dim)
        """
        # Learned gate: how much scene info to mix in
        g = nb.sigmoid(self.gate(render_latent))
        fused = self.norm(render_latent + g * scene_latent)
        return fused
