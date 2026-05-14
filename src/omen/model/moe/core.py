"""TileMoERouter - main MoE routing class."""

import logging

try:
    import nabla as nb
    from nabla import nn
    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False

from omen.config import OmenConfig
from omen.model.moe.experts import ExpertGroup, SharedExpert
from omen.model.moe.routing import FINGERPRINT_DIM, route_from_fingerprints, route_from_scene_graph

logger = logging.getLogger("omen.model.moe")

# Expert group sizes
MATERIAL_EXPERTS = 8
LIGHT_EXPERTS = 5
GEOMETRY_EXPERTS = 5
MOTION_EXPERTS = 4


class TileMoERouter(nn.Module):
    """Route 8x8 tiles to expert groups via dual routing paths.

    Routing modes (controlled by config.components.scene_graph_routing):
    - False (default): 23-dim tile fingerprint from aux buffers
    - True: material_id + light_type embeddings from scene graph
    """

    def __init__(
        self,
        channels: int,
        top_k: int = 2,
        config: OmenConfig = None,
    ):
        super().__init__()
        self.config = config or OmenConfig()
        self.shared = SharedExpert(channels)
        self.materials = ExpertGroup(MATERIAL_EXPERTS, channels, top_k)
        self.lights = ExpertGroup(LIGHT_EXPERTS, channels, top_k=1)
        self.geometry = ExpertGroup(GEOMETRY_EXPERTS, channels, top_k=1)
        self.motion = ExpertGroup(MOTION_EXPERTS, channels, top_k=1)

        # Pixel fingerprint routing gates
        self.mat_gate = nn.Linear(FINGERPRINT_DIM, MATERIAL_EXPERTS)
        self.light_gate = nn.Linear(FINGERPRINT_DIM, LIGHT_EXPERTS)
        self.geo_gate = nn.Linear(FINGERPRINT_DIM, GEOMETRY_EXPERTS)
        self.motion_gate = nn.Linear(FINGERPRINT_DIM, MOTION_EXPERTS)

        # Scene-graph routing (only created when enabled)
        if self.config.components.scene_graph_routing:
            from omen.model.moe.routing import NUM_LIGHT_TYPES, NUM_MATERIAL_TYPES, SCENE_GRAPH_EMBED_DIM
            self.mat_embed = nn.Embedding(NUM_MATERIAL_TYPES, SCENE_GRAPH_EMBED_DIM)
            self.light_embed = nn.Embedding(NUM_LIGHT_TYPES, SCENE_GRAPH_EMBED_DIM)
            from omen.model.moe.routing import SCENE_GRAPH_EMBED_DIM
            self.route_proj = nn.Linear(SCENE_GRAPH_EMBED_DIM * 2, FINGERPRINT_DIM)

    def forward(self, x, aux=None, scene_graph=None):
        """Route tile tokens through MoE.

        Args:
            x: (B, N, C) tile token features
            aux: (B, H, W, C) auxiliary buffer
            scene_graph: dict for scene-graph routing

        Returns:
            output: (B, N, C) routed features
        """
        c = self.config.components

        # Build routing logits
        if c.scene_graph_routing and scene_graph is not None:
            fingerprints = route_from_scene_graph(
                scene_graph, aux, self.mat_embed, self.light_embed, self.route_proj
            )
        else:
            if aux is not None:
                fingerprints = route_from_fingerprints(aux)
            else:
                import nabla as nb
                fingerprints = nb.zeros((x.shape[0], 1, 1, FINGERPRINT_DIM))

        fp_flat = fingerprints.reshape(fingerprints.shape[0], -1, FINGERPRINT_DIM)

        # Route through shared + expert groups
        result = self.shared(x)

        if c.moe:
            if c.moe_materials:
                mat_out, _ = self.materials(x, self.mat_gate(fp_flat))
                result = result + mat_out
            if c.moe_lights:
                light_out, _ = self.lights(x, self.light_gate(fp_flat))
                result = result + light_out
            if c.moe_geometry:
                geo_out, _ = self.geometry(x, self.geo_gate(fp_flat))
                result = result + geo_out
            if c.moe_motion:
                motion_out, _ = self.motion(x, self.motion_gate(fp_flat))
                result = result + motion_out

        return result
