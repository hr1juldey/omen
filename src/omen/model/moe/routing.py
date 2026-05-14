"""Routing methods for MoE system."""

FINGERPRINT_DIM = 23
NUM_MATERIAL_TYPES = 6
NUM_LIGHT_TYPES = 4
SCENE_GRAPH_EMBED_DIM = 64


def route_from_scene_graph(
    scene_graph,
    aux,
    mat_embed,
    light_embed,
    route_proj,
):
    """Build routing logits from material_ids and light_type_ids.

    Args:
        scene_graph: dict with material_ids and lights params
        aux: auxiliary buffer for shape reference
        mat_embed: nn.Embedding for material types
        light_embed: nn.Embedding for light types
        route_proj: nn.Linear for final projection

    Returns:
        routing: (B, H/8, W/8, 23) routing logits
    """
    material_ids = scene_graph.get("geometry", {}).get("material_ids")
    light_params = scene_graph.get("lights", {}).get("params")

    if material_ids is None or light_params is None:
        # Fallback to zeros if scene graph data missing
        h_tiles = aux.shape[1] // 8 if aux is not None else 1
        w_tiles = aux.shape[2] // 8 if aux is not None else 1
        import nabla as nb
        return nb.zeros((1, h_tiles, w_tiles, FINGERPRINT_DIM))

    # Get dominant material and light type
    import nabla as nb
    mat_id = int(nb.mean(material_ids.astype(nb.float32)).to_numpy().item())
    mat_id = min(mat_id, NUM_MATERIAL_TYPES - 1)

    light_type = int(light_params[0, 0].to_numpy().item()) if light_params.shape[0] > 0 else 0
    light_type = min(light_type, NUM_LIGHT_TYPES - 1)

    # Embed and project
    mat_emb = mat_embed(nb.tensor([[mat_id]]))
    light_emb = light_embed(nb.tensor([[light_type]]))
    combined = nb.concat([mat_emb, light_emb], axis=-1)
    routing_flat = route_proj(combined)

    # Expand to tile grid shape
    h_tiles = aux.shape[1] // 8 if aux is not None else 1
    w_tiles = aux.shape[2] // 8 if aux is not None else 1
    routing = routing_flat.repeat(1, h_tiles, w_tiles, 1)

    return routing


def route_from_fingerprints(aux):
    """Build routing logits from pixel-derived fingerprints.

    Args:
        aux: (B, H, W, C) auxiliary buffer

    Returns:
        routing: (B, H/8, W/8, 23) routing logits
    """
    import nabla as nb

    fingerprint = aux[:, :, :, :FINGERPRINT_DIM]
    h_tiles = aux.shape[1] // 8
    w_tiles = aux.shape[2] // 8

    fingerprint_tiles = (
        fingerprint.reshape(aux.shape[0], h_tiles, 8, w_tiles, 8, FINGERPRINT_DIM)
        .mean(axis=(2, 4))
    )

    return fingerprint_tiles
