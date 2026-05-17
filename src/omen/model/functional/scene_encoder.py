"""Functional scene encoder — JAX-style forward pass.

Mirrors SceneGraphEncoder.forward but takes params dict directly.
"""

import nabla as nb


def scene_encoder_fn(p, scene_graph):
    """Encode scene graph into latent vector using params dict.

    Args:
        p: prefix-stripped params with geom_linear, mat_linear, light_linear, proj.
        scene_graph: dict with geometry/materials/lights tensors.

    Returns:
        (batch, latent_dim) scene latent.
    """
    features = []

    # Geometry: centroid + spread -> Linear(6, 64)
    geom = scene_graph.get("geometry", {})
    if isinstance(geom, dict):
        verts = geom.get("vertices")
        if verts is not None and len(verts.shape) >= 2:
            if len(verts.shape) == 3:
                centroid = verts.mean(axis=1)
                B, D = int(centroid.shape[0]), int(centroid.shape[1])
                spread = nb.mean((verts - nb.reshape(centroid, (B, 1, D))) ** 2, axis=1)
                face_feats = nb.concatenate([centroid, spread], axis=-1)
            else:
                face_feats = nb.reshape(verts, (1, int(verts.shape[-1])))
                n = int(face_feats.shape[-1])
                if n < 6:
                    face_feats = nb.pad(face_feats, ((0, 0), (0, 6 - n)))
                face_feats = face_feats[:, :6]
            features.append(
                face_feats @ p["geom_linear.weight"] + p["geom_linear.bias"]
            )

    # Materials: Linear(5, 64) -> mean pool
    mats = scene_graph.get("materials", {})
    if isinstance(mats, dict):
        params_m = mats.get("params")
        if params_m is not None and len(params_m.shape) >= 2:
            mat_emb = params_m @ p["mat_linear.weight"] + p["mat_linear.bias"]
            features.append(nb.mean(mat_emb, axis=1))

    # Lights: Linear(7, 64) -> mean pool
    lights = scene_graph.get("lights", {})
    if isinstance(lights, dict):
        params_l = lights.get("params")
        if params_l is not None and len(params_l.shape) >= 2:
            light_emb = params_l @ p["light_linear.weight"] + p["light_linear.bias"]
            features.append(nb.mean(light_emb, axis=1))

    if not features:
        return nb.zeros((1, int(p["proj.weight"].shape[0])))

    all_feats = nb.concatenate(features, axis=0)
    pooled = nb.mean(all_feats, axis=0)
    pooled = nb.reshape(pooled, (1, int(pooled.shape[0])))
    return pooled @ p["proj.weight"] + p["proj.bias"]
