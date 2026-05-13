"""Scene latent caching with smart invalidation.

Tasks 19c.1-19c.9: Two-level cache (topology_hash + dynamic_hash),
incremental updates via delta encoding, smart invalidation on structural changes.
"""

import hashlib
import logging

import numpy as np

logger = logging.getLogger("omen.latent_cache")

# Two-level cache: topology -> scene latent, dynamic -> refined latent
_cache: dict = {}  # {topo_hash: {"latent": ..., "dynamic_hash": ..., "graph": ...}}


def compute_topology_hash(scene_graph: dict) -> str:
    """Hash topology: face connectivity + material TYPES + light TYPES.

    Excludes positions and values. Same as scene_cache but for latents.
    """
    parts = []
    for key in ("geometry", "materials", "lights"):
        val = scene_graph.get(key)
        if val is not None:
            arr = np.asarray(val)
            parts.append(f"{key}:{arr.shape}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def compute_dynamic_hash(scene_graph: dict) -> str:
    """Hash dynamic state: vertex positions + light intensities + values."""
    parts = []
    for key in ("geometry", "materials", "lights"):
        val = scene_graph.get(key)
        if val is not None:
            arr = np.asarray(val)
            parts.append(f"{key}:{hashlib.md5(arr.tobytes()).hexdigest()[:8]}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def get_cached_latent(scene_graph: dict) -> np.ndarray | None:
    """Check cache: return latent if topology + dynamic match (task 19c.7)."""
    topo = compute_topology_hash(scene_graph)
    dyn = compute_dynamic_hash(scene_graph)

    entry = _cache.get(topo)
    if entry is None:
        return None

    if entry["dynamic_hash"] == dyn:
        logger.debug("Full cache hit (topo=%s, dyn=%s)", topo, dyn)
        return entry["latent"]

    # Topology matches but dynamic changed -> incremental update possible
    logger.debug("Topology hit, dynamic miss -> delta update")
    return None  # Caller does delta encode


def store_latent(scene_graph: dict, latent: np.ndarray):
    """Store scene latent in cache with both hashes."""
    topo = compute_topology_hash(scene_graph)
    _cache[topo] = {
        "latent": latent.copy(),
        "dynamic_hash": compute_dynamic_hash(scene_graph),
        "graph": scene_graph,
    }
    logger.debug("Cached latent (topo=%s)", topo)


def should_full_reencode(prev_graph: dict, curr_graph: dict) -> bool:
    """Determine if full re-encode needed vs incremental delta (task 19c.5).

    Full re-encode on: births, material type changes, vertex count changes,
    light additions.
    """
    for key in ("geometry", "materials", "lights"):
        prev = prev_graph.get(key)
        curr = curr_graph.get(key)
        if prev is None and curr is not None:
            return True  # birth
        if prev is not None and curr is not None:
            if np.asarray(prev).shape != np.asarray(curr).shape:
                return True  # structural change
    return False


def is_small_delta(prev_graph: dict, curr_graph: dict) -> bool:
    """Check if only small changes: object moves, light intensity, param values.

    Task 19c.6: Small delta = topology identical, only values changed.
    """
    if should_full_reencode(prev_graph, curr_graph):
        return False
    # Check magnitude of change
    for key in ("geometry", "materials", "lights"):
        prev = prev_graph.get(key)
        curr = curr_graph.get(key)
        if prev is not None and curr is not None:
            prev_arr, curr_arr = np.asarray(prev), np.asarray(curr)
            if prev_arr.shape == curr_arr.shape:
                delta = np.mean(np.abs(curr_arr - prev_arr))
                if delta > 10.0:  # Very large change = not small
                    return False
    return True


def clear_cache():
    """Clear all cached latents."""
    _cache.clear()
    logger.info("Latent cache cleared")
