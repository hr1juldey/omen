"""Scene hashing and model caching for per-scene fine-tuning.

Topology-based scene hashing, base model download,
scene-specific fine-tuned model cache at ~/.cache/omen/models/scenes/<hash>/.
"""

import hashlib
import json
import logging
import os
import urllib.request

import numpy as np

logger = logging.getLogger("omen.scene_cache")

CACHE_DIR = os.path.expanduser("~/.cache/omen")
MODELS_DIR = os.path.join(CACHE_DIR, "models")
SCENES_DIR = os.path.join(MODELS_DIR, "scenes")
BASE_MODEL_URL = (
    "https://github.com/mojocycles/omen/releases/download/v0.1/base_v0.omen"
)
BASE_MODEL_DIR = os.path.join(MODELS_DIR, "base_v0")


def ensure_cache_dirs():
    """Create cache directories if they don't exist."""
    for d in (CACHE_DIR, MODELS_DIR, SCENES_DIR):
        os.makedirs(d, exist_ok=True)


def compute_topology_hash(scene_graph: dict) -> str:
    """Hash scene topology: shapes + material/light types (not values)."""
    parts = []

    # Geometry topology: vertex count + face count per mesh
    geom = scene_graph.get("geometry", None)
    if geom is not None:
        arr = np.asarray(geom) if not isinstance(geom, np.ndarray) else geom
        parts.append(f"geo:{arr.shape}")

    # Material types (not values): just count and type identifiers
    mats = scene_graph.get("materials", None)
    if mats is not None:
        arr = np.asarray(mats) if not isinstance(mats, np.ndarray) else mats
        parts.append(f"mat:{arr.shape}")

    # Light types: count + emitter classifications
    lights = scene_graph.get("lights", None)
    if lights is not None:
        arr = np.asarray(lights) if not isinstance(lights, np.ndarray) else lights
        parts.append(f"light:{arr.shape}")

    # Camera: just presence
    if "camera" in scene_graph:
        parts.append("cam:present")

    topo_str = "|".join(parts)
    return hashlib.sha256(topo_str.encode()).hexdigest()[:16]


def compute_dynamic_hash(scene_graph: dict) -> str:
    """Hash dynamic state: positions + intensities + values for full cache hit."""
    parts = []
    for key in ("geometry", "materials", "lights"):
        val = scene_graph.get(key)
        if val is not None:
            arr = np.asarray(val)
            parts.append(f"{key}:{arr.tobytes().hex()[:128]}")
    dyn_str = "|".join(parts)
    return hashlib.sha256(dyn_str.encode()).hexdigest()[:16]


def get_scene_cache_dir(scene_graph: dict) -> str:
    """Get per-scene cache directory based on topology hash."""
    topo_hash = compute_topology_hash(scene_graph)
    return os.path.join(SCENES_DIR, topo_hash)


def save_scene_model(scene_graph: dict, checkpoint_path: str, metadata: dict):
    """Cache a fine-tuned model for a specific scene topology."""
    ensure_cache_dirs()
    cache_dir = get_scene_cache_dir(scene_graph)
    os.makedirs(cache_dir, exist_ok=True)

    # Copy checkpoint files into scene cache
    for fname in ("weights.npz", "optimizer.npz", "metadata.json"):
        src = os.path.join(checkpoint_path, fname)
        dst = os.path.join(cache_dir, fname)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)

    # Write scene-specific metadata
    dyn_hash = compute_dynamic_hash(scene_graph)
    scene_meta = {
        "topology_hash": compute_topology_hash(scene_graph),
        "dynamic_hash": dyn_hash,
        "source": checkpoint_path,
    }
    scene_meta.update(metadata)
    with open(os.path.join(cache_dir, "scene_meta.json"), "w") as f:
        json.dump(scene_meta, f, indent=2)
    logger.info("Scene model cached: %s", cache_dir)


def load_scene_model(scene_graph: dict):
    """Load fine-tuned model for a scene if cached.

    Returns: checkpoint directory path or None
    """
    cache_dir = get_scene_cache_dir(scene_graph)
    weights = os.path.join(cache_dir, "weights.npz")
    if not os.path.exists(weights):
        return None
    logger.info("Found scene-specific model: %s", cache_dir)
    return cache_dir


def download_base_model(force: bool = False):
    """Download base model on first use.

    Downloads to ~/.cache/omen/models/base_v0/.
    Skips if already downloaded unless force=True.
    """
    ensure_cache_dirs()
    meta_path = os.path.join(BASE_MODEL_DIR, "metadata.json")
    if os.path.exists(meta_path) and not force:
        logger.info("Base model already exists at %s", BASE_MODEL_DIR)
        return BASE_MODEL_DIR

    logger.info("Downloading base model from %s ...", BASE_MODEL_URL)
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)
    try:
        dest = os.path.join(BASE_MODEL_DIR, "base_v0.omen")
        urllib.request.urlretrieve(BASE_MODEL_URL, dest)
        meta = {"source": BASE_MODEL_URL, "type": "base"}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Base model downloaded to %s", BASE_MODEL_DIR)
        return BASE_MODEL_DIR
    except Exception as exc:
        logger.error("Base model download failed: %s", exc)
        logger.info("Train your own with: omen train")
        return None
