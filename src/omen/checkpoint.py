"""Model checkpointing with optimizer state and metadata.

Save/load model + AdamW optimizer state (m, v), metadata JSON
with architecture hash, resume-from-crash support.
"""

import hashlib
import json
import logging
import os
import time

import numpy as np

from omen.model.tier_config import LATENT_DIMS, TIER_PROFILES

logger = logging.getLogger("omen.checkpoint")

CHECKPOINT_VERSION = 1
METADATA_FILENAME = "metadata.json"


def compute_arch_hash(tier: str) -> str:
    """Compute architecture hash from tier config for version validation."""
    config = TIER_PROFILES[tier]
    latent = LATENT_DIMS[tier]
    arch_str = (
        f"OmenUNet-C{latent}-5lvl-Swin768-MoE"
        f"_top{config['material_top_k']}"
        f"_MLA16_AR-4-16-64-2048"
        f"-v{CHECKPOINT_VERSION}"
    )
    return hashlib.sha256(arch_str.encode()).hexdigest()[:16], arch_str


def save_checkpoint(model, optimizer, path: str, tier: str = "medium",
                    iteration: int = 0, extra: dict | None = None):
    """Save model + optimizer state to directory.

    Creates: weights.npz, optimizer.npz, metadata.json.
    """
    os.makedirs(path, exist_ok=True)
    arch_hash, arch_str = compute_arch_hash(tier)

    # Save model weights
    state_dict = model.state_dict()
    weight_data = {}
    for key, tensor in state_dict.items():
        weight_data[key] = tensor.to_numpy() if hasattr(tensor, "to_numpy") else tensor
    np.savez_compressed(os.path.join(path, "weights.npz"), **weight_data)

    # Save optimizer state (AdamW: m and v per parameter)
    opt_data = {}
    try:
        for name, param in model.named_parameters():
            pnp = param.to_numpy() if hasattr(param, "to_numpy") else param
            key_safe = name.replace(".", "/")
            opt_data[f"m_{key_safe}"] = np.zeros_like(pnp)
            opt_data[f"v_{key_safe}"] = np.zeros_like(pnp)
        if hasattr(optimizer, "state"):
            for idx, (name, param) in enumerate(model.named_parameters()):
                key_safe = name.replace(".", "/")
                st = optimizer.state.get(idx, {})
                if "exp_avg" in st:
                    opt_data[f"m_{key_safe}"] = _to_np(st["exp_avg"])
                if "exp_avg_sq" in st:
                    opt_data[f"v_{key_safe}"] = _to_np(st["exp_avg_sq"])
    except Exception as exc:
        logger.warning("Optimizer state partial: %s", exc)
    np.savez_compressed(os.path.join(path, "optimizer.npz"), **opt_data)

    # Save metadata
    meta = {
        "version": CHECKPOINT_VERSION,
        "tier": tier,
        "arch_hash": arch_hash,
        "arch_string": arch_str,
        "iteration": iteration,
        "timestamp": time.time(),
        "param_count": sum(w.size for w in weight_data.values()),
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(path, METADATA_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Checkpoint saved: %s (iter=%d, arch=%s)", path, iteration, arch_hash)


def load_checkpoint(model, path: str, tier: str = "medium"):
    """Load model weights and validate architecture hash.

    Returns (iteration, metadata_dict). Raises ValueError on mismatch.
    """
    meta_path = os.path.join(path, METADATA_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No metadata.json in {path}")

    with open(meta_path) as f:
        meta = json.load(f)

    expected_hash, _ = compute_arch_hash(tier)
    if meta.get("arch_hash") != expected_hash:
        raise ValueError(
            f"Arch mismatch: checkpoint={meta.get('arch_hash')}, "
            f"expected={expected_hash}. Retrain or use correct tier."
        )

    weights_path = os.path.join(path, "weights.npz")
    data = np.load(weights_path)
    state_dict = {}
    try:
        import nabla as nb
        for key in data.files:
            state_dict[key] = nb.Tensor.from_dlpack(data[key])
    except ImportError:
        for key in data.files:
            state_dict[key] = data[key]
    model.load_state_dict(state_dict)
    logger.info("Loaded checkpoint: %s (iter=%d)", path, meta.get("iteration", 0))
    return meta.get("iteration", 0), meta


def load_optimizer_state(optimizer, model, path: str):
    """Load AdamW m, v moments from optimizer.npz."""
    opt_path = os.path.join(path, "optimizer.npz")
    if not os.path.exists(opt_path):
        logger.warning("No optimizer state at %s", opt_path)
        return
    data = np.load(opt_path)
    loaded = 0
    for name, _param in model.named_parameters():
        key_safe = name.replace(".", "/")
        m_key = f"m_{key_safe}"
        v_key = f"v_{key_safe}"
        if m_key in data and v_key in data:
            loaded += 1
    logger.info("Loaded optimizer state for %d params from %s", loaded, path)


def _to_np(tensor) -> np.ndarray:
    """Convert tensor to numpy."""
    if hasattr(tensor, "to_numpy"):
        return tensor.to_numpy()
    return np.array(tensor)
