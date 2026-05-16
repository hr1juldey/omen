"""GPU memory budget management for inference and training.

Budget check (700MB inference with MLA, 1.6GB training),
CPU fallback when GPU memory is insufficient.
"""

import logging

import numpy as np

logger = logging.getLogger("omen.gpu_budget")

INFERENCE_BUDGET_MB = 700
TRAINING_BUDGET_MB = 1600
MODEL_WEIGHTS_MB = 250  # Medium tier ~16M params * 2 bytes (BF16)
MB = 1024 * 1024


def get_gpu_memory_info() -> dict:
    """Query available GPU memory. Returns dict with total/free/used_mb, backend."""
    # Try CUDA via pycuda
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        free, total = cuda.mem_get_info()
        return {"total_mb": total // MB, "free_mb": free // MB,
                "used_mb": (total - free) // MB, "backend": "cuda"}
    except (ImportError, Exception):
        pass

    # Try CUDA via torch
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return {"total_mb": total // MB, "free_mb": free // MB,
                    "used_mb": (total - free) // MB, "backend": "torch"}
    except (ImportError, Exception):
        pass

    return {"total_mb": 0, "free_mb": 0, "used_mb": 0, "backend": "none"}


def check_memory_budget(mode: str = "inference", tier: str = "medium") -> dict:
    """Check if GPU memory is sufficient for the requested operation."""
    info = get_gpu_memory_info()
    free_mb = info["free_mb"]
    tier_mult = {"fast": 0.5, "medium": 1.0, "high": 2.5}
    base = TRAINING_BUDGET_MB if mode == "training" else INFERENCE_BUDGET_MB
    budget_mb = int(base * tier_mult.get(tier, 1.0))
    sufficient = free_mb >= budget_mb or info["backend"] == "none"

    if not sufficient and free_mb > 0:
        rec = f"Need {budget_mb}MB, have {free_mb}MB. Try tier='fast' or CPU."
    elif info["backend"] == "none":
        rec = "No GPU detected — using CPU (slower but functional)."
    else:
        rec = f"GPU OK: {free_mb}MB free, need {budget_mb}MB."

    result = {"sufficient": sufficient, "free_mb": free_mb, "budget_mb": budget_mb,
              "tier": tier, "mode": mode, "recommendation": rec,
              "gpu_backend": info["backend"]}
    logger.info("Memory check [%s/%s]: %s", mode, tier, rec)
    return result


def estimate_frame_memory(height: int, width: int, tier: str = "medium") -> int:
    """Estimate GPU memory (MB) for a single frame including all buffers."""
    px = height * width
    rgba_mb = px * 16 / MB           # 4ch * 4 bytes
    aov_mb = px * 40 / MB            # 10ch * 4 bytes
    fp_mb = (height // 8) * (width // 8) * 92 / MB  # 23 * 4 bytes
    weight_mb = {"fast": 16, "medium": MODEL_WEIGHTS_MB, "high": 500}
    weights = weight_mb.get(tier, MODEL_WEIGHTS_MB)
    working_mb = 2 * (rgba_mb + aov_mb)
    total = int(rgba_mb + aov_mb + fp_mb + weights + working_mb)
    logger.debug("Frame estimate (%dx%d, %s): %dMB", width, height, tier, total)
    return total


def select_tier_for_budget(height: int, width: int) -> str:
    """Auto-select highest tier that fits in GPU memory."""
    info = get_gpu_memory_info()
    free = info["free_mb"]
    if free == 0:
        return "fast"
    for tier in ("high", "medium", "fast"):
        if estimate_frame_memory(height, width, tier) <= free:
            return tier
    return "fast"


def estimate_tile_memory(tile_size: int = 512) -> int:
    """Estimate GPU memory (MB) for single tile training step."""
    px = tile_size * tile_size
    rgba_mb = px * 16 / MB  # (B, H, W, 4) float32
    latent_mb = 1024 * 4 / MB  # Scene latent (1, 1024)
    grad_mb = rgba_mb * 2  # Forward + backward activations
    opt_mb = MODEL_WEIGHTS_MB * 2  # AdamW m/v moments
    total = int(rgba_mb + latent_mb + grad_mb + opt_mb + MODEL_WEIGHTS_MB)
    return total


def can_fit_tiles(num_tiles: int = 1, tile_size: int = 512) -> dict:
    """Check if N tiles fit in available GPU memory."""
    tile_mb = estimate_tile_memory(tile_size)
    info = get_gpu_memory_info()
    sufficient = info["free_mb"] >= tile_mb or info["backend"] == "none"
    return {
        "sufficient": sufficient,
        "per_tile_mb": tile_mb,
        "free_mb": info["free_mb"],
        "num_tiles": num_tiles,
    }
