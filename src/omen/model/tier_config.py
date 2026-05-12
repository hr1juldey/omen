"""Tier configuration for Omen JEPA model.

Controls MoE routing behavior per model tier:
- Fast (~4M params): No MoE, single shared expert for speed
- Medium (~16M params): MoE at bottleneck only, top-2 material + top-1 motion
- High (~64M params): MoE at bottleneck + encoder, top-3 material + top-1 motion

From design Decision 1: Scene-Aware U-Net Denoiser with JEPA Conditioning (3 Tiers).
"""

import logging

# Expert counts — must match omen.model.moe definitions
MATERIAL_EXPERTS = 8
LIGHT_EXPERTS = 5
GEOMETRY_EXPERTS = 5
MOTION_EXPERTS = 4
TOTAL_EXPERTS = MATERIAL_EXPERTS + LIGHT_EXPERTS + GEOMETRY_EXPERTS + MOTION_EXPERTS + 1

logger = logging.getLogger("omen.model.tier_config")

TIER_PROFILES = {
    "fast": {
        "use_moe": False,
        "material_top_k": 0,
        "light_top_k": 0,
        "geometry_top_k": 0,
        "motion_top_k": 0,
        "encoder_moe": False,
        "description": "No MoE — shared expert only, max speed",
    },
    "medium": {
        "use_moe": True,
        "material_top_k": 2,
        "light_top_k": 1,
        "geometry_top_k": 1,
        "motion_top_k": 1,
        "encoder_moe": False,
        "description": "MoE bottleneck only, top-2 material + top-1 others",
    },
    "high": {
        "use_moe": True,
        "material_top_k": 3,
        "light_top_k": 2,
        "geometry_top_k": 2,
        "motion_top_k": 1,
        "encoder_moe": True,
        "description": "MoE bottleneck + encoder, top-3 material + top-2 light/geo",
    },
}

LATENT_DIMS = {"fast": 192, "medium": 256, "high": 512}
PARAM_COUNTS = {"fast": "4M", "medium": "16M", "high": "64M"}


def get_tier_config(tier: str) -> dict:
    """Get model configuration for the specified tier.

    Args:
        tier: One of 'fast', 'medium', 'high'

    Returns:
        Configuration dict with MoE routing parameters
    """
    if tier not in TIER_PROFILES:
        raise ValueError(f"Unknown tier '{tier}'. Use: fast, medium, high")
    config = TIER_PROFILES[tier].copy()
    config["latent_dim"] = LATENT_DIMS[tier]
    config["param_count"] = PARAM_COUNTS[tier]
    config["total_experts"] = TOTAL_EXPERTS
    return config


def log_tier_config(tier: str):
    """Log the active tier configuration."""
    cfg = get_tier_config(tier)
    logger.info(
        "Tier '%s' (%s params, latent=%d): %s",
        tier,
        cfg["param_count"],
        cfg["latent_dim"],
        cfg["description"],
    )
    if cfg["use_moe"]:
        logger.info(
            "MoE routing: material top-%d, light top-%d, geo top-%d, motion top-%d",
            cfg["material_top_k"],
            cfg["light_top_k"],
            cfg["geometry_top_k"],
            cfg["motion_top_k"],
        )
