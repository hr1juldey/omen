"""GPU acceleration for Omen integrator.

Manages:
1. GPU memory allocation for tile processing
2. Mojo/MAX kernel launch for tile fingerprint computation
3. DLPack interop between Dr.Jit CUDA tensors and Nabla GPU tensors
4. VRAM budget tracking for FP8 / MLA compression decisions

GPU kernel pipeline:
    Dr.Jit tensor → DLPack → Nabla GPU → Mojo kernel → tile fingerprint (23-dim)
"""

import logging

logger = logging.getLogger("omen_integrator.gpu")

try:
    import nabla as nb
    NABLA_AVAILABLE = True
except ImportError:
    nb = None
    NABLA_AVAILABLE = False

# VRAM budgets (bytes) per model tier
VRAM_BUDGET = {
    "fast": 500 * 1024 * 1024,     # 500MB
    "medium": 700 * 1024 * 1024,   # 700MB
    "high": 1500 * 1024 * 1024,    # 1.5GB
}


def get_gpu_memory_info():
    """Get available GPU memory.

    Returns:
        dict with total, used, free memory in bytes, or None if no GPU.
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            return {
                "total": int(parts[0]) * 1024 * 1024,
                "used": int(parts[1]) * 1024 * 1024,
                "free": int(parts[2]) * 1024 * 1024,
            }
    except Exception:
        pass
    return None


def check_vram_budget(tier="medium"):
    """Check if GPU has enough VRAM for the requested model tier.

    Args:
        tier: Model tier (fast/medium/high)

    Returns:
        True if VRAM is sufficient, False otherwise.
    """
    info = get_gpu_memory_info()
    if info is None:
        logger.warning("Cannot determine GPU memory — assuming CPU mode")
        return False

    budget = VRAM_BUDGET.get(tier, VRAM_BUDGET["medium"])
    if info["free"] >= budget:
        logger.info(
            "VRAM sufficient for %s tier: %.0fMB free, %.0fMB needed",
            tier, info["free"] / 1024 / 1024, budget / 1024 / 1024,
        )
        return True
    else:
        logger.warning(
            "VRAM insufficient for %s tier: %.0fMB free, %.0fMB needed — using CPU fallback",
            tier, info["free"] / 1024 / 1024, budget / 1024 / 1024,
        )
        return False


def supports_fp8():
    """Check if GPU supports FP8 (E4M3) operations.

    FP8 requires Ada Lovelace (RTX 4000) or Hopper (H100) or newer.

    Returns:
        True if FP8 is supported.
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            cap = float(result.stdout.strip())
            return cap >= 8.9  # Ada Lovelace = 8.9, Hopper = 9.0
    except Exception:
        pass
    return False
