"""Backward compatibility shim for trainer module.

Original module refactored into trainer/ package for LOC compliance.
This file re-exports the public API for existing imports.
"""

# Re-export public API
from omen.training.trainer.gradient import _clip_grad_norm
from omen.training.trainer.optimizers import COMPONENT_LRS
from omen.training.trainer import OmenTrainer

__all__ = ["OmenTrainer", "_clip_grad_norm", "COMPONENT_LRS"]

# Training config constants (for backward compat)
DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_GRADIENT_CLIP = 1.0
SIGREG_LAMBDA = 0.09
