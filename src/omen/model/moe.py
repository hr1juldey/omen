"""Backward compatibility shim for moe module.

Original module refactored into moe/ package for LOC compliance.
This file re-exports the public API for existing imports.
"""

from omen.model.moe.core import TileMoERouter
from omen.model.moe.experts import ExpertFFN, ExpertGroup, SharedExpert

__all__ = ["TileMoERouter", "ExpertFFN", "ExpertGroup", "SharedExpert"]

# Constants for backward compat
FINGERPRINT_DIM = 23
MATERIAL_EXPERTS = 8
LIGHT_EXPERTS = 5
GEOMETRY_EXPERTS = 5
MOTION_EXPERTS = 4
SHARED_EXPERTS = 1
TOTAL_EXPERTS = 23
