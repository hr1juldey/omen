"""OmenConfig — Component switch architecture for Omen JEPA.

Thin interface: imports from subfiles for modularity.
"""

from omen.config.core import OmenConfig
from omen.config.switches import ComponentSwitches, ModeSwitches, TrainingSwitches

__all__ = ["OmenConfig", "ComponentSwitches", "TrainingSwitches", "ModeSwitches"]
