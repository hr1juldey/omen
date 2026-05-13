"""Omen Engine - In-process render pipeline with pluggable backends.

This package lives outside omen_blender so it can be iterated
without reinstalling the Blender addon. The addon imports this
via bridge.py with reload support.
"""

from omen_engine.session import OmenSession
from omen_engine.sync import OmenSync

__all__ = ["OmenSession", "OmenSync"]
