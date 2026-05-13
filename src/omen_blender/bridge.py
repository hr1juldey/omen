"""Bridge between Blender addon and omen_engine.

Supports reload: call reload_engine() after editing omen_engine
code, then re-render without restarting Blender.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omen_engine.session import OmenSession

log = logging.getLogger(__name__)

_session: OmenSession | None = None


def get_session() -> OmenSession:
    """Return the current OmenSession, creating one if needed."""
    global _session
    if _session is None:
        from omen_engine.session import OmenSession
        _session = OmenSession()
        log.info("OmenSession created")
    return _session


def reload_engine() -> None:
    """Reload omen_engine modules for live development."""
    global _session
    import omen_engine
    import omen_engine.backends
    import omen_engine.backends.mitsuba_backend
    import omen_engine.session
    import omen_engine.sync

    importlib.reload(omen_engine.backends)
    importlib.reload(omen_engine.backends.mitsuba_backend)
    importlib.reload(omen_engine.sync)
    importlib.reload(omen_engine.session)
    importlib.reload(omen_engine)

    _session = None
    log.info("Omen engine modules reloaded")


def destroy_session() -> None:
    """Tear down the current session."""
    global _session
    _session = None
    log.info("OmenSession destroyed")
