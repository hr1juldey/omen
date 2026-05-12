"""Omen - JEPA-accelerated path tracing render engine for Mitsuba 3."""

__version__ = "0.1.0"

# Nabla import guard - graceful degradation if not installed
try:
    import nabla as nb
    NABLA_AVAILABLE = True
except ImportError:
    nb = None
    NABLA_AVAILABLE = False
