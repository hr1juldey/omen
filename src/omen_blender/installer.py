"""Auto-installer for Omen dependencies on first addon enable.

Installs into Blender's bundled Python site-packages:
  - modular (nightly)
  - nabla-ml
  - mitsuba
  - numpy

Uses subprocess to call Blender's pip, never touches system Python.
"""

import logging
import subprocess
import sys

log = logging.getLogger(__name__)

_PACKAGES = [
    ("modular", "--pre", "--index",
     "https://whl.modular.com/nightly/simple/", "--prerelease", "allow"),
    ("nabla-ml",),
    ("mitsuba",),
    ("numpy",),
]

_MARKER = "omen_deps_installed"


def is_installed() -> bool:
    """Check if all required packages are importable."""
    for pkg in ("mitsuba", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            return False
    try:
        from importlib.metadata import version
        mod_ver = version("modular")
        if ".dev" not in mod_ver:
            return False
    except Exception:
        return False
    return True


def install_dependencies(python_path: str | None = None) -> bool:
    """Install all dependencies via pip. Returns True on success."""
    pip = python_path or sys.executable
    pip = f"{pip} -m pip"

    for entry in _PACKAGES:
        pkg_name = entry[0]
        args = list(entry)
        cmd = f"{pip} install --quiet {' '.join(args)}"
        log.info("Installing %s...", pkg_name)
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                log.error("Failed to install %s: %s", pkg_name, result.stderr)
                return False
        except subprocess.TimeoutExpired:
            log.error("Timeout installing %s", pkg_name)
            return False

    log.info("All dependencies installed successfully")
    return True
