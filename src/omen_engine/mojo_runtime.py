"""Mojo runtime setup — LD_LIBRARY_PATH, modular version check, ctypes loader.

Called once at addon enable. Must run before any Mojo .so loading.
"""

import ctypes
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_KERNELS_STEM = "omen_kernels"


def setup_ld_library_path() -> str | None:
    """Find modular/lib/ and add to LD_LIBRARY_PATH. Returns lib dir or None."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("modular")
        if not spec or not spec.submodule_search_locations:
            log.warning("modular package not found")
            return None

        for search_path in spec.submodule_search_locations:
            lib_dir = os.path.join(search_path, "lib")
            if os.path.isdir(lib_dir):
                existing = os.environ.get("LD_LIBRARY_PATH", "")
                if lib_dir not in existing:
                    os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + existing
                log.info("LD_LIBRARY_PATH set: %s", lib_dir)
                return lib_dir
    except Exception as exc:
        log.error("Failed to setup LD_LIBRARY_PATH: %s", exc)
    return None


def check_modular_nightly() -> bool:
    """Verify modular nightly (version contains .dev). Returns True if OK."""
    try:
        from importlib.metadata import version
        mod_ver = version("modular")
        if ".dev" not in mod_ver:
            log.error(
                "modular %s is stable, not nightly. "
                "Install: uv add --pre modular "
                "--index https://whl.modular.com/nightly/simple/ "
                "--prerelease allow",
                mod_ver,
            )
            return False
        log.info("modular nightly %s detected", mod_ver)
        return True
    except Exception:
        log.warning("Cannot check modular version")
        return False


def load_kernels_so(search_paths: list[str] | None = None) -> ctypes.CDLL | None:
    """Load omen_kernels.so via ctypes. Returns handle or None."""
    candidates: list[str] = []
    if search_paths:
        candidates.extend(search_paths)

    # Check relative to this file
    this_dir = Path(__file__).parent
    candidates.append(str(this_dir / "kernels"))
    candidates.append(str(this_dir.parent / "omen" / "kernels"))

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    candidates.extend(ld_path.split(":"))

    for base in candidates:
        if not base:
            continue
        so_path = os.path.join(base, f"{_KERNELS_STEM}.so")
        if os.path.isfile(so_path):
            try:
                lib = ctypes.CDLL(so_path)
                log.info("Loaded Mojo kernels: %s", so_path)
                return lib
            except OSError as exc:
                log.error("Failed to load %s: %s", so_path, exc)

    log.warning(
        "omen_kernels.so not found. Compile with: "
        "mojo build src/omen/kernels/omen_kernels.mojo "
        "--emit shared-lib -o omen_kernels.so"
    )
    return None


def get_kernel_func(
    lib: ctypes.CDLL, name: str, restype: Any, argtypes: list[Any],
) -> Any:
    """Get a typed ctypes function from the loaded .so."""
    try:
        func = getattr(lib, name)
        func.restype = restype
        func.argtypes = argtypes
        return func
    except AttributeError:
        log.warning("Kernel function '%s' not found in .so", name)
        return None
