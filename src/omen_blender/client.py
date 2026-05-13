"""Subprocess client for calling Omen CLI from Blender.

Launches the Omen render process via pixi in the Omen project directory.
Passes scene JSON path, reads back the rendered result image.
"""

import json
import logging
import os
import subprocess
import sys

logger = logging.getLogger("omen_blender.client")

DEFAULT_OMEN_PATH = os.path.expanduser(
    "~/Documents/Projects/MOJO/Cycles_mojo/omen"
)


def find_omen_path(scene_settings):
    """Resolve the Omen project root path.

    Checks: scene setting > OMEN_PATH env > default path.
    Returns None if not found.
    """
    candidates = [
        getattr(scene_settings, "omen_path", ""),
        os.environ.get("OMEN_PATH", ""),
        DEFAULT_OMEN_PATH,
    ]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "pixi.toml")):
            return os.path.abspath(path)
    return None


def render_scene(omen_path, scene_json, output_path, settings):
    """Call Omen CLI to render a scene.

    Args:
        omen_path: Root of Omen project (contains pixi.toml).
        scene_json: Path to exported scene JSON.
        output_path: Where to write the result image (.exr or .png).
        settings: OmenSettings PropertyGroup with mode, tier, spp, etc.

    Returns:
        True if render succeeded, False otherwise.
    """
    cmd = _build_command(omen_path, scene_json, output_path, settings)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(omen_path, "src")

    logger.info("Launching Omen: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=omen_path,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        if result.returncode != 0:
            logger.error("Omen failed (exit %d): %s", result.returncode, result.stderr)
            return False
        if not os.path.isfile(output_path):
            logger.error("Output file not created: %s", output_path)
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Omen render timed out after 600s")
        return False
    except FileNotFoundError:
        logger.error("pixi not found in PATH")
        return False


def _build_command(omen_path, scene_json, output_path, settings):
    """Build the pixi subprocess command."""
    mode = getattr(settings, "mode", "denoise")
    tier = getattr(settings, "tier", "medium")
    spp = getattr(settings, "spp", 4)
    max_depth = getattr(settings, "max_depth", 8)
    use_gpu = getattr(settings, "use_gpu", True)

    return [
        "pixi", "run", "python", "-m", "omen.cli_render",
        "--scene", scene_json,
        "--output", output_path,
        "--mode", mode,
        "--tier", tier,
        "--spp", str(spp),
        "--max-depth", str(max_depth),
        "--gpu" if use_gpu else "--no-gpu",
    ]


def check_omen_available(omen_path):
    """Quick check if Omen environment is functional.

    Returns (available: bool, message: str).
    """
    if omen_path is None:
        return False, "Omen project path not found. Set in render settings."
    cli_path = os.path.join(omen_path, "src", "omen", "cli_render.py")
    if not os.path.isfile(cli_path):
        return False, f"Omen CLI not found at {cli_path}"
    return True, f"Omen found at {omen_path}"
