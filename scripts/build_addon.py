#!/usr/bin/env python3
"""Build script for omen_blender.zip — distributable Blender addon.

Produces a ZIP containing:
  omen_blender/         — addon wrapper (thin, stays in Blender)
  omen_engine/          — engine code (reloadable without addon reinstall)
  omen_kernels.so       — pre-compiled Mojo GPU kernels (optional)

Usage:
  python scripts/build_addon.py [--skip-mojo] [--output /path/to/zip]
"""

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def compile_mojo_kernels(staging: Path) -> bool:
    """Compile omen_kernels.so from Mojo source."""
    mojo_src = SRC / "omen" / "kernels"
    main_file = mojo_src / "omen_kernels.mojo"

    if not main_file.exists():
        print(f"  SKIP: {main_file} not found")
        return False

    out_so = staging / "omen_engine" / "omen_kernels.so"
    try:
        subprocess.run(
            ["mojo", "build", str(main_file), "--emit", "shared-lib",
             "-o", str(out_so)],
            check=True, capture_output=True, text=True,
        )
        print(f"  OK: compiled {out_so}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  WARN: Mojo compilation failed: {exc}")
        return False


def build_zip(output_path: Path, skip_mojo: bool) -> Path:
    """Build the distributable ZIP."""
    staging = Path(tempfile.mkdtemp(prefix="omen_build_"))
    addon_dir = staging / "omen_blender"
    engine_dir = staging / "omen_engine"

    try:
        # Copy addon wrapper
        addon_src = SRC / "omen_blender"
        for f in addon_src.iterdir():
            if f.suffix == ".py":
                shutil.copy2(f, addon_dir / f.name)

        # Copy engine module
        engine_src = SRC / "omen_engine"
        shutil.copytree(engine_src, engine_dir, dirs_exist_ok=True)

        # Copy __init__.py for omen_engine
        (addon_dir / "__init__.py").write_text(
            (addon_src / "__init__.py").read_text()
        )

        # Compile Mojo kernels
        if not skip_mojo:
            compile_mojo_kernels(staging)

        # Create ZIP
        zip_path = output_path / "omen_blender.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root_dir in [addon_dir, engine_dir]:
                for f in root_dir.rglob("*"):
                    if f.is_file():
                        arcname = f.relative_to(staging)
                        zf.write(f, arcname)
                        print(f"  + {arcname}")

        print(f"\nBuilt: {zip_path}")
        return zip_path

    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build omen_blender.zip")
    parser.add_argument("--skip-mojo", action="store_true",
                        help="Skip Mojo .so compilation")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "dist",
                        help="Output directory for ZIP")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    build_zip(args.output, args.skip_mojo)


if __name__ == "__main__":
    main()
