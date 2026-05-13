#!/usr/bin/env bash
# Omen Render Engine - Setup (uv-based, verified across Python 3.11-3.14)
#
# Usage:
#   ./setup.sh              # uv workflow (recommended)
#   ./setup.sh --pixi       # pixi workflow (alternative)
#
# For Blender addon install, see: scripts/build_addon.py

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="${OMEN_PYTHON:-3.12}"

echo "Omen Render Engine - Setup"
echo "=========================="
echo ""

# --- uv workflow (default) ---
if [ "$1" != "--pixi" ]; then
    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if [ ! -d ".venv" ]; then
        echo "Creating uv venv (Python $PYTHON_VERSION)..."
        uv venv --python "$PYTHON_VERSION"
    fi

    source .venv/bin/activate

    echo "Installing modular nightly + nabla-ml..."
    uv pip install --pre modular \
        --index https://whl.modular.com/nightly/simple/ \
        --prerelease allow

    uv pip install --pre nabla-ml \
        --index https://whl.modular.com/nightly/simple/ \
        --prerelease allow

    echo "Installing mitsuba + drjit..."
    uv pip install mitsuba drjit numpy

    echo "Installing omen integrator..."
    uv pip install -e "$PROJECT_ROOT/integrator-package" --no-deps

    echo "Installing omen engine..."
    uv pip install -e "$PROJECT_ROOT" --no-deps

    echo ""
    echo "Verifying..."
    python -c "
import mitsuba as mi
from omen_integrator import register
register()
print('  mitsuba + omen_integrator OK')
from omen_engine import OmenSession, OmenSync
print('  omen_engine OK')
"

    deactivate
    echo ""
    echo "Setup complete. Activate with: source .venv/bin/activate"
    echo "Build Blender addon: python scripts/build_addon.py"
    exit 0
fi

# --- pixi workflow (alternative) ---
if ! command -v pixi &> /dev/null; then
    echo "Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi

echo "Creating pixi environment..."
pixi install

echo "Installing mitsuba..."
pixi run pip install mitsuba drjit numpy --quiet

echo "Installing omen integrator..."
cd "$PROJECT_ROOT/integrator-package"
pixi run uv pip install -e . --prefix "$PROJECT_ROOT/.pixi/envs/default" > /dev/null 2>&1
cd "$PROJECT_ROOT"

echo "Installing omen engine..."
pixi run pip install -e . --no-deps > /dev/null 2>&1

echo ""
echo "Verifying..."
pixi run python -c "
import mitsuba as mi
from omen_integrator import register
register()
print('  mitsuba + omen_integrator OK')
from omen_engine import OmenSession, OmenSync
print('  omen_engine OK')
" 2>&1 | grep -v WARN | grep -v deprecated

echo ""
echo "Setup complete. Run with: pixi run python your_script.py"
