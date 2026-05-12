#!/usr/bin/env bash
# Omen Render Engine - One-Click Setup
# Based on findings from BPY_MITSUBA_FINDINGS.md
#
# Usage:
#   ./setup.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔧 Omen Render Engine - Setup"
echo "================================"
echo ""

# Check if pixi is installed
if ! command -v pixi &> /dev/null; then
    echo "📦 Installing pixi package manager..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi

# Create pixi environment (Python 3.14)
echo "📦 Creating pixi environment..."
pixi install
echo ""

# Install Mitsuba 3 (from PyPI - proven working)
echo "📦 Installing Mitsuba 3 renderer..."
pixi run pip install mitsuba drjit --quiet
echo ""

# Install Omen integrator to pixi environment
echo "📦 Installing Omen integrator..."
cd "$PROJECT_ROOT/integrator-package"
pixi run uv pip install -e . --prefix "$PROJECT_ROOT/.pixi/envs/default" > /dev/null 2>&1
cd "$PROJECT_ROOT"
echo ""

# Verify installation
echo "🔍 Verifying installation..."
pixi run python -c "
import mitsuba as mi
from omen_integrator import register
register()
print('✅ Omen integrator ready!')
" 2>&1 | grep -v WARN | grep -v deprecated
echo ""

echo "✅ Setup complete!"
echo ""
echo "🚀 Usage:"
echo "   pixi run python your_script.py"
echo ""
echo "   In your Python code:"
echo "   import mitsuba as mi"
echo "   from omen_integrator import register"
echo "   register()  # Registers 'omen' integrator"
echo ""
