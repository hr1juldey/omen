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

# Verify Mitsuba installation
echo "🔍 Verifying Mitsuba installation..."
pixi run python -c "import mitsuba as mi; print(f'✅ Mitsuba {mi.__version__}')"
echo ""

echo "✅ Setup complete!"
echo ""
echo "🚀 Quick test:"
echo "   pixi run python -c \"import mitsuba as mi; print('Omen ready!')\""
echo ""
