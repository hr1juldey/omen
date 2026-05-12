#!/bin/bash
# Development setup script for Omen Render Engine addon
# This creates a symlink from Blender's addons directory to the omen folder

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_VERSION="5.1"
BLENDER_ADDONS_DIR="$HOME/.config/blender/${BLENDER_VERSION}/scripts/addons"
ADDON_NAME="omen"

echo "Omen Render Engine - Development Setup"
echo "======================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Blender addons: $BLENDER_ADDONS_DIR"
echo ""

# Create addons directory if it doesn't exist
mkdir -p "$BLENDER_ADDONS_DIR"

# Remove existing symlink if present
if [ -L "$BLENDER_ADDONS_DIR/$ADDON_NAME" ]; then
    echo "Removing existing symlink..."
    rm "$BLENDER_ADDONS_DIR/$ADDON_NAME"
fi

# Create new symlink
echo "Creating symlink: $BLENDER_ADDONS_DIR/$ADDON_NAME -> $PROJECT_ROOT"
ln -s "$PROJECT_ROOT" "$BLENDER_ADDONS_DIR/$ADDON_NAME"

echo ""
echo "✓ Setup complete!"
echo ""
echo "To use:"
echo "  1. Restart Blender"
echo "  2. Go to Edit > Preferences > Add-ons"
echo "  3. Search for 'Omen' and enable the addon"
echo "  4. Set Render Engine to 'Omen' in Render Properties"
echo ""
echo "For development changes:"
echo "  - Blender will reload the addon automatically"
echo "  - Or press F8 in the Script workspace to reload scripts"
