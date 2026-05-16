#!/usr/bin/env bash
# Launch tiled GPU training in a tmux session.
# Usage: bash scripts/start_training.sh [scene] [resolution] [steps]
# Attach: tmux attach -t omen_training

set -euo pipefail

SESSION="omen_training"
SCENE="${1:-cornell}"
RESOLUTION="${2:-512x512}"
STEPS="${3:-10}"

cd "$(dirname "$0")/.."

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" "
    source .venv/bin/activate
    export NABLA_DEFAULT_DEVICE=gpu
    python scripts/start_training.py --scene $SCENE --resolution $RESOLUTION --steps $STEPS
    echo 'Training finished. Press Enter to close.'
    read
"

echo "Training started in tmux session '$SESSION'"
echo "  Scene: $SCENE | Resolution: $RESOLUTION | Steps: $STEPS"
echo "  Attach: tmux attach -t $SESSION"
