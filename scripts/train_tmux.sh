#!/usr/bin/env bash
# Omen Training — tmux launcher with log file
# Usage: bash scripts/train_tmux.sh [--scenes all --steps 10 --steps-per-frame 100 ...]
# Monitor: tail -f logs/training_*.log
# Reattach: tmux attach -t omen-training

set -euo pipefail

# Resolve project root (one dir up from this script)
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

SESSION="omen-training"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Starting omen-training in tmux"
echo "  Project:  ${PROJECT_DIR}"
echo "  Log file: ${LOG_FILE}"
echo "  Monitor:  tail -f ${LOG_FILE}"
echo "  Reattach: tmux attach -t ${SESSION}"
echo ""

tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR" "bash" \
  || { echo "Failed to create tmux session"; exit 1; }

# Small delay to let bash initialize inside tmux
sleep 0.5

# Build command string with proper spacing (plain $@ inside quotes loses spaces for send-keys)
TRAIN_CMD="cd ${PROJECT_DIR} && uv run python scripts/start_training.py --log-file ${LOG_FILE}"
for arg in "$@"; do
  TRAIN_CMD="${TRAIN_CMD} ${arg}"
done
TRAIN_CMD="${TRAIN_CMD} 2>&1; echo 'Training exited with code:' \$?"

# Send the training command via send-keys (more robust than inline command)
tmux send-keys -t "$SESSION" "$TRAIN_CMD" Enter

echo "Training started. Detaching..."
