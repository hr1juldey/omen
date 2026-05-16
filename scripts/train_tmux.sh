#!/usr/bin/env bash
# Omen Training — tmux launcher with log file
# Usage: bash scripts/train_tmux.sh [--scenes all --steps 10 --steps-per-frame 100 ...]
# Monitor: tail -f logs/training_*.log
# Reattach: tmux attach -t omen-training

set -euo pipefail

SESSION="omen-training"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Starting omen-training in tmux"
echo "  Log file: ${LOG_FILE}"
echo "  Monitor:  tail -f ${LOG_FILE}"
echo "  Reattach: tmux attach -t ${SESSION}"
echo ""

tmux new-session -d -s "$SESSION" \
  "uv run python scripts/start_training.py --log-file ${LOG_FILE} $@ 2>&1 | tee -a ${LOG_FILE}" \
  || { echo "Failed to create tmux session"; exit 1; }

echo "Training started. Detaching..."
