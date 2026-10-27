#!/bin/sh
# Overnight: Gemma 2 2B then Gemma 3 1B layer sweeps (T9 part 1), resumable.
# Keeps the Mac awake (caffeinate) while it runs. Plug in power and leave the lid
# open: closing the lid still sleeps a MacBook. Logs go outside the repo.
#   sh scripts/overnight_gemma_sweeps.sh
cd "$(dirname "$0")/.." || exit 1
LOG="$HOME/brain-llm-data/logs"
mkdir -p "$LOG"
caffeinate -is sh -c '
  .venv/bin/python -u -W ignore scripts/sweep_layers.py --model gemma-2-2b > "'"$LOG"'/sweep_gemma2.log" 2>&1 &&
  .venv/bin/python -u -W ignore scripts/sweep_layers.py --model gemma-3-1b-pt > "'"$LOG"'/sweep_gemma3.log" 2>&1
'
echo "exit $? at $(date)" >> "$LOG/sweep_gemma3.log"
