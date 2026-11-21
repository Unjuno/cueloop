#!/bin/bash
# Record N paired CUE/SILENCE human sessions with the fixed PTY runtime,
# then re-evaluate H1/H2. Each pair is ~2 * DURATION seconds.
# Usage: scripts/run_accuracy_batch.sh N [duration_seconds]
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-8}"
DURATION="${2:-90}"

echo "=== accuracy batch: $N pairs x ${DURATION}s arms ==="
for i in $(seq 1 "$N"); do
  echo
  echo "--- batch pair $i/$N ---"
  scripts/record_self_pair.sh "accuracy-batch${i}" "$DURATION" >/tmp/cueloop-batch-pair.log 2>&1 || {
    echo "pair $i failed; last lines:"
    tail -10 /tmp/cueloop-batch-pair.log
    continue
  }
  grep -E 'predictions=|WARNING' /tmp/cueloop-batch-pair.log | head -3
done

echo
echo "=== accumulating clean pairs and re-evaluating H1/H2 ==="
.venv/bin/python scripts/evaluate_h1h2.py
