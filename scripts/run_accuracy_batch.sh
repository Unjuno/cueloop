#!/bin/bash
# Record N paired CUE/SILENCE human sessions with the fixed PTY runtime.
# Logs per-pair metrics to artifacts/feedback-loop.jsonl and re-evaluates
# H1/H2 after each batch. This is the improvement loop driver.
#
# Usage: scripts/run_accuracy_batch.sh N [duration_seconds] [batch_label]
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-8}"
DURATION="${2:-75}"
LABEL="${3:-b$(date +%H%M%S)}"
FEEDBACK="artifacts/feedback-loop.jsonl"
mkdir -p artifacts

echo "=== accuracy batch $LABEL: $N pairs x ${DURATION}s arms ==="
for i in $(seq 1 "$N"); do
  TOPIC="accuracy-${LABEL}-p${i}"
  echo
  echo "--- $LABEL pair $i/$N ($TOPIC) ---"
  if scripts/record_self_pair.sh "$TOPIC" "$DURATION" >/tmp/cueloop-batch-pair.log 2>&1; then
    grep -E 'predictions=|WARNING' /tmp/cueloop-batch-pair.log | head -3
    # Record per-pair metrics into the feedback log for the improvement loop.
    .venv/bin/python - "$TOPIC" "$FEEDBACK" <<'PY'
import glob, json, sys
from pathlib import Path
topic, feedback = sys.argv[1], Path(sys.argv[2])
dirs = sorted(glob.glob(f"logs/self-{topic}-paired-*"))
if not dirs:
    sys.exit(0)
d = Path(dirs[-1])
def load(f):
    try: return json.loads((d / f).read_text())
    except Exception: return {}
cue, sil = load("cue.json"), load("silence.json")
row = {
    "pair": d.name,
    "cue_delivered": cue.get("delivered", 0),
    "cue_outcomes": len(cue.get("outcomes", [])),
    "cue_uptake": sum(1 for o in cue.get("outcomes", []) if o.get("target_operation_seen")),
    "sil_outcomes": len(sil.get("outcomes", [])),
    "clean": len(cue.get("outcomes", [])) > 0 and len(sil.get("outcomes", [])) == 0,
}
with feedback.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"    feedback: delivered={row['cue_delivered']} uptake={row['cue_uptake']}/{row['cue_outcomes']} clean={row['clean']}")
PY
  else
    echo "pair $i failed; last lines:"
    tail -10 /tmp/cueloop-batch-pair.log
  fi
done

echo
echo "=== feedback-loop summary ==="
.venv/bin/python - "$FEEDBACK" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    sys.exit(0)
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
clean = [r for r in rows if r["clean"]]
print(f"total recorded={len(rows)} clean={len(clean)}")
if clean:
    print(f"mean delivered/pair={sum(r['cue_delivered'] for r in clean)/len(clean):.1f} "
          f"mean uptake/pair={sum(r['cue_uptake'] for r in clean)/len(clean):.2f}")
PY
echo
echo "=== accumulating clean pairs and re-evaluating H1/H2 ==="
.venv/bin/python scripts/evaluate_h1h2.py
