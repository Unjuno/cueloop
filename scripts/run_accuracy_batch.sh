#!/bin/bash
# Record N paired CUE/SILENCE human sessions with the fixed PTY runtime.
# IMPORTANT: each pair uses a RANDOMIZED condition (seed, assignment, speech
# db, duration, prompt style) and a RANDOM inter-pair gap, so repeated runs
# do not sample one fixed condition (anti-overfitting / diversity).
# Per-pair metrics + conditions are logged to artifacts/feedback-loop.jsonl.
# Usage: scripts/run_accuracy_batch.sh N [min_duration] [batch_label]
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-8}"
MIN_DURATION="${2:-60}"
LABEL="${3:-b$(date +%H%M%S)}"
FEEDBACK="artifacts/feedback-loop.jsonl"
# Optional anti-over-insertion gate (set GATE_WEIGHTS=artifacts/operation-weights.json)
GATE_ENV=()
if [ -n "${GATE_WEIGHTS:-}" ]; then
  GATE_ENV=(CUELOOP_OPERATION_WEIGHTS="$GATE_WEIGHTS" CUELOOP_MIN_OP_WEIGHT="${GATE_FLOOR:-0.5}")
  echo "=== operation gate ENABLED: $GATE_WEIGHTS floor=${GATE_FLOOR:-0.5} ==="
fi
if [ -n "${FATIGUE:-}" ]; then
  GATE_ENV+=(CUELOOP_FATIGUE="$FATIGUE")
  echo "=== fatigue penalty override: $FATIGUE ==="
fi
mkdir -p artifacts

# Prompt styles: different operation-word sets and speaking instructions.
PROMPTS_A=(
  "これから話してください。結論は、なぜなら、逆に、具体例は、といった言葉を使いながら、日本語で話し続けてください。"
  "お話しください。まず前提を述べ、次に根拠を挙げ、最後に結論を言う形で、日本語で話してください。"
  "この話題について考え声に出してください。つまり、だから、例えば、といった言葉を自然に使ってください。"
  "日本語で自由に話してください。違う見方をするときは逆に、理由を言うときはなぜなら、と声に出してください。"
)
PROMPTS_B=(
  "もう一度、同じ話題で話してください。先ほどと同じように、結論は、なぜなら、逆に、といった言葉を使ってください。"
  "同じ話題をもう一度。今度は前提、根拠、結論の順で、声に出して話してください。"
  "先ほどと同じ内容を、今度はつまり、だから、例えば、を使って話してください。"
  "同じ話題を繰り返してください。逆に、なぜなら、を意識しながら日本語で話してください。"
)

echo "=== accuracy batch $LABEL: $N DIVERSE pairs (min ${MIN_DURATION}s arms) ==="
for i in $(seq 1 "$N"); do
  # Randomize condition for this pair.
  SEED=$(( (RANDOM % 97) + 1 ))
  # Only assigners that work without an explicit --cue (fixed requires one).
  case $(( RANDOM % 2 )) in
    0) ASSIGNMENT=canonical ;;
    *) ASSIGNMENT=random-operation ;;
  esac
  case $(( RANDOM % 4 )) in
    0) SPEECH_DB=-45 ;;
    1) SPEECH_DB=-50 ;;
    2) SPEECH_DB=-55 ;;
    *) SPEECH_DB=-48 ;;
  esac
  DURATION=$(( MIN_DURATION + (RANDOM % 45) ))   # 60..104s
  PIDX=$(( RANDOM % ${#PROMPTS_A[@]} ))
  PA="${PROMPTS_A[$PIDX]}"
  PB="${PROMPTS_B[$PIDX]}"
  TOPIC="accuracy-${LABEL}-p${i}"

  # Random inter-pair gap (60..240s) breaks the metronomic 3-minute timing.
  if [ "$i" -gt 1 ]; then
    GAP=$(( 60 + (RANDOM % 180) ))
    echo "--- resting ${GAP}s before pair $i (anti-overfitting jitter) ---"
    sleep "$GAP"
  fi

  echo
  echo "--- $LABEL pair $i/$N ($TOPIC) seed=$SEED assign=$ASSIGNMENT db=$SPEECH_DB dur=$DURATION prompt=$PIDX ---"
  if env CUELOOP_SEED="$SEED" CUELOOP_ASSIGNMENT="$ASSIGNMENT" CUELOOP_SPEECH_DB="$SPEECH_DB" \
     CUELOOP_ANNOUNCE_CUE="$PA" CUELOOP_ANNOUNCE_SIL="$PB" "${GATE_ENV[@]}" \
     scripts/record_self_pair.sh "$TOPIC" "$DURATION" >/tmp/cueloop-batch-pair.log 2>&1; then
    grep -E 'predictions=|WARNING' /tmp/cueloop-batch-pair.log | head -3
    .venv/bin/python - "$TOPIC" "$FEEDBACK" "$SEED" "$ASSIGNMENT" "$SPEECH_DB" "$DURATION" "$PIDX" "${GATE_WEIGHTS:-}" "${GATE_FLOOR:-0.5}" "${FATIGUE:-2.0}" <<'PY'
import glob, json, sys
from pathlib import Path
topic, feedback = sys.argv[1], Path(sys.argv[2])
seed, assignment, db, duration, prompt, gated, floor, fatigue = sys.argv[3:11]
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
    "seed": int(seed), "assignment": assignment, "speech_db": int(db),
    "duration": int(duration), "prompt_style": int(prompt), "gated": bool(gated),
    "floor": float(floor) if gated else None, "fatigue": float(fatigue),
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
echo "=== feedback-loop summary (diversity audit) ==="
.venv/bin/python - "$FEEDBACK" <<'PY'
import json, statistics, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
clean = [r for r in rows if r.get("clean")]
print(f"total recorded={len(rows)} clean={len(clean)}")
for key in ("seed", "assignment", "speech_db", "duration", "prompt_style"):
    vals = [r.get(key) for r in rows if key in r]
    if vals:
        print(f"  {key:13s}: {len(set(vals))} distinct values -> {sorted(set(vals))[:6]}")
if clean:
    print(f"  mean delivered/pair={sum(r['cue_delivered'] for r in clean)/len(clean):.1f} "
          f"mean uptake/pair={sum(r['cue_uptake'] for r in clean)/len(clean):.2f}")
PY
echo
echo "=== accumulating clean pairs and re-evaluating H1/H2 ==="
.venv/bin/python scripts/evaluate_h1h2.py
