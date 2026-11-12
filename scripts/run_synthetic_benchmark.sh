#!/usr/bin/env bash
# Regenerate the synthetic calibration benchmark end to end.
# Everything here is synthetic (programmed effects). It calibrates the analysis
# and training pipeline; it is NOT evidence about human speech.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

OUT="${1:-artifacts/benchmark}"
mkdir -p "$OUT"

echo "== analyzer grid: effects {0, 0.10, 0.20, 0.30} x seeds {0, 1, 2} =="
for seed in 0 1 2; do
  for effect in 0 0.10 0.20 0.30; do
    dir="$OUT/effect-${effect}-s${seed}"
    if [ ! -d "$dir" ]; then
      "$PYTHON" scripts/synthesize_experiment.py \
        --output-dir "$dir" --sessions 24 --clauses 8 --seed "$seed" \
        --uptake-effect "$effect" --carryover-effect "$effect" >/dev/null
    fi
    "$PYTHON" scripts/analyze_experiment.py "$dir" \
      --json "$OUT/effect-${effect}-s${seed}.json" \
      --markdown "$OUT/effect-${effect}-s${seed}.md" >/dev/null
  done
done

echo "== policy corpus + training at scale =="
corpus="$OUT/corpus-train"
if [ ! -d "$corpus" ]; then
  "$PYTHON" scripts/synthesize_logs.py \
    --output-dir "$corpus" --sessions 600 --clauses 8 --seed 0 \
    --assignments canonical >/dev/null
fi
"$PYTHON" scripts/train_policy.py "$corpus"/*.jsonl \
  --label carryover \
  --output "$OUT/sync-policy.json" \
  --report "$OUT/sync-policy-eval.json" >/dev/null
holdout_corpus="$OUT/corpus-holdout"
if [ ! -d "$holdout_corpus" ]; then
  "$PYTHON" scripts/synthesize_logs.py \
    --output-dir "$holdout_corpus" --sessions 150 --clauses 8 --seed 999 \
    --assignments canonical,random-operation >/dev/null
fi
"$PYTHON" scripts/evaluate_policy.py "$holdout_corpus"/*.jsonl \
  --model "$OUT/sync-policy.json" \
  --report "$OUT/sync-policy-holdout.json" >/dev/null

echo "== summary =="
"$PYTHON" - "$OUT" <<'PY'
import glob
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
excluded = total = 0
null_bad = 0
detected = 0
for path in sorted(glob.glob(str(out / "effect-*-s*.json"))):
    effect = float(path.split("effect-")[1].split("-s")[0])
    data = json.loads(Path(path).read_text())
    for backend, comparison in data["comparisons"].items():
        metric = comparison["metrics"]["uptake_rate"]
        total += 1
        if effect == 0.0:
            if abs(metric["point"]) > 1e-9:
                null_bad += 1
        elif metric["ci_low"] != "insufficient" and metric["ci_low"] > 0:
            detected += 1
excluded = detected
print(f"analyzer grid: {total} cells")
print(f"  null cells with delta != 0: {null_bad} (expect 0)")
print(f"  effect>=0.10 cells with CI excluding 0: {excluded}/27 (expect 27 at n_pairs=24)")
eval_path = out / "sync-policy-eval.json"
holdout_path = out / "sync-policy-holdout.json"
if eval_path.exists():
    train = json.loads(eval_path.read_text())
    print(f"policy: {train['samples']} samples "
          f"(cv holdout auc={train['eval_metrics']['roc_auc']:.3f}, "
          f"train auc={train['train_metrics']['roc_auc']:.3f})")
if holdout_path.exists():
    holdout = json.loads(holdout_path.read_text())
    print(f"policy on independent holdout: "
          f"auc={holdout['metrics']['roc_auc']:.3f} "
          f"samples={holdout['metrics']['samples']}")
PY

echo "benchmark written under $OUT (synthetic calibration, not human evidence)"
