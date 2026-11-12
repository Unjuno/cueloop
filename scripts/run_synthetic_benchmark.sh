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

echo "== real-speech replay: Lux corpus =="
lux_out="$OUT/simulated-lux-s0"
if [ ! -d "$lux_out" ]; then
  if [ -f data/real-speech/lux/lux-whisper-results.json ] && command -v /opt/homebrew/bin/python3 >/dev/null; then
    PYTHONPATH="$PWD/.venv/lib/python$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages" \
      /opt/homebrew/bin/python3 scripts/simulate_lux_replay.py \
        --output-dir "$lux_out" --whisper-model /tmp/whisper-base >/dev/null || \
      echo "(lux replay skipped: mlx_whisper unavailable)"
  else
    echo "(lux replay skipped: corpus/cache missing)"
  fi
fi
if [ -d "$lux_out/cue" ] && [ -d "$lux_out/silence" ]; then
  "$PYTHON" scripts/analyze_experiment.py "$lux_out/cue" "$lux_out/silence" \
    --json "$OUT/simulated-lux-s0.json" \
    --markdown "$OUT/simulated-lux-s0.md" >/dev/null
fi

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

echo "== end-to-end runtime simulation: effects {0, 0.2, 0.4} x seeds {0, 1} =="
for seed in 0 1 2 3; do
  for effect in 0 0.2 0.4; do
    dir="$OUT/simulated-${effect}-s${seed}"
    if [ ! -d "$dir" ]; then
      "$PYTHON" scripts/simulate_sessions.py \
        --output-dir "$dir" --sessions 24 --clauses 8 --seed "$seed" \
        --uptake-effect "$effect" --carryover-effect "$effect" >/dev/null
    fi
    "$PYTHON" scripts/analyze_experiment.py "$dir" \
      --json "$OUT/simulated-${effect}-s${seed}.json" \
      --markdown "$OUT/simulated-${effect}-s${seed}.md" >/dev/null
  done
done

echo "== stem-cue arm: effects {0, 0.2, 0.4} x seeds {0, 1} =="
for seed in 0 1; do
  for effect in 0 0.2 0.4; do
    dir="$OUT/simulated-stem-${effect}-s${seed}"
    if [ ! -d "$dir" ]; then
      "$PYTHON" scripts/simulate_sessions.py \
        --output-dir "$dir" --sessions 24 --clauses 8 --seed "$seed" \
        --uptake-effect "$effect" --carryover-effect "$effect" \
        --cue-style stem >/dev/null
    fi
    "$PYTHON" scripts/analyze_experiment.py "$dir" \
      --json "$OUT/simulated-stem-${effect}-s${seed}.json" \
      --markdown "$OUT/simulated-stem-${effect}-s${seed}.md" >/dev/null
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

for path in sorted(glob.glob(str(out / "simulated-[0-9]*-s*.json"))):
    data = json.loads(Path(path).read_text())
    effect = float(path.split("simulated-")[1].split("-s")[0])
    seed = path.split("-s")[-1].split(".")[0]
    comp = data.get("comparisons", {}).get("rule", {})
    if comp:
        metric = comp["metrics"]["uptake_rate"]
        carry = comp["metrics"]["carryover_rate"]
        print(f"simulated effect={effect:.1f} seed={seed}: "
              f"uptake_delta={metric['point']:+.3f} "
              f"CI=[{metric['ci_low']:+.3f},{metric['ci_high']:+.3f}] "
              f"carryover_delta={carry['point']:+.3f}")

for path in sorted(glob.glob(str(out / "simulated-stem-*-s*.json"))):
    data = json.loads(Path(path).read_text())
    effect = float(path.split("simulated-stem-")[1].split("-s")[0])
    seed = path.split("-s")[-1].split(".")[0]
    comp = data.get("comparisons", {}).get("rule", {})
    if comp:
        metric = comp["metrics"]["uptake_rate"]
        print(f"simulated stem effect={effect:.1f} seed={seed}: "
              f"uptake_delta={metric['point']:+.3f} "
              f"CI=[{metric['ci_low']:+.3f},{metric['ci_high']:+.3f}]")
PY

echo "benchmark written under $OUT (synthetic calibration, not human evidence)"
