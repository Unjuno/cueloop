#!/bin/bash
# Autonomous CUE/SILENCE A/B WITHOUT microphone: local LLM writes a Japanese
# reasoning monologue, agent_text_events.py feeds it as transcript events to
# CUELOOP through stdin. Same text + seed in both arms; only CUE/SILENCE differs.
set -euo pipefail

TOPIC="${1:-agenttune}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
MODEL="${CUELOOP_AGENT_MODEL:-qwen3:8b}"
CUE_STYLE="${CUELOOP_CUE_STYLE:-bare}"
ASSIGNMENT="${CUELOOP_ASSIGNMENT:-alternate}"
cd "$ROOT"

mkdir -p logs/agent-text
TEXTFILE="logs/agent-text/${TOPIC}-${STAMP}.txt"

echo "=== LLM ($MODEL) writes monologue on: $TOPIC"
curl -s --max-time 180 http://localhost:11434/api/generate \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"次のトピックについて、日本語の推論モノローグを約900文字で書いてください: 『${TOPIC}』。まず結論を言い、第一に〜、第二に〜、つまり〜、逆に〜、依存関係、根拠 の順で展開し、話し言葉に近い自然な文章にしてください。見出しや箇条書きは付けず、連続した口語だけを書いてください。\",\"stream\":false}" \
  | /opt/homebrew/bin/python3 -c 'import sys,json; print(json.load(sys.stdin)["response"])' > "$TEXTFILE"
echo "generated $(wc -c < "$TEXTFILE") chars"

EXTRA=()
if [ -n "${CUELOOP_INSERTION_DELAY_MS:-}" ]; then EXTRA+=(--insertion-delay-ms "$CUELOOP_INSERTION_DELAY_MS"); fi
if [ -n "${CUELOOP_DIVERGENCE_TOLERANCE:-}" ]; then EXTRA+=(--divergence-tolerance "$CUELOOP_DIVERGENCE_TOLERANCE"); fi
if [ -n "${CUELOOP_COOLDOWN_MS:-}" ]; then EXTRA+=(--cooldown-ms "$CUELOOP_COOLDOWN_MS"); fi
if [ -n "${CUELOOP_MIN_CONFIDENCE:-}" ]; then EXTRA+=(--min-confidence "$CUELOOP_MIN_CONFIDENCE"); fi
if [ -n "${CUELOOP_CUE:-}" ]; then EXTRA+=(--cue "$CUELOOP_CUE"); fi
if [ "${CUELOOP_ATTENTION_SELECT:-0}" = "1" ]; then EXTRA+=(--attention-select); fi

run_arm() {
  local condition="$1" silent_flag="$2" stem="$3"
  .venv/bin/python scripts/agent_text_events.py "$TEXTFILE" | \
  CUELOOP_TTS=0 .venv/bin/cueloop-sync \
    --events - --decision rule --predictor ngram \
    --condition-label "$condition" --assignment "$ASSIGNMENT" --seed 3 \
    --cue-style "$CUE_STYLE" $silent_flag ${EXTRA[@]+"${EXTRA[@]}"} \
    --pair-key "agent-text-$TOPIC-$STAMP" \
    --session-id "agent-text-$TOPIC-$(echo "$condition" | tr A-Z a-z)" \
    --log "logs/agent-text/${TOPIC}-${condition}-${STAMP}.jsonl" \
    --report "logs/agent-text/${TOPIC}-${condition}-${STAMP}.report"
}

echo "=== CUE arm"
run_arm CUE "" ""
echo "=== SILENCE arm"
run_arm SILENCE "--silent" ""

D="logs/agent-text-${TOPIC}-paired-${STAMP}"
mkdir -p "$D"
cp "logs/agent-text/${TOPIC}-CUE-${STAMP}.jsonl" "$D/cue.jsonl"
cp "logs/agent-text/${TOPIC}-CUE-${STAMP}.report" "$D/cue.json"
cp "logs/agent-text/${TOPIC}-SILENCE-${STAMP}.jsonl" "$D/silence.jsonl"
cp "logs/agent-text/${TOPIC}-SILENCE-${STAMP}.report" "$D/silence.json"
echo "=== done: $D"
.venv/bin/python scripts/analyze_experiment.py "$D" --markdown "$D/analysis.md" | tail -12
