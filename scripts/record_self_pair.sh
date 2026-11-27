#!/bin/bash
# Record one paired CUE/SILENCE self-experiment session on this Mac.
# When the 3rd arg is given, only those operations are eligible for the
# rule decision (e.g. 'boundary,comparison,invert' to suppress speaker-natural
# op-tokens). Without it, all 8 ops are eligible (default PRIORITY).
# Example:
#   scripts/record_self_pair.sh design-choice
#  -> creates logs/self/design-choice-{cue,silence}.jsonl/.report with the
#     same pair_key so analyze_experiment.py can pair them automatically.

set -euo pipefail

warn_if_silent() {
  # a session where SFSpeechRecognizer never produced a single character of
  # text is empty from an experiment standpoint; say so loudly instead of
  # letting an empty "success" be mistaken for data. The .jsonl prediction
  # lines carry the recognized clause text in a `"clause"` field (there are
  # no separate transcript events), so we scan for non-empty clause text.
  local arm_log="$1"
  if ! grep -qE '"clause":[[:space:]]*"[^"]*[^"[:space:]][^"]*"' "$arm_log" 2>/dev/null; then
    echo ""
    echo "WARNING: this arm captured NO recognizable speech from the microphone."
    echo "         (no prediction carried any Japanese clause text.)"
    echo "         The paired data is empty; speak at the mic and rerun."
    echo ""
  fi
}

if [ $# -lt 1 ]; then
  echo "usage: $0 TOPIC_SLUG [duration_seconds] [restrict_ops ops,comma,separated]"
  echo "example: $0 design-choice 180 boundary,comparison,invert"
  echo "env:  RESTRICT_OPS=boundary,comparison,invert (default: all ops)"
  exit 1
fi
TOPIC="$1"
DURATION="${2:-180}"
RESTRICT_OPS="${3:-${RESTRICT_OPS:-}}"
# Diversity knobs (set by the batch driver) so repeated sessions are NOT
# identical samples from one fixed condition. The CUE and SILENCE arms of a
# single pair always share these values; only the labeled condition differs.
SEED="${CUELOOP_SEED:-3}"
ASSIGNMENT="${CUELOOP_ASSIGNMENT:-alternate}"
SPEECH_DB="${CUELOOP_SPEECH_DB:--50}"
# Experiment configuration (previously hard-coded). Override to run the
# richer stack: CUELOOP_CUE_STYLE=stem CUELOOP_DECISION=local CUELOOP_PREDICTOR=local.
DECISION="${CUELOOP_DECISION:-rule}"
PREDICTOR="${CUELOOP_PREDICTOR:-ngram}"
CUE_STYLE="${CUELOOP_CUE_STYLE:-bare}"
LLM_ARGS=()
if [ -n "${CUELOOP_BASE_URL:-}" ]; then LLM_ARGS+=(--base-url "$CUELOOP_BASE_URL"); fi
if [ -n "${CUELOOP_MODEL:-}" ]; then LLM_ARGS+=(--model "$CUELOOP_MODEL"); fi
# TTS command; faster rate shortens the cue's airtime.
TTS_JSON='["say","-v","Kyoko"]'
if [ -n "${CUELOOP_TTS_RATE:-}" ]; then
  TTS_JSON="[\"say\",\"-v\",\"Kyoko\",\"-r\",\"${CUELOOP_TTS_RATE}\"]"
  echo "    TTS rate: ${CUELOOP_TTS_RATE} wpm"
fi
ANNOUNCE_CUE="${CUELOOP_ANNOUNCE_CUE:-これから話してください。結論は、なぜなら、逆に、具体例は、といった言葉を使いながら、日本語で話し続けてください。}"
ANNOUNCE_SIL="${CUELOOP_ANNOUNCE_SIL:-もう一度、同じ話題で話してください。先ほどと同じように、結論は、なぜなら、逆に、といった言葉を使ってください。}"
# Optional anti-over-insertion gate (derived by scripts/tune_operation_weights.py).
GATE_ARGS=()
if [ -n "${CUELOOP_OPERATION_WEIGHTS:-}" ]; then
  WEIGHTS_JSON="$(tr -d '\n ' < "${CUELOOP_OPERATION_WEIGHTS}")"
  GATE_ARGS=(--operation-weights "$WEIGHTS_JSON" --min-operation-weight "${CUELOOP_MIN_OP_WEIGHT:-0.5}")
  echo "    operation gate: weights=$CUELOOP_OPERATION_WEIGHTS floor=${CUELOOP_MIN_OP_WEIGHT:-0.5}"
fi
if [ -n "${CUELOOP_FATIGUE:-}" ]; then
  GATE_ARGS+=(--fatigue-penalty "$CUELOOP_FATIGUE")
  echo "    fatigue penalty: $CUELOOP_FATIGUE"
fi
if [ "${CUELOOP_ATTENTION_SELECT:-0}" = "1" ]; then
  GATE_ARGS+=(--attention-select)
  echo "    attention-grounded cue selection: ON"
fi
if [ -n "${CUELOOP_INSERTION_DELAY_MS:-}" ]; then
  GATE_ARGS+=(--insertion-delay-ms "$CUELOOP_INSERTION_DELAY_MS")
  echo "    insertion delay: ${CUELOOP_INSERTION_DELAY_MS}ms"
fi
if [ -n "${CUELOOP_DIVERGENCE_TOLERANCE:-}" ]; then
  GATE_ARGS+=(--divergence-tolerance "$CUELOOP_DIVERGENCE_TOLERANCE")
  echo "    divergence tolerance: ${CUELOOP_DIVERGENCE_TOLERANCE} chars"
fi
if [ -n "${CUELOOP_COOLDOWN_MS:-}" ]; then
  GATE_ARGS+=(--cooldown-ms "$CUELOOP_COOLDOWN_MS")
  echo "    cooldown: ${CUELOOP_COOLDOWN_MS}ms"
fi
STAMP="$(date +%Y%m%d-%H%M%S)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -x ".build/macos-speech-events" ]; then
  echo "error: macOS adapter not built. run:"
  echo "  mkdir -p .build && swiftc tools/macos_speech_events.swift -o .build/macos-speech-events \\"
  echo "    -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist \\"
  echo "    -Xlinker tools/macos_speech_events.Info.plist"
  exit 1
fi

mkdir -p logs/self

STAMP2="$(date +%Y%m%d-%H%M%S)"

# raw logs get a timestamp suffix so accumulated sessions are never
# overwritten; the paired staging dir is likewise unique per invocation
CUE_STEM="logs/self/${TOPIC}-cue-${STAMP2}"
SILENCE_STEM="logs/self/${TOPIC}-silence-${STAMP2}"

announce() {
  echo "$1"
  say -v Kyoko "$1" >/dev/null 2>&1 || true
}

echo "=== recording CUE arm: $TOPIC (speaker will hear cues in Japanese)"
echo "    condition diversity: seed=$SEED assignment=$ASSIGNMENT speech_db=$SPEECH_DB duration=$DURATION"
announce "$ANNOUNCE_CUE"
# Let the announcement finish and give SFSpeechRecognizer time to warm up;
# the first recognition result can lag by ~20s after the adapter starts.
sleep 3
python3 -m pip install -q -e . 2>/dev/null || true
.venv/bin/cueloop-sync \
  --event-command-json "[\".build/macos-speech-events\",\"--locale\",\"ja-JP\",\"--duration\",\"$DURATION\",\"--speech-db\",\"$SPEECH_DB\"]" \
  --restrict-ops "${RESTRICT_OPS}" \
  --decision "$DECISION" --predictor "$PREDICTOR" ${LLM_ARGS[@]+"${LLM_ARGS[@]}"} \
  --tts-command-json "$TTS_JSON" \
  --condition-label CUE --assignment "$ASSIGNMENT" --seed "$SEED" --cue-style "$CUE_STYLE" \
  ${GATE_ARGS[@]+"${GATE_ARGS[@]}"} \
  --pair-key "self-$TOPIC-${STAMP}" \
  --session-id "self-$TOPIC-cue" \
  --log "${CUE_STEM}.jsonl" --report "${CUE_STEM}.report"
warn_if_silent "${CUE_STEM}.jsonl"

echo
echo "=== recording SILENCE arm: $TOPIC (no cues voiced)"
announce "$ANNOUNCE_SIL"
# Same warm-up as the CUE arm so the two arms stay comparable.
sleep 3
.venv/bin/cueloop-sync \
  --event-command-json "[\".build/macos-speech-events\",\"--locale\",\"ja-JP\",\"--duration\",\"$DURATION\",\"--speech-db\",\"$SPEECH_DB\"]" \
  --restrict-ops "${RESTRICT_OPS}" \
  --decision "$DECISION" --predictor "$PREDICTOR" ${LLM_ARGS[@]+"${LLM_ARGS[@]}"} \
  --condition-label SILENCE --assignment "$ASSIGNMENT" --seed "$SEED" --cue-style "$CUE_STYLE" \
  ${GATE_ARGS[@]+"${GATE_ARGS[@]}"} \
  --silent \
  --pair-key "self-$TOPIC-${STAMP}" \
  --session-id "self-$TOPIC-silence" \
  --log "${SILENCE_STEM}.jsonl" --report "${SILENCE_STEM}.report"
warn_if_silent "${SILENCE_STEM}.jsonl"

PAIR_DIR="logs/self-$(basename "$TOPIC")-paired-${STAMP2}"
mkdir -p "$PAIR_DIR"
# a totally silent arm may produce no .jsonl at all; keep the paired dir complete
touch "${CUE_STEM}.jsonl" "${SILENCE_STEM}.jsonl"
cp "${CUE_STEM}.jsonl" "$PAIR_DIR/cue.jsonl"
cp "${CUE_STEM}.report" "$PAIR_DIR/cue.json"
cp "${SILENCE_STEM}.jsonl" "$PAIR_DIR/silence.jsonl"
cp "${SILENCE_STEM}.report" "$PAIR_DIR/silence.json"

echo
echo "=== done"
echo "analyzer will pair these automatically:"
echo "  scripts/analyze_experiment.py $PAIR_DIR --markdown $PAIR_DIR/self-pair-analysis.md"
