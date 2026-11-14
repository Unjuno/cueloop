#!/bin/bash
# Record one paired CUE/SILENCE self-experiment session on this Mac.
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
  echo "usage: $0 TOPIC_SLUG [duration_seconds]"
  echo "example: $0 design-choice 60"
  exit 1
fi
TOPIC="$1"
DURATION="${2:-60}"
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

echo "=== recording CUE arm: $TOPIC (speaker will hear cues in Japanese)"
python3 -m pip install -q -e . 2>/dev/null || true
.venv/bin/cueloop-sync \
  --event-command-json "[\".build/macos-speech-events\",\"--locale\",\"ja-JP\",\"--duration\",\"$DURATION\",\"--speech-db\",\"-50\"]" \
  --decision rule --predictor ngram \
  --tts-command-json '["say","-v","Kyoko"]' \
  --condition-label CUE --assignment canonical --seed 3 --cue-style bare \
  --pair-key "self-$TOPIC-${STAMP}" \
  --session-id "self-$TOPIC-cue" \
  --log "logs/self/${TOPIC}-cue.jsonl" --report "logs/self/${TOPIC}-cue.report"
warn_if_silent "logs/self/${TOPIC}-cue.jsonl"

echo
echo "=== recording SILENCE arm: $TOPIC (no cues voiced)"
.venv/bin/cueloop-sync \
  --event-command-json "[\".build/macos-speech-events\",\"--locale\",\"ja-JP\",\"--duration\",\"$DURATION\",\"--speech-db\",\"-50\"]" \
  --decision rule --predictor ngram \
  --condition-label SILENCE --assignment canonical --seed 3 --cue-style bare \
  --silent \
  --pair-key "self-$TOPIC-${STAMP}" \
  --session-id "self-$TOPIC-silence" \
  --log "logs/self/${TOPIC}-silence.jsonl" --report "logs/self/${TOPIC}-silence.report"
warn_if_silent "logs/self/${TOPIC}-silence.jsonl"

PAIR_DIR="logs/self-$(basename "$TOPIC")-paired"
mkdir -p "$PAIR_DIR"
# a totally silent arm may produce no .jsonl at all; keep the paired dir complete
touch "logs/self/${TOPIC}-cue.jsonl" "logs/self/${TOPIC}-silence.jsonl"
cp "logs/self/${TOPIC}-cue.jsonl" "$PAIR_DIR/cue.jsonl"
cp "logs/self/${TOPIC}-cue.report" "$PAIR_DIR/cue.json"
cp "logs/self/${TOPIC}-silence.jsonl" "$PAIR_DIR/silence.jsonl"
cp "logs/self/${TOPIC}-silence.report" "$PAIR_DIR/silence.json"

echo
echo "=== done"
echo "analyzer will pair these automatically:"
echo "  scripts/analyze_experiment.py $PAIR_DIR --markdown $PAIR_DIR/self-pair-analysis.md"
