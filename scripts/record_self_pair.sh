#!/bin/bash
# Record one paired CUE/SILENCE self-experiment session on this Mac.
# Example:
#   scripts/record_self_pair.sh design-choice
#  -> creates logs/self/design-choice-{cue,silence}.jsonl/.report with the
#     same pair_key so analyze_experiment.py can pair them automatically.

set -euo pipefail

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

PAIR_DIR="logs/self-$(basename "$TOPIC")-paired"
mkdir -p "$PAIR_DIR"
cp "logs/self/${TOPIC}-cue.jsonl" "logs/self/${TOPIC}-cue.report" "$PAIR_DIR/a.jsonl"
cp "logs/self/${TOPIC}-cue.report" "$PAIR_DIR/a.json"
cp "logs/self/${TOPIC}-silence.jsonl" "logs/self/${TOPIC}-silence.report" "$PAIR_DIR/s.jsonl"
cp "logs/self/${TOPIC}-silence.report" "$PAIR_DIR/s.json"

echo
echo "=== done"
echo "analyzer will pair these automatically:"
echo "  scripts/analyze_experiment.py $PAIR_DIR --markdown $PAIR_DIR/self-pair-analysis.md"
