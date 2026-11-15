#!/bin/bash
# Record one paired GPT Live CUE/SILENCE session on this Mac using the
# GPT-live-as-speaker configuration described in docs/gpt-live-protocol.md.
# Example:
#   scripts/record_gpt_live_pair.sh design-choice
#  -> creates logs/gpt-live/<TOPIC>-{cue,silence}-<STAMP>.jsonl/.report with
#     the same pair_key so the analyzer pairs them automatically.
set -euo pipefail

TOPIC="${1:-design-choice}"
DURATION="${2:-300}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
cd "$ROOT"

if [ ! -x ".build/macos-speech-events" ]; then
  echo "error: macOS adapter not built"
  exit 1
fi

mkdir -p logs/gpt-live
echo "STAMP: $STAMP"

announce() {
  echo "$1"
  say -v Kyoko "$1" >/dev/null 2>&1 || true
}

echo "=== CUE arm: listen for GPT Live reasoning, then intervene"
announce "これから試します、どうぞ話してください"
.venv/bin/cueloop-sync \
  --event-command-json "[\".build/macos-speech-events\",\"--locale\",\"ja-JP\",\"--duration\",\"$DURATION\",\"--speech-db\",\"-50\"]" \
  --decision rule --predictor ngram \
  --tts-command-json '["say","-v","Kyoko"]' \
  --condition-label CUE --assignment canonical --seed 3 --cue-style bare \
  --pair-key "gpt-live-$TOPIC-$STAMP" \
  --session-id "gpt-live-$TOPIC-cue" \
  --log "logs/gpt-live/${TOPIC}-cue-${STAMP}.jsonl" --report "logs/gpt-live/${TOPIC}-cue-${STAMP}.report"

echo
echo "=== SILENCE arm: same GPT Live speech, silent CUELOOP"
announce "もう一度、同じ話題で話してください"
.venv/bin/cueloop-sync \
  --event-command-json "[\".build/macos-speech-events\",\"--locale\",\"ja-JP\",\"--duration\",\"$DURATION\",\"--speech-db\",\"-50\"]" \
  --decision rule --predictor ngram \
  --condition-label SILENCE --assignment canonical --seed 3 --cue-style bare \
  --silent \
  --pair-key "gpt-live-$TOPIC-$STAMP" \
  --session-id "gpt-live-$TOPIC-silence" \
  --log "logs/gpt-live/${TOPIC}-silence-${STAMP}.jsonl" --report "logs/gpt-live/${TOPIC}-silence-${STAMP}.report"

PAIR_DIR="logs/gpt-live-${TOPIC}-paired-${STAMP}"
mkdir -p "$PAIR_DIR"
touch "logs/gpt-live/${TOPIC}-cue-${STAMP}.jsonl" "logs/gpt-live/${TOPIC}-silence-${STAMP}.jsonl"
cp "logs/gpt-live/${TOPIC}-cue-${STAMP}.jsonl" "$PAIR_DIR/cue.jsonl"
cp "logs/gpt-live/${TOPIC}-cue-${STAMP}.report" "$PAIR_DIR/cue.json"
cp "logs/gpt-live/${TOPIC}-silence-${STAMP}.jsonl" "$PAIR_DIR/silence.jsonl"
cp "logs/gpt-live/${TOPIC}-silence-${STAMP}.report" "$PAIR_DIR/silence.json"

echo
echo "=== done"
echo "analyzer will pair these automatically:"
echo "  scripts/analyze_experiment.py $PAIR_DIR --markdown $PAIR_DIR/self-pair-analysis.md"
