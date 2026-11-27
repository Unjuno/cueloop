#!/bin/bash
# Run one paired GPT-Live-style agent-live CUE/SILENCE session on this Mac.
# The "speaker" is a local LLM (qwen2.5:7b-instruct via Ollama) generating a
# Japanese reasoning monologue on demand; Kyoko reads it aloud; SFSpeechRecognizer
# captures the audio through speakers->mic (verified 2026-11-16). Not human data.
# Agent-side uptake only. NOT human H1/H2 evidence.
#
# Example:
#   scripts/run_agent_live_pair.sh design-choice
set -euo pipefail

TOPIC="${1:-design-choice}"
DURATION="${2:-300}"
AGENT_MODEL="${CUELOOP_AGENT_MODEL:-qwen3:8b}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Tuning knobs (same as record_self_pair.sh) so the autonomous voice-agent
# path can be A/B'd under the tuned latency settings without human speech.
CUE_STYLE="${CUELOOP_CUE_STYLE:-bare}"
EXTRA_ARGS=()
if [ -n "${CUELOOP_INSERTION_DELAY_MS:-}" ]; then EXTRA_ARGS+=(--insertion-delay-ms "$CUELOOP_INSERTION_DELAY_MS"); fi
if [ -n "${CUELOOP_DIVERGENCE_TOLERANCE:-}" ]; then EXTRA_ARGS+=(--divergence-tolerance "$CUELOOP_DIVERGENCE_TOLERANCE"); fi
if [ -n "${CUELOOP_COOLDOWN_MS:-}" ]; then EXTRA_ARGS+=(--cooldown-ms "$CUELOOP_COOLDOWN_MS"); fi
if [ "${CUELOOP_ATTENTION_SELECT:-0}" = "1" ]; then EXTRA_ARGS+=(--attention-select); fi
STAMP="$(date +%Y%m%d-%H%M%S)"
cd "$ROOT"

if [ ! -x ".build/macos-speech-events" ]; then
  echo "error: macOS adapter not built"
  exit 1
fi

mkdir -p logs/agent-live
TEXTFILE=logs/agent-live/$(echo "$TOPIC" | tr -cd 'a-zA-Z0-9-')-prompt-${STAMP}.txt

# 1) Ask the local LLM for a Japanese reasoning monologue (~1100 chars for ~180 s)
echo "=== step 1: ask local LLM (qwen2.5:7b) for a monologue on $TOPIC"
curl -s --max-time 180 http://localhost:11434/api/generate -d "{\"model\":\"$AGENT_MODEL\",\"prompt\":\"次のトピックについて、日本語の推論モノローグを約1000文字で書いてください: 『なぜ${TOPIC}を選ぶのか?』。まず結論を言う、そのあと第一に〜、第二に〜、つまり〜、逆に〜、依存関係、根拠 の展開で、話し言葉に近い自然な表現を使ってください。見出しやマークアップは付けず、ただ連続した口語で書いてください。\",\"stream\":false}" \
  | /opt/homebrew/bin/python3 -c 'import sys, json; print(json.load(sys.stdin)["response"])' \
  > "$TEXTFILE"
echo "generated text:"; head -c 200 "$TEXTFILE"; echo; echo "... ($(wc -c < "$TEXTFILE") chars)"

# 2) CUE arm: CUELOOP listens + speaks Kyoko cues when boundaries/triggers fire
echo
echo "=== step 2: CUE arm — Kyoko speaks the monologue; CUELOOP interjects"
announce_done=0
CUELOOP_TTS=1 .venv/bin/cueloop-sync \
  --event-command-json "[\".build/macos-speech-events\",\"--locale\",\"ja-JP\",\"--duration\",\"$DURATION\",\"--speech-db\",\"-50\"]" \
  --decision rule --predictor ngram \
  --tts-command-json '["say","-v","Kyoko"]' \
  --condition-label CUE --assignment canonical --seed 3 --cue-style "$CUE_STYLE" \
  "${EXTRA_ARGS[@]}" \
  --pair-key "agent-$TOPIC-$STAMP" \
  --session-id "agent-$TOPIC-cue" \
  --log "logs/agent-live/${TOPIC}-cue-${STAMP}.jsonl" --report "logs/agent-live/${TOPIC}-cue-${STAMP}.report" &
CPID=$!
sleep 2
say -v Kyoko -f "$TEXTFILE"
wait "$CPID"

# 3) SILENCE arm: same text, same seed, no Kyoko cues
echo
echo "=== step 3: SILENCE arm — Kyoko reads the same text, CUELOOP silent"
.venv/bin/cueloop-sync \
  --event-command-json "[\".build/macos-speech-events\",\"--locale\",\"ja-JP\",\"--duration\",\"$DURATION\",\"--speech-db\",\"-50\"]" \
  --decision rule --predictor ngram \
  --condition-label SILENCE --assignment canonical --seed 3 --cue-style "$CUE_STYLE" \
  "${EXTRA_ARGS[@]}" \
  --silent \
  --pair-key "agent-$TOPIC-$STAMP" \
  --session-id "agent-$TOPIC-silence" \
  --log "logs/agent-live/${TOPIC}-silence-${STAMP}.jsonl" --report "logs/agent-live/${TOPIC}-silence-${STAMP}.report" &
SPPID=$!
sleep 2
say -v Kyoko -f "$TEXTFILE"
wait "$SPPID"

# 4) stage a paired directory
PAIR_DIR="logs/agent-live-${TOPIC}-paired-${STAMP}"
mkdir -p "$PAIR_DIR"
cp "logs/agent-live/${TOPIC}-cue-${STAMP}.jsonl" "$PAIR_DIR/cue.jsonl"
cp "logs/agent-live/${TOPIC}-cue-${STAMP}.report" "$PAIR_DIR/cue.json"
cp "logs/agent-live/${TOPIC}-silence-${STAMP}.jsonl" "$PAIR_DIR/silence.jsonl"
cp "logs/agent-live/${TOPIC}-silence-${STAMP}.report" "$PAIR_DIR/silence.json"

echo
echo "=== done"
echo "analyzer command: scripts/analyze_experiment.py $PAIR_DIR --markdown $PAIR_DIR/self-pair-analysis.md"
