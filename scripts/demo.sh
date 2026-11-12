#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-replay}"
duration="${2:-45}"
decision="${CUELOOP_DECISION:-rule}"
events="${CUELOOP_EVENTS:-experiments/sync-smoke.jsonl}"
stamp="$(date +%Y%m%d-%H%M%S)"
outdir="artifacts/demo"
mkdir -p "$outdir"
log="$outdir/${stamp}-${mode}.jsonl"
report="$outdir/${stamp}-${mode}-report.json"

if [[ -x "$ROOT/.venv/bin/cueloop-sync" ]]; then
  sync=("$ROOT/.venv/bin/cueloop-sync")
elif command -v cueloop-sync >/dev/null 2>&1; then
  sync=(cueloop-sync)
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  sync=("$ROOT/.venv/bin/python" -m cueloop.sync.cli)
else
  sync=(python3 -m cueloop.sync.cli)
fi

tts=()
if [[ "${CUELOOP_TTS:-1}" != "0" ]] && command -v say >/dev/null 2>&1; then
  if [[ "$(say -v '?' 2>/dev/null)" == *Kyoko* ]]; then
    tts=(--tts-command-json '["say","-v","Kyoko"]')
  else
    tts=(--tts-command-json '["say"]')
  fi
fi

# The replay demo is a deterministic pipeline check, so it pins the bare
# condition by default; pass CUELOOP_CUE_STYLE=stem to watch the stem pipeline.
# The live demo defaults to stem (the new default).
cue_style="${CUELOOP_CUE_STYLE:-}"

args=(--log "$log" --report "$report" --decision "$decision")
if [[ "$decision" == "local" ]]; then
  args+=(--decision-fallback rule)
fi
if [[ "$decision" == "linear" ]]; then
  args+=(--policy-model "${CUELOOP_POLICY_MODEL:-artifacts/sync-policy.json}")
fi
args+=("${tts[@]}")

case "$mode" in
  replay)
    style="${cue_style:-bare}"
    echo "CUELOOP replay demo: ${events}"
    echo "A cue is spoken at the first boundary. Set CUELOOP_TTS=0 to mute."
    echo "cue style: ${style} (set CUELOOP_CUE_STYLE=stem to watch the stem pipeline)"
    "${sync[@]}" --events "$events" --realtime "${args[@]}" --cue-style "$style"
    ;;
  live)
    style="${cue_style:-stem}"
    adapter=".build/macos-speech-events"
    if [[ ! -x "$adapter" ]]; then
      echo "missing $adapter" >&2
      echo "build it with the command in README.md, then retry." >&2
      exit 1
    fi
    echo "CUELOOP live demo: ${duration}s"
    echo "Speak a sentence containing 結論は / つまり / 前提は / 逆に."
    echo "The cue is spoken through macOS say. Press Ctrl-C to stop."
    echo "cue style: ${style}"
    event_cmd=$(printf '[".build/macos-speech-events","--duration","%s"]' "$duration")
    "${sync[@]}" --event-command-json "$event_cmd" --tts-async "${args[@]}" --cue-style "$style"
    ;;
  *)
    echo "usage: $0 [replay|live] [duration_seconds]" >&2
    exit 2
    ;;
esac

echo "demo artifacts under $outdir"
