# Voice MVP

The voice core is deliberately independent of a speech-recognition vendor. A VAD/ASR adapter emits ordered JSONL events, CUELOOP detects a stable turn boundary, and the deterministic controller decides whether anything may be spoken.

## Event contract

Each line is one JSON object. `at_ms` is monotonic milliseconds from the beginning of the session.

```json
{"type":"speech_started","at_ms":0}
{"type":"transcript_partial","at_ms":300,"text":"この設計は"}
{"type":"transcript_final","at_ms":900,"text":"この設計は前提を分けて考える。"}
{"type":"speech_ended","at_ms":1000}
{"type":"tick","at_ms":1700}
{"type":"session_ended","at_ms":2000}
```

Allowed event types are `speech_started`, `transcript_partial`, `transcript_final`, `speech_ended`, `tick`, `interventions_paused`, `interventions_resumed`, `participant_stop`, `adapter_warning`, and `session_ended`. `adapter_warning` requires a text description; other control events do not. A turn is not analyzed until the user is no longer speaking and the configured endpoint silence has elapsed. If speech resumes first, the pending boundary is cancelled and the transcript continues.

## Commands

Deterministic replay:

```bash
cueloop-voice --events experiments/voice-smoke.jsonl --log artifacts/voice-smoke.jsonl
```

Streaming adapter with audible delivery:

```bash
cueloop-voice --event-command path/to/asr-adapter --tts-command say
```

When a live `--event-command` is used without TTS, `cueloop-voice` runs in observation-only mode: it records state hypotheses but suppresses every intervention. Replays may omit TTS because they evaluate decisions rather than claim human delivery. `cueloop-session run --condition CUELOOP` requires a speaker for live input; the `CONTROL` condition remains silent by design. All integrated live sessions additionally require `--consent-acknowledged` after the participant has been informed about recording, transcription, storage, and deletion.

On macOS, build the included local-first adapter and run a 10-minute session:

```bash
mkdir -p .build
swiftc tools/macos_speech_events.swift -o .build/macos-speech-events \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist \
  -Xlinker tools/macos_speech_events.Info.plist
cueloop-voice \
  --event-command-json '[".build/macos-speech-events","--duration","600"]' \
  --tts-command say
```

The adapter defaults to on-device recognition. If the installed locale does not support it, it exits instead of silently sending speech away from the device. `--allow-network-recognition` is an explicit opt-in and should be used only after confirming consent and data handling.

The adapter prints its process ID. Send `SIGUSR1` to toggle intervention delivery without stopping transcription; the pause/resume event is preserved in the session manifest. Pressing Ctrl-C emits `participant_stop` before shutdown, allowing the experiment record to derive the stop from an integrity-checked artifact.

For continuous sessions, the adapter rolls the Apple recognition task over every 45 seconds at the next silent boundary. The audio engine remains active and raw audio is not written to disk.

The adapter must write only event JSONL to standard output. Diagnostics belong on standard error. CUELOOP never invokes a shell; command arguments are passed directly.

An external semantic analyzer can be added with `--analyzer-command`. To enable real-time `RETRIEVE`, also pass trusted local files through `--sources`. CUELOOP discards analyzer-provided retrieval text and references, searches the local index again, and only delivers a short passage whose chunk ID exists in that index. Without a matching trusted chunk, the controller chooses `SILENCE`.

## Timing and failure behavior

- Default endpoint silence: 700 ms.
- Default candidate delivery window: 1,500 ms after speech end; late analysis expires silently.
- Default voice cooldown: 5,000 ms between delivered interventions.
- Default streaming-adapter wall-time limit: 1,900 seconds, covering a 30-minute session plus shutdown margin.
- Linguistic output is never delivered while `speaking` is true.
- Resumed speech cancels a pending candidate and terminates command-based TTS already playing.
- Late final ASR hypotheses are accepted during a bounded 3-second grace period instead of dropping the turn at the first endpoint tick.
- Out-of-order or malformed events terminate the session with a non-zero status.
- Analyzer failure and invalid analyzer output fail closed to `SILENCE`.
- TTS is optional for replay and monitoring; live CUELOOP delivery requires it.
- Candidate logs include the formal candidate, typed evidence, delivery constraints, and `DELIVERED`, `CANCELLED`, or `EXPIRED` lifecycle status.
- Raw audio is neither required nor retained by the core.

## Privacy checklist before a live session

1. Confirm which ASR adapter receives audio and whether it sends data off-device.
2. State clearly that recording/transcription is active.
3. Use a private local directory for logs.
4. Do not retain raw audio unless the experiment requires it and the participant consents.
5. Remove `data/`, `logs/`, and `artifacts/` when the session data is no longer needed.
