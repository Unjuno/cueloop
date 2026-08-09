# Offline validation

CUELOOP does not require live self-use or longitudinal cognitive experiments to validate the software path.

The engineering validation target is:

```text
audio file
  -> ASR/VAD event stream
  -> stable reasoning turns
  -> Analyzer
  -> InterventionController
  -> timestamped SILENCE / CONTINUE / CUE timeline
```

This validates whether the system runs end to end and whether intervention timing is inspectable on ordinary recorded speech.

It does **not** establish durable cognitive enhancement, transfer, learning, or long-term behavior change. Those remain research questions outside the initial implementation scope.

## Replay mode

Use a deterministic event trace:

```bash
cueloop-offline --events examples/replay/thinking.jsonl
```

The output defaults to:

```text
logs/offline-timeline.jsonl
```

## Audio-file mode

CUELOOP keeps ASR outside the core package. An ASR command receives the audio file path as its final argument and must emit replay JSONL to stdout.

Example using the included faster-whisper adapter:

```bash
python -m pip install faster-whisper
cueloop-offline \
  --audio path/to/speech.wav \
  --asr-command python scripts/asr_faster_whisper.py --model small --language ja
```

The adapter uses faster-whisper segmentation plus VAD and creates `TURN_CANDIDATE` events when the gap between adjacent ASR segments exceeds a configurable threshold.

Important: a `TURN_CANDIDATE` is an observable interaction boundary only. It is not evidence that the speaker is cognitively stuck, overloaded, or finished thinking.

## Event JSONL contract

Transcript event:

```json
{"type":"transcript","kind":"FINAL","text":"...","start_ms":0,"end_ms":900}
```

Turn candidate:

```json
{"type":"boundary","kind":"TURN_CANDIDATE","at_ms":1400,"silence_ms":500}
```

Times must be monotonically non-decreasing within a trace.

## What to inspect

For recorded speech, inspect:

- whether productive reasoning produces `SILENCE`
- whether short unfinished reasoning produces `CONTINUE` rather than a full answer
- whether repeated/filler-heavy local stalls produce at most one short `CUE`
- whether cooldown suppresses bursts of interventions
- whether partial ASR hypotheses never directly trigger analysis
- whether final transcript boundaries align sufficiently with natural pauses

These are software and interaction checks, not claims of cognitive effectiveness.
