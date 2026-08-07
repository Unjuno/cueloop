# M2 Audio Event Boundary

M2 introduces audio without binding CUELOOP to a specific ASR, VAD, microphone, or TTS provider.

## Objective

Convert provider-specific streaming audio output into a small, deterministic event vocabulary that the reasoning controller can consume.

The audio layer must report observable interaction events. It must not infer hidden cognitive states.

## Event model

### Transcript events

`TranscriptEvent` carries:

- `kind`: `PARTIAL` or `FINAL`
- `text`
- `start_ms`
- `end_ms`
- optional provider confidence

Partial text is unstable and may be revised. Only final text is committed to the stable transcript.

### Boundary events

`BoundaryEvent` carries:

- `SPEECH_STARTED`
- `SPEECH_ENDED`
- `TURN_CANDIDATE`

`TURN_CANDIDATE` may include observed silence duration. It means only that the audio layer considers a turn boundary plausible. It does not mean the user is stuck or finished thinking.

## Invariants

1. Time is monotonic milliseconds relative to the current audio session.
2. Partial transcripts never become stable context until finalized.
3. A VAD pause is not a cognitive-state classification.
4. Turn candidates are proposals; higher-level control may ignore them.
5. Provider confidence is optional and must not be treated as calibrated unless the provider documents it as such.
6. Provider-specific objects must not cross this boundary.

## Transcript buffer

`TranscriptBuffer` stores:

- committed final segments
- one replaceable partial segment

This prevents repeated partial ASR hypotheses from being appended as if they were independent user statements.

## Next step

Build a replay source that reads recorded event sequences from JSONL and drives the existing analyzer/controller loop. This allows the turn-taking logic to be debugged deterministically before live microphone integration.

Only after replay behavior is stable should CUELOOP choose an initial streaming ASR/VAD provider.
