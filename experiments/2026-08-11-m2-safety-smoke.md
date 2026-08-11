# M2 Safety Smoke Experiment — 2026-08-11

## Scope
System-behavior experiment only. This does **not** test long-term human cognitive improvement.

Runtime: containerized Python execution against the current `feat/m2-audio-events` logic reconstructed from repository source.

## Hypotheses and pass criteria

- H1: FLOWING turns must not trigger an intervention. PASS if false interventions = 0.
- H2: Low-confidence ASR must not trigger an intervention. PASS if interventions = 0 at ASR confidence 0.20 with threshold 0.45.
- H3: A short ordinary pause must not consume the transcript or trigger a decision. PASS if 200 ms pause produces no decision and a later 500 ms boundary still evaluates the buffered turn.
- H4: Repeated STUCK-like turns must respect turn/time cooldown. PASS if repeated 1 s-spaced turns alternate intervention/silence rather than delivering every turn.
- H5: Barge-in cancellation must terminate an active output process promptly. PASS if process termination completes under 100 ms in the container surrogate.

## Results

| Test | Result | Status |
|---|---:|---|
| FLOWING false interventions | 0 | PASS |
| Low-ASR interventions | 0 | PASS |
| 200 ms short-pause decision | none | PASS |
| Later 500 ms boundary | CUE | PASS |
| Repeated STUCK actions | CUE, SILENCE, CUE, SILENCE | PASS |
| Barge-in stop latency | 1.179 ms | PASS |

## Interpretation

The current gating stack behaves conservatively under this deterministic smoke test: productive flow is preserved, ASR uncertainty cannot create an intervention, ordinary short pauses do not create false turn boundaries, intervention cooldown suppresses repeated cues, and output cancellation is fast in the process-level surrogate.

## Limits

- Synthetic/deterministic transcript events, not natural speech.
- Barge-in measurement is process termination latency, not acoustic playback cessation latency on real hardware.
- No production Japanese streaming ASR or production TTS in this run.
- No claim about human learning, reasoning improvement, or long-term cognitive effects.

## Next experiment

Run recorded natural Japanese speech through the actual ASR adapter and measure false intervention rate, ASR suppression rate, pause-gate behavior, cue density, and end-to-end boundary-to-output latency.
