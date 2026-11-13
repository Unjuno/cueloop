# Self-Experiment: use your own voice as the Japanese-reasoning speaker

This is a pilot recipe for running the `experiment-protocol.md` design with
your own voice and this Mac — no second participant, no hardware purchase.
It is not a substitute for the protocol's 24–32 pairs/cell power analysis;
it gives a real, transcribed, paired CUE/SILENCE signal on human reasoning
speech instead of a simulation.

## Why this exists

All public Japanese corpora with per-word timestamps are read/recited
(Lux, CC-BY-4.0) or broadcast (ReazonSpeech, CDLA-Sharing-1.0). Neither
contains spontaneous reasoning with operation announcements. The one voice
that can produce it is yours.

## What you need

- macOS with the built-in microphone (`system_profiler SPAudioDataType` —
  currently: MacBook Pro Microphone + MacBook Pro Speakers, no headphones)
- the repository checked out and `.venv` created (see README Quickstart)
- ~15 minutes

## Tasks (3 topics, 2 passes each)

Pick three topics in Japanese that naturally produce reasoning operations.
For example:

1. "How do I decide whether to roll this service back?" — operational triage
2. "Explain why caching is or is not appropriate for this endpoint." — design trade-off
3. "What can you infer from this stack trace?" — debugging reasoning

For each topic, record **two passes**:

- **Pass A**: reason aloud. Keep going until you would naturally stop.
- **Pass B**: reason aloud the same topic again. Different content, same task.

Then alternate conditions at the clause level when you record with the tool —
the analyzer handles pairing by `pair_key`.

## Recording

One-command paired recording (sets the same `pair_key` for both arms):

```bash
scripts/record_self_pair.sh design-choice 60
```

That opens the mic for 60 s of CUE (cues spoken in Japanese via Kyoko) then
60 s of SILENCE (no voice), using the same `pair_key`, and writes
`logs/self/design-choice-{cue,silence}.jsonl` plus matching reports under
`logs/self-design-choice-paired/`. It fails loud if the adapter is missing.

To pair them manually:

```bash
# 1) record a CUE session (cues spoken, recorded, suppressed in transcript)
cueloop-sync \
  --event-command-json '[".build/macos-speech-events","--locale","ja-JP","--duration","300","--speech-db","-38"]' \
  --decision rule --condition-label CUE --assignment random --seed 3 \
  --pair-key self-voice-a --session-id self-voice-a --cue-style bare \
  --log logs/self-voice-a.jsonl --report logs/self-voice-a.report.json

# 2) record the SILENCE twin on the SAME topic and SAME seed
cueloop-sync \
  --event-command-json '[".build/macos-speech-events","--locale","ja-JP","--duration","300","--speech-db","-38"]' \
  --decision rule --condition-label SILENCE --assignment random --seed 3 \
  --pair-key self-voice-a --session-id self-voice-a --cue-style bare \
  --log logs/self-voice-a-silence.jsonl --report logs/self-voice-a-silence.report.json
```

`pair-key` must be identical between the two arms so the analyzer pairs them.
`--silent` is NOT wanted here: you want the cue audible.

## Analysis

```bash
scripts/analyze_experiment.py logs/self-voice-a.report.json logs/self-voice-a-silence.report.json \
  --markdown logs/self-voice-a-analysis.md
```

Report the uptake/carryover deltas from `analyze_experiment.py`. If a pass
contains no boundaries (single-τmonologue with no operation announcements),
it contributes zero deliveries and the pair is excluded.

## Honest limits

- One speaker, one task ×3, one condition assignment seeded — n is tiny and
  within-participant repeated measures are correlated. Treat the deltas as
  pilot evidence only, not as H1/H2 estimates.
- Voice recorded through speakers and re-captured acousticly may leak the
  cue into the transcript; the runtime's suppression window masks known cue
  text, but verify `--speech-db` and test before relying on this.
- analysis cannot tell whether an operation was adopted because of the cue —
  it can only tell whether the continuation contained it. A positive uptake
  here is a mechanism check, not human-reasoning efficacy, and it does not
  constitute H1 or H2.

For the H1/H2 numbers the protocol itself requires 24–32 pairs per cell of
randomized assignment with blind scoring. Please read `docs/experiment-protocol.md`.
