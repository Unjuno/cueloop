# CUELOOP

CUELOOP synchronizes an inserted cue with a person's speech while they are
still talking. It predicts the next short span, tracks how the live transcript
aligns with that prediction, and inserts a very short cue before the next
clause is produced.

The goal is to test a weak linguistic-framing hypothesis: does hearing a
specific operation word at the moment the speaker announces that operation
change what they say next, and does repeated practice improve the structure of
their reasoning? The mechanism can be measured now. Claims about durable
writing or reasoning improvement require a delayed, unassisted task.

## The loop

```text
speech events (partial / final / VAD)
        ↓
clause boundary or announced operation ("結論は", "つまり", pause)
        ↓
predict the next short span
        ↓
synchronize: match spoken text against the prediction, estimate divergence
        ↓
local decision model: does the next span omit a useful operation?
        ↓
assign a cue (experimental variable)
        ↓
speak the cue before the next clause is produced
        ↓
record continuation → operation uptake / prediction agreement
```

The two concerns are deliberately separate:

| Concern | Owner | Experimental role |
| --- | --- | --- |
| When to intervene | `RuleDecisionModel` or `LocalLLMDecisionModel` | held comparable across conditions |
| What cue is heard | `CueAssigner` | manipulation (canonical, fixed, alternate, randomized) |

A model must not choose the cue from measured effectiveness in a live
experiment. That confounds the framing effect with selection bias. Effectiveness
can rank cues offline; the experiment assigns them.

## Quickstart

```bash
python -m pip install -e .

# Deterministic replay: at the 900 ms boundary, delivers a cue 100 ms later.
cueloop-sync --events experiments/sync-smoke.jsonl \
  --log artifacts/sync-smoke.jsonl \
  --report artifacts/sync-report.json
```

Use the local decision model on an OpenAI-compatible server. The default server
is Ollama at `http://127.0.0.1:11434`; the model is chosen from the server's list.

```bash
cueloop-sync --events experiments/sync-smoke.jsonl \
  --decision local
```

Record the experimental condition in every log entry:

```bash
cueloop-sync --events experiments/sync-smoke.jsonl \
  --decision local \
  --assignment random --seed 7 \
  --condition-label CUE
```

## Demo

Run the deterministic replay in real time (about 3 seconds):

```bash
scripts/demo.sh replay
```

The script uses the local `.venv` entry point when present, speaks the cue with
macOS `say`, and writes logs under `artifacts/demo/`. Set `CUELOOP_TTS=0` to
mute, or `CUELOOP_DECISION=local` to use the local model with a rule fallback.

Run the live microphone demo (speak a sentence containing `結論は`, `つまり`,
`前提は`, or `逆に`):

```bash
scripts/demo.sh live 45
```

This requires `.build/macos-speech-events`; build it with the command in the
next section. `scripts/demo.sh live 10` is a quick check.

`cueloop-sync --realtime --events <file>` paces any replay by event timestamps;
without it, replay runs as fast as possible.

## Live macOS session

Build the microphone adapter once:

```bash
mkdir -p .build
swiftc tools/macos_speech_events.swift -o .build/macos-speech-events \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist \
  -Xlinker tools/macos_speech_events.Info.plist
```

Then run the loop. Use the JSON form for option-like arguments:

```bash
cueloop-sync \
  --event-command-json '[".build/macos-speech-events","--locale","ja-JP"]' \
  --decision local \
  --tts-command-json '["say","-v","Kyoko"]' \
  --condition-label CUE
```

The same applies to any command that contains option-like tokens; plain
`--tts-command say` works, but `say -v Kyoko` must use the JSON form.

macOS asks for Speech Recognition and microphone permission on first use.
The adapter emits the same event contract used by replay:

```json
{"type":"speech_started","at_ms":0}
{"type":"transcript_partial","at_ms":900,"text":"この設計の結論は"}
{"type":"tick","at_ms":1200}
{"type":"transcript_final","at_ms":1600,"text":"この設計の結論は、観測データでは弱い。"}
{"type":"speech_ended","at_ms":1800}
{"type":"session_ended","at_ms":2000}
```

## Prediction and decision backends

- `--predictor ngram` (default): a character n-gram model trained on a small
  Japanese corpus. It is fast enough for short pauses.
- `--predictor local`: an OpenAI-compatible model predicts the next span. It
  runs in the background with an n-gram fallback (`--async-prediction`), so a
  slow or malformed model never blocks the boundary. Small models sometimes
  fail the JSON contract; the log shows the fallback source.
- `--decision rule` (default): insert when the predicted next span omits a
  priority operation.
- `--early-lead-ms` (default 0): plan a cue before the pause threshold is
  reached, using the current prediction. This is a predicted early insert; the
  speaker may resume first, so keep the lead small and keep cancellation on.
- `--decision local`: the model sees only the missing operations and returns
  `CUE` or `SILENCE`. Invalid JSON, low confidence, a covered operation, or a
  contentless prediction fails closed to silence. With `--async-decision`
  (default) the decision is pre-computed before the boundary. If it is not
  ready, `--decision-fallback rule` (default) keeps the cue on time;
  `--decision-fallback none` waits for the local model.

The reliable low-latency live configuration is n-gram prediction plus the local
decision model with async decision. Measured on an M-series laptop with Ollama,
a 1.5B decision model adds roughly 0.2–0.6 s; `--insertion-delay-ms` defaults to
100 ms.

There is no fixed delivery-latency target. The smoke fixture has a boundary at
900 ms and delivers at 1000 ms because the insertion delay is 100 ms; that is a
property of the fixture, not a goal. The meaningful timing is relative to the
announced operation: the cue should start at the boundary and land before the
next clause is produced. The delivery log records `boundary_at_ms` and
`boundary_to_delivery_ms` so the timing can be checked against the transcript
(`text_at_delivery`) instead of optimized for a round number.

Real-time insertion exists because the hypothesis is anticipatory: an external
cue must frame the clause that is about to be produced. A later summary or
post-hoc suggestion would test a different mechanism. The experiment asks
whether hearing an operation the speaker did not produce changes the next
clause, and whether that externally supplied operation carries into later,
uncued speech or a delayed reconstruction.

## Output and measurement

`--log` is append-only JSONL:

- `prediction` — span, operations, confidence, source, speech rate, latency
- `intervention` includes `prediction_penalty_ms` and `prediction_precomputed`
  so async prediction latency is not confused with the delivery budget
- `intervention` — requested operation, cue, assignment, schedule, model reason,
  `decision_source` (`precomputed`/`fallback`/`synchronous`), `precomputed`
- `delivery` — `DELIVERED`, `CANCELLED`, or `EXPIRED`
- `outcome` — continuation window, cue uptake, prediction agreement

`--report` aggregates deliveries, cancellations, and per-operation uptake. The
`summary` block also reports prediction agreement and `novel_uptake`: whether a
cued operation that was absent before the cue appears in the continuation. That
is the closest current mechanism check for the external-framing hypothesis; it
is still not evidence of durable writing or reasoning improvement. Carryover
into later, uncued speech and the delayed reconstruction remain the outcomes
that can support that claim.

## Experiment

Use `SILENCE`, `NEUTRAL`, and `CUE` conditions with the same predictor,
boundary detector, assignment policy, and timing. Only the spoken content
differs. The primary outcome is operation uptake in the continuation window.
The secondary outcome is a delayed, unassisted reconstruction scored by a
blinded rubric. See `docs/experiment-protocol.md` and `docs/sync-core.md`.

## Failure behavior

- Predictor or decision failure returns no cue; the runtime stays silent.
- A cue is cancelled when the speaker diverges from the prediction before
  delivery, when speech ends, or when the participant pauses interventions.
- A cue not delivered inside its window is `EXPIRED`.
- TTS echo is suppressed for the measured (or, in async mode, estimated) cue
  duration plus 250 ms.
- A speaker command failure is logged as `SUPPRESSED`, not delivered.
- Context is capped; long sessions do not grow the prompt without bound.

## Repository layout

```text
src/cueloop/sync/       event loop, predictors, policy, runtime, CLI
scripts/demo.sh         replay and live microphone demo
tools/macos_speech_events.swift  macOS ASR adapter
tests/                  deterministic unit and replay tests
experiments/            replay fixtures
docs/                   design and experiment protocol
```

The earlier turn-based stack has been removed. The sync core is the primary
path; the event contract and macOS adapter are retained.
