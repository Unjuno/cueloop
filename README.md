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
record continuation → uptake, carryover / prediction agreement
```

The two concerns are deliberately separate:

| Concern | Owner | Experimental role |
| --- | --- | --- |
| When to intervene | `RuleDecisionModel`, `LinearDecisionModel`, or `LocalLLMDecisionModel` | held comparable across conditions |
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

The local LLM is a semantic prior, but it is too slow for every boundary. A
fast linear policy can be trained offline from the same logs:

```bash
cueloop-sync --events experiments/sync-smoke.jsonl --decision local \
  --log logs/session-001.jsonl
scripts/train_policy.py logs/*.jsonl \
  --label carryover --output artifacts/sync-policy.json
cueloop-sync --events experiments/sync-smoke.jsonl --decision linear
```

The linear backend scores only the operations the rule policy considers
missing. It is a dot product plus a threshold, so it does not add LLM latency
at the boundary.

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
When the speaker is a `say` voice tuned to air-conduction audio, the mic hears
the cue too; the runtime suppresses the cue's text for a short window after
playback, but an *inanimate* talker cannot react to cues, so yield is
structurally zero and intervention counts may differ from the SILENCE arm.
`--adapter-args-json '["--speech-db","-50"]'` lowers the adapter's speech
detection threshold for such weak-signal loops; it is not needed when speaking
directly into the microphone.

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
- `--decision linear`: score the missing operations with a small linear model
  trained offline from `intervention`, `outcome`, and `carryover` records.
  `--policy-model` defaults to `artifacts/sync-policy.json`. Train it with
  `scripts/train_policy.py`; the default reward is `carryover`, and `uptake`
  and `blend` are also available. If the model predicts no carryover above its
  threshold, the runtime stays silent. This is a contextual reward model, not
  full RL: causal claims still require randomized cues and exploration.
- `--early-lead-ms` (default 0): plan a cue before the pause threshold is
  reached, using the current prediction. This is a predicted early insert; the
  speaker may resume first, so keep the lead small and keep cancellation on.
- `--decision local`: the model sees only the missing operations and returns
  `CUE` or `SILENCE`. Invalid JSON, low confidence, a covered operation, or a
  contentless prediction fails closed to silence. With `--async-decision`
  (default) the decision is pre-computed before the boundary. If it is not
  ready, `--decision-fallback rule` (default) keeps the cue on time;
  `--decision-fallback none` waits for the local model.

The reliable low-latency live configuration is n-gram prediction plus a fast
decision backend. A local 1.5B decision model measured on an M-series laptop
adds roughly 0.2–0.6 s even with async pre-computation; that is why the linear
policy exists. `--insertion-delay-ms` defaults to 100 ms.

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
- `carryover` — cued operation in later uncued text, emitted at session end

`--report` aggregates deliveries, cancellations, and per-operation uptake. The
`summary` block also reports prediction agreement, `novel_uptake` (a cued
operation absent before the cue that then appears in the continuation), and
`carryover` (the operation appearing in later uncued speech), and
`novel_carryover` (carryover for an operation absent before the cue).
`repertoire_*`
lists the operations observed before the first cue, by session end, and the
growth between them.

Immediate uptake is only the mechanism check. It is not evidence of durable
writing or reasoning improvement. Carryover is a stronger signal but still not
proof: a delayed unassisted reconstruction scored by a blinded rubric remains
the outcome that can support the claim.

## Policy comparison experiment

Use the same predictor, boundary detector, assignment policy, and timing for
each backend. Only the decision backend changes. The experiment runner records
a live session with randomized cue wording and writes a manifest:

```bash
scripts/run_experiment.py --mode live --condition CUE \
  --decision rule --duration 60 --seed 1 --session rule-01
scripts/run_experiment.py --mode live --condition CUE \
  --decision linear --duration 60 --seed 1 --session linear-01
scripts/run_experiment.py --mode live --condition CUE \
  --decision local --duration 60 --seed 1 --session llm-01
```

`--condition SILENCE` records the same timing decisions without speaking the
cue. `linear` uses `--policy-model` (default `artifacts/sync-policy.json`).
Replay is available for smoke tests with `--mode replay`; it defaults to TTS
off and to unpaced timing, so paired replay runs are deterministic. Pass
`--realtime` to pace a replay by its event timestamps.

### Exploration arm: randomize the operation

`--assignment random` randomizes wording only; the policy still chooses the
operation, so it must not be used for operation-level causal comparisons. To
compare operations directly, use the exploration arm:

```bash
scripts/run_experiment.py --mode replay --assignment random-operation \
  --condition CUE --decision rule --seed 1 \
  --events experiments/sync-carryover-smoke.jsonl --outcome-window-ms 1000 \
  --output-dir logs/experiment-explore-01 --session rule-cue-1
```

The wrapped policy still decides *when* to fire. When it fires, one operation is
drawn uniformly from the same missing-operation set it considered, so
`requested_operation` (what the policy wanted) and `operation` (what was heard)
differ. The draw is reproducible from `--seed`, and the assignment string
records both the seed and the wording strategy.

### SILENCE control and paired analysis

Run the same fixture, seed, and timing with `--condition SILENCE`. The cue is
held rather than spoken: the report shows `delivered=0` and `held=1`, and the
decision is still logged, so the timing and the operation draw stay paired.
Paired runs share `--pair-key` (derived as
`<decision>-seed<seed>-<fixture>`), which is what the analysis uses:

```bash
scripts/analyze_experiment.py logs/experiment-explore-01/reports \
  --json artifacts/experiment-analysis.json \
  --markdown artifacts/experiment-analysis.md
```

Sessions are clustered by pair. The tool reports per-condition aggregates and
the paired difference (CUE - SILENCE) with a seeded percentile bootstrap
(`--resamples`, default 2000; `--seed`; `--confidence`, default 0.95). An
interval is reported only when at least two complete pairs exist; otherwise
`ci_low` and `ci_high` are `"insufficient"`. A policy with no CUE/SILENCE pair
is reported as `"not_comparable"` rather than as a zero effect.

Replay deltas are zero by construction because the transcript is fixed: cueing
cannot change a static fixture. Zero deltas from replay show that the pipeline
works, not that the cue had no effect.

### Synthetic calibration (not human evidence)

To show the analysis can recover a *known* effect, generate paired reports
from a generative speaker model with a programmed cue effect:

```bash
scripts/synthesize_experiment.py --output-dir logs/synthetic-effect030 \
  --sessions 24 --clauses 8 --seed 0 --uptake-effect 0.30
scripts/analyze_experiment.py logs/synthetic-effect030 \
  --json artifacts/synthetic-effect030.json
```

The CUE and SILENCE sessions of a pair share the missing-operation path and
per-boundary uniforms (common random numbers); only the programmed effect
differs. With `--uptake-effect 0.30` and 24 pairs, the analyzer reports
`+0.229` to `+0.328` for uptake across rule/linear/llm with CIs excluding 0;
with `--uptake-effect 0.0` it reports exactly `+0.000`. Delays are paired by
construction. These are calibration results of the pipeline, not evidence
about human speech. All parameters are recorded in the output manifest.

Aggregate the report files by policy family:

```bash
scripts/compare_policies.py logs/experiment/reports \
  --json artifacts/policy-comparison.json \
  --markdown artifacts/policy-comparison.md
```

The table reports uptake, novel uptake, carryover, novel carryover, repertoire
growth, and boundary-to-delivery delay for `rule`, `linear`, and `llm`. The
linear model is trained from the recorded interventions:

```bash
scripts/train_policy.py logs/*.jsonl \
  --label carryover --output artifacts/sync-policy.json \
  --report artifacts/sync-policy-eval.json
scripts/evaluate_policy.py logs/*.jsonl \
  --model artifacts/sync-policy.json \
  --report artifacts/sync-policy-holdout.json
```

`train_policy.py` holds out a stratified fraction (`--eval-fraction`, default
0.25) and writes accuracy, precision, recall, F1, ROC-AUC, and Brier score for
both the training split and the held-out split. `evaluate_policy.py` scores a
saved model against any later logs, which is how the policy is checked on a new
session before it is used live. All metrics describe logged actions only, so
they are not an off-policy estimate of the deployed policy.

Immediate uptake is the mechanism check; carryover and delayed reconstruction
are the persistence outcomes. See `docs/experiment-protocol.md` and
`docs/sync-core.md`.

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
scripts/run_experiment.py live/replay policy-comparison sessions
scripts/train_policy.py offline linear policy trainer + holdout metrics
scripts/evaluate_policy.py score a saved policy on recorded logs
scripts/compare_policies.py aggregate rule/linear/LLM reports
scripts/analyze_experiment.py paired CUE/SILENCE analysis with bootstrap CI
tools/macos_speech_events.swift  macOS ASR adapter
tests/                  deterministic unit and replay tests
experiments/            replay fixtures
docs/                   design and experiment protocol
```

The earlier turn-based stack has been removed. The sync core is the primary
path; the event contract and macOS adapter are retained.
