# Sync Core

The sync core replaces turn-based response with predictive insertion. The
system does not wait for silence and then reply. It predicts what the speaker
is about to say, tracks how the live transcript aligns with that prediction,
and inserts a very short cue while the speaker is still talking.

## Loop

```text
speech events (partial / final / VAD)
        ↓
clause boundary or announced operation ("結論は", "つまり", pause)
        ↓
predict the next short span
        ↓
synchronize: match spoken text against the prediction, estimate divergence
        ↓
decide whether the next span omits a useful operation
        ↓
assign a cue (experimental variable)
        ↓
speak the cue before the next clause is produced
        ↓
record continuation → operation uptake / prediction agreement
```

## Why prediction rather than classification

A turn-level classifier can only react after the speaker has finished. A
predictor can act at the moment the speaker announces an operation:

```text
speaker: "この設計の結論は"
cue:                         "根拠は？"
speaker: "……観測データではまだ弱い。"
```

The cue lands between the announcement and its content. That is the
linguistic-framing intervention the project is trying to measure.

## Components

- `cueloop.sync.models` — events, predictions, interventions, deliveries,
  outcomes
- `cueloop.sync.synchronizer` — clause boundaries, trigger phrases, divergence
- `cueloop.sync.predictors` — `NgramPredictor`, `LocalLLMPredictor`,
  `AsyncPredictor`, `ScriptedPredictor`
- `cueloop.sync.policy` — `RuleDecisionModel`, `LocalLLMDecisionModel`,
  `AsyncDecisionModel`, and cue assignment
- `cueloop.sync.runtime` — event loop, delivery, cancellation, echo
  suppression, logging
- `cueloop.sync.evaluate` — uptake and prediction-agreement summary
- `cueloop.sync.cli` — `cueloop-sync` entry point

## Two separable concerns

| Concern | Owner | Experimental role |
| --- | --- | --- |
| When to intervene | `RuleDecisionModel` or `LocalLLMDecisionModel` | held comparable across conditions |
| What cue is heard | `CueAssigner` | manipulation (canonical, fixed, alternate, randomized) |

Do not let a model choose the cue from measured effectiveness in a live
experiment. That confounds the cue's framing effect with the model's selection
bias. Effectiveness can rank cues offline; the experiment should assign them.

## Decision models

`RuleDecisionModel` inserts when the predicted next span omits the highest
priority operation not already present in the clause or recent deliveries.

`LocalLLMDecisionModel` sends the local model a narrowed prompt containing only
the missing operations. It returns one of:

```json
{"action": "CUE", "operation": "ground", "cue": "根拠は？", "confidence": 0.8, "reason": "..."}
{"action": "SILENCE", "confidence": 0.2, "reason": "..."}
```

The runtime fails closed on invalid JSON, low confidence, a covered operation,
a contentless prediction, or a cue that does not map to an operation. The
`operation` field in the log is the operation the cue actually performs;
`requested_operation` is what the decision model asked for.

`AsyncDecisionModel` pre-computes the local decision while the speaker is
still producing the clause. The runtime submits the state on every partial
that has a prediction, and at the boundary takes the newest completed decision
whose text is still a prefix of the utterance. The intervention is rebased to
the actual boundary, so the cue is no longer delayed by the model call. If no
pre-computed decision is ready, `--decision-fallback rule` (the default) uses
the rule decision at the boundary; `--decision-fallback none` waits for the
local model and accepts the extra latency. The log records `decision_source`
(`precomputed`, `fallback`, or `synchronous`) and `precomputed`. Use the
fallback for latency-sensitive live runs and for condition-balanced
experiments; use `none` when the pure local decision is the object of study.

Model selection is server-driven. The decision preference is a 1.5B instruct
model; the prediction preference is a mid-size model. The exact identifiers are
resolved from `/v1/models` with separator normalization, so Ollama, LM Studio,
and vLLM identifiers all match.

## Prediction and latency

The n-gram predictor is the default because a boundary pause is short. A local
language model can take longer than the pause. `AsyncPredictor` submits the
current text on every partial and takes the most recent completed prediction
whose base text is still a prefix of the current utterance; otherwise it falls
back to the n-gram model. The log's `source` records which one was used, and
`latency_ms` is the model's compute time, not the time spent waiting. A
pre-computed prediction adds no scheduling penalty (`prediction_penalty_ms`
is 0); only a prediction computed on the boundary is added to the schedule.

`--early-lead-ms` plans a cue a little before the pause threshold is reached.
It is a predicted early insert, not a confirmed boundary: the state is built
from the current clause and the current prediction, and a small lead is
recommended because the speaker can resume and diverge. With a confirmed
boundary the cue is scheduled at the boundary; with early insertion it is
scheduled at the tick that enters the lead window. The feature only fires when
a prediction exists.

The reliable live configuration is n-gram prediction plus the local decision
model. The local decision model adds roughly 0.2–0.6 s for a 1.5B model on an
M-series laptop. The default insertion delay is 100 ms. Without a speculative
decision the cue is scheduled at the boundary plus the decision and predictor
latency plus the insertion delay; with `AsyncDecisionModel` and a ready
decision, the model latency is removed from the schedule. If the speaker has
already moved on, the delivery is cancelled as divergent. With async
prediction and a ready result, the predictor does not consume the pause budget
either.

The smoke replay originally delivered at 1400–1500 ms with an unprepared
local decision. With async prediction, async decision, and the rule fallback,
the same 900 ms boundary delivers at 1000 ms; the local model is used when its
pre-computed result exists, otherwise the rule fallback keeps the timing
budget.

## Run

Replay a recorded session:

```bash
cueloop-sync --events experiments/sync-smoke.jsonl \
  --log artifacts/sync-smoke.jsonl \
  --report artifacts/sync-report.json
```

Add `--realtime` to pace the replay by event timestamps instead of running as
fast as possible. `scripts/demo.sh replay` wraps this for a short audible demo,
and `scripts/demo.sh live 45` runs the macOS microphone loop.

Use a local OpenAI-compatible model for the decision:

```bash
cueloop-sync --events experiments/sync-smoke.jsonl \
  --decision local \
  --base-url http://127.0.0.1:11434
```

Use a local model for prediction as well (best effort, with an n-gram fallback):

```bash
cueloop-sync --events experiments/sync-smoke.jsonl \
  --predictor local \
  --base-url http://127.0.0.1:11434 \
  --model qwen2.5:1.5b-instruct
```

Add audible cues with a TTS command. The cue is appended as one argument:

```bash
cueloop-sync --events experiments/sync-smoke.jsonl \
  --tts-command-json '["say","-v","Kyoko"]' \
  --tts-async
```

`--tts-async` (default) starts the cue without blocking the event loop, so
partial transcripts keep arriving during playback. Because the process has not
finished, the runtime suppresses echo for a conservative estimate
(`1000 ms + 150 ms per cue character`, capped at 8 s). `--no-tts-async` waits
and uses the measured duration.

Experimental controls:

```bash
cueloop-sync --events experiments/sync-smoke.jsonl \
  --assignment random --seed 7 \
  --condition-label CUE \
  --insertion-delay-ms 100 \
  --cooldown-ms 2500 \
  --min-confidence 0.55 \
  --replan-on-divergence \
  --min-replan-interval-ms 1000
```

## Events

The sync core consumes the same JSONL event contract as the macOS adapter:

```json
{"type":"speech_started","at_ms":0}
{"type":"transcript_partial","at_ms":900,"text":"この設計の結論は"}
{"type":"tick","at_ms":1200}
{"type":"transcript_final","at_ms":1600,"text":"この設計の結論は、観測データでは弱い。"}
{"type":"speech_ended","at_ms":1800}
{"type":"session_ended","at_ms":2000}
```

`transcript_partial` is treated as the current utterance text: a new value
replaces the previous partial. `tick` drives pause boundaries and delivery.
`interventions_paused`, `interventions_resumed`, and `participant_stop` are
honored as controls.

## Output

`--log` is an append-only JSONL record of:

- `prediction` — predicted span, operations, confidence, source, speech rate,
  base length, compute latency
- `intervention` — requested operation, delivered cue, assignment, schedule,
  prediction and decision latency
- `delivery` — `DELIVERED`, `CANCELLED`, or `EXPIRED`
- `outcome` — continuation window, cue uptake, prediction agreement

`--report` aggregates deliveries, cancellations, and per-operation uptake. The
`summary` block reports prediction agreement. Every log entry carries the
`condition` label.

## Failure behavior

- A predictor failure returns no prediction; the runtime stays silent.
- A local decision failure returns no cue.
- A cue is cancelled when the speaker diverges from the prediction before
  delivery, when speech ends, or when the participant pauses interventions.
- A cue that is not delivered inside its window is marked `EXPIRED`.
- A speaker command failure is logged as `SUPPRESSED`, not as delivered.
- After a TTS cue, transcript events inside the measured or estimated speech
  duration plus 250 ms are ignored, so the cue is not mistaken for the
  participant's words.
- Observations are capped by the context window; long sessions do not grow
  the prompt without bound.

## What is not measured

Uptake and prediction agreement are mechanism checks. They do not establish
that a person's writing or reasoning improved. That still requires a delayed,
unassisted reconstruction task with blinded scoring. See
[`experiment-protocol.md`](experiment-protocol.md).

## Framing experiment

The sync core exists to test a weak linguistic-framing hypothesis, not strong
linguistic determinism:

> Inserting a short cue at the point where a speaker announces an operation
> changes the operation distribution of the next clause.

### Conditions

- `SILENCE`: predict, but do not speak.
- `NEUTRAL`: speak a length-matched neutral filler at the same boundary.
- `CUE`: speak the assigned operation cue at the same boundary.

Randomize the cue wording and condition at the clause level within a person.
Use the same predictor, the same boundary detector, and the same timing in all
conditions. Only the spoken content differs.

### Primary outcome

The operation performed in the continuation window (default 8 seconds), read
with the same vocabulary tagger used by the policy. A cue is effective for an
operation when its uptake rate exceeds the neutral and silence conditions.

### Secondary outcome

A delayed, unassisted reconstruction scored by a blinded rubric (claims,
premises, dependencies, counterexamples, unresolved questions, unsupported
claims). This is the only outcome that can speak to retained capability.

### Controls and limits

- Report clause-level paired differences with a seeded bootstrap interval.
- Keep task, speech rate, cue position, and ASR latency comparable.
- The vocabulary tagger is a proxy; manual adjudication is required for a
  publication claim.
- A null result is a valid result and must be reported.
