# Experiment Protocol

This protocol tests whether a short cue inserted during speech shifts the
operation the speaker performs next, and whether repeated cueing changes a
delayed, unassisted reconstruction. It does not assume linguistic determinism.

## Hypotheses

- **H1 (mechanism).** At an announced operation boundary, a cue increases the
  probability that the next clause performs the cued operation relative to
  silence and a length-matched neutral filler.
- **H2 (retention).** Repeated cueing improves a delayed, unassisted
  reconstruction scored by a blinded rubric.
- **H0.** Cueing changes neither the next operation nor the delayed
  reconstruction beyond the neutral condition.

H1 and H2 are separate. A positive H1 with a null H2 means the cue changed the
local continuation but did not transfer. Report both.

## Design

Within-participant, clause-level randomization. Every eligible boundary is
assigned to one of:

- `SILENCE`: predict and log, but do not speak.
- `NEUTRAL`: speak a length-matched neutral filler at the same boundary.
- `CUE`: speak the operation cue assigned for that clause.

Hold constant across conditions:

- predictor and decision model (including `--async-decision` and
  `--decision-fallback`; both are part of the timing condition)
- boundary detector, pause threshold, and trigger phrases
- insertion delay and delivery window
- cue assignment policy (the cue wording is randomized within `CUE`)
- task prompt, microphone, and ASR locale

Only the spoken content varies.

## Procedure

1. Consent, audio/transcript retention choice, and a microphone check.
2. A short practice task with cues disabled.
3. Three timed speaking tasks of comparable difficulty, counterbalanced.
4. A filler task (5–10 minutes) before reconstruction.
5. A delayed, unassisted written reconstruction of each task.
6. A short debrief; allow withdrawal and deletion.

The system must remain silent when the predictor or decision model fails. A
failed decision is not a `SILENCE` condition; log it separately and exclude it
from the condition comparison.

## Primary outcome: operation uptake

For each delivery, take the continuation window (default 8 s) and tag its
operations with the same vocabulary used by the policy. Define **uptake** as
the cued operation appearing in the continuation.

For each operation, compare:

```text
P(uptake | CUE) - P(uptake | NEUTRAL)
P(uptake | CUE) - P(uptake | SILENCE)
```

Use clause-level paired differences and a seeded bootstrap interval. Report
the number of eligible boundaries, deliveries, cancellations, expirations, and
failed decisions. Do not drop cancelled cues silently; cancellation is a
timing outcome, not a random omission.

## Secondary outcome: delayed reconstruction

Score the delayed reconstruction with a rubric fixed before data collection:

- explicit claims
- premises and evidence
- dependencies between claims
- counterexamples or alternative explanations
- unresolved questions
- unsupported assertions
- global coherence

Two raters blind to condition score each reconstruction. Report inter-rater
agreement; adjudicate disagreements with a third rater. Pre-register the rubric
and the primary comparison.

## Controls

- Cue wording is assigned, not selected by a model from live effectiveness.
- Length-match `NEUTRAL` to `CUE` so duration is not confounded.
- Randomize condition within person and clause, not by session.
- Counterbalance task order.
- Record speech rate, pause structure, ASR confidence, delivery latency, and
  `decision_source` (precomputed/fallback/synchronous).
- Exclude boundaries whose delivery arrives after the next clause begins.
- Keep the decision model and its fallback policy identical across conditions;
  if the model is stochastic, fix its temperature and seed.
- Log the model identifier and version.

## Analysis

Primary: mixed-effects logistic regression of uptake on condition, with random
intercepts for participant and task, and fixed effects for speech rate,
operation, and cue position. Report the `CUE - NEUTRAL` contrast as the
focal estimate.

Secondary: the same structure for rubric total and subscores. Report effect
sizes with intervals, not only p-values.

## Stopping and exclusions

Pre-register the target sample and stopping rule. Exclude a session only for a
documented technical failure (ASR unavailable, TTS failure, audio loss).
Report all exclusions and their reasons. A null result is a valid result.

## Threats to validity

- The operation tagger is a proxy; manual adjudication is required for any
  publication claim.
- Cueing may change local wording without changing reasoning quality.
- Repeated exposure may create demand characteristics; the neutral condition
  controls for speaking at the boundary but not for knowing the hypothesis.
- A single session cannot establish durable learning.
- The local model's latency can bias which boundaries are eligible; log and
  report latency and cancellation rates by condition.
