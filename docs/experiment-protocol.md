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

## Why the timing is boundary-relative

The intervention is anticipatory. The cue must be heard at the moment the
speaker announces an operation, before the next clause is produced. A fixed
delivery latency is not the target: the same absolute millisecond can be early
in one pause and late in another. Record `boundary_at_ms`,
`boundary_to_delivery_ms`, and `text_at_delivery`; exclude deliveries that
arrive after the next clause begins.

The hypothesis is not that the cue summarizes the speaker's own thought. It is
that an externally supplied operation the speaker did not produce can enter the
next clause and, with repetition, change the operations available in later
uncued speech. Immediate uptake is necessary but not sufficient.

## Immediate outcome: operation uptake

For each delivery, take the continuation window (default 8 s) and tag its
operations with the same vocabulary used by the policy. Define **uptake** as
the cued operation appearing in the continuation.

Also report `novel_uptake`: uptake of a cued operation that was absent from the
speaker's text before the cue. This is the mechanism check for an external
frame, as opposed to a cue that merely repeats what the speaker already said.

## Primary persistence outcome: carryover

Uptake inside the outcome window still measures immediate compliance. Define
**carryover** as the cued operation appearing after the outcome window and
before the next cue for that operation. This excludes text that a later cue
could explain. Report `carryover_rate = carryover / delivered` and compare it
across conditions as above.

Carryover is stronger evidence than uptake, but it is not proof of a durable
change: the operation may be repeated because of the task, not because the
speaker's operational repertoire changed. Report `novel_carryover` separately:
carryover of an operation that was absent before the cue. Also report
`repertoire_growth`, the operations observed by session end that were absent
before the first cue.

For each operation, compare:

```text
P(uptake | CUE) - P(uptake | NEUTRAL)
P(uptake | CUE) - P(uptake | SILENCE)
```

Repeat the same contrasts for carryover. Use the delayed reconstruction as the
final test of persistence.

## Recording comparison sessions

Use the same predictor, boundary detector, timing, and assignment policy for
each decision backend. `run_experiment.py` records the session, writes the
report, and appends a manifest entry:

```bash
scripts/run_experiment.py --mode live --condition CUE \
  --decision rule --duration 60 --seed 1 --session rule-01
scripts/run_experiment.py --mode live --condition CUE \
  --decision linear --duration 60 --seed 1 --session linear-01
scripts/run_experiment.py --mode live --condition CUE \
  --decision local --duration 60 --seed 1 --session llm-01
```

`--assignment random` randomizes the cue wording for the operation selected by
the policy; it does not randomize the operation itself. That limits the
comparison to correlation between the policy's choices and outcomes.
`--assignment random-operation` is the exploration arm: the wrapped policy keeps
its timing decision, but the delivered operation is drawn uniformly from the
same missing-operation set. `requested_operation` records the policy's choice
and `operation` records what the participant heard, so the two can be analysed
separately. The draw is reproducible from `--seed`.

`--condition SILENCE` keeps the same fixture, seed, and decision timing but
holds the cue. The delivery is logged with status `SILENT`, the report shows
`delivered=0` and `held=1`, and an outcome is still recorded so the no-cue
continuation is measured the same way as the cued one. CUE and SILENCE runs of
one comparison share `--pair-key` (default
`<decision>-seed<seed>-<fixture>`).

Replay (`--mode replay`) is available for smoke tests and defaults to muted TTS
and unpaced timing, so paired replays are deterministic; pass `--realtime` only
when pacing itself is under test. Live runs are always paced by the microphone.

Every JSONL record carries `condition`, `decision_model`, and `predictor`, so
reports can be grouped without relying on filenames. The runner writes
`logs/experiment/manifest.jsonl`, `logs/experiment/logs/*.jsonl`, and
`logs/experiment/reports/*.json`.

## Comparing backends

```bash
scripts/compare_policies.py logs/experiment/reports \
  --json artifacts/policy-comparison.json \
  --markdown artifacts/policy-comparison.md
```

The aggregate groups reports into `rule`, `linear`, and `llm` families and
reports delivered count, uptake, novel uptake, carryover, novel carryover,
repertoire growth, and median boundary-to-delivery delay. Treat the report as
descriptive until the paired analysis and the delayed-reconstruction analysis
are complete.

## Paired CUE/SILENCE analysis

```bash
scripts/analyze_experiment.py logs/experiment-explore-01/reports \
  --json artifacts/experiment-analysis.json \
  --markdown artifacts/experiment-analysis.md
```

Sessions are the unit of analysis and are clustered by `pair_key`. For each
metric the script computes the per-pair difference (CUE - SILENCE), then takes
the mean and a percentile bootstrap interval over pairs. Resamples (default
2000), confidence (default 0.95), and seed are written to the JSON, and the
output contains no wall-clock timestamps, so rerunning on the same reports is
byte-identical.

Read the output as follows:

- `status: "ok"` with `n_pairs >= 2` gives a point estimate and an interval.
- `ci_low`/`ci_high` of `"insufficient"` means fewer than two complete pairs;
  the point estimate is shown but no interval is claimed.
- `status: "not_comparable"` means the policy has no CUE/SILENCE pair with the
  same `pair_key`; this is not evidence of a zero effect.

Replay reports have zero deltas by construction: the transcript is a fixed
file, so a spoken cue cannot change what is said next. Zero deltas there show
that the pipeline and the pairing work, not that cues have no effect. Only live
sessions can estimate an effect.

## Synthetic calibration of the analysis

`scripts/synthesize_experiment.py` writes paired reports from a generative
speaker with a programmed cue effect. The cue effect acts on the delivered
operation; requested and delivered operations diverge in the exploration arm,
so CUE and SILENCE share the boundary skeleton and the per-boundary uniforms
(common random numbers), and only the effect term differs. Speaker state
(missing-operation sets) evolves from the outcomes, so downstream clauses
diverge causally — that is the effect being measured.

Commands and coverage:

```bash
for effect in 0 0.10 0.20 0.30; do
  scripts/synthesize_experiment.py --output-dir logs/synthetic-effect${effect} \
    --sessions 24 --clauses 8 --seed 0 --uptake-effect "$effect" --carryover-effect "$effect"
done
```

Measured on the generated reports (24 pairs per backend, seed 0):

| programmed effect | backend | uptake delta (95% CI) |
| --- | --- | --- |
| 0.00 | rule, linear, llm | +0.000 (all exact) |
| 0.10 | rule / linear / llm | +0.130 [+0.083, +0.182] / +0.125 [+0.083, +0.167] / +0.078 [+0.047, +0.109] |
| 0.20 | rule / linear / llm | +0.240 [+0.182, +0.302] / +0.229 [+0.172, +0.292] / +0.161 [+0.115, +0.208] |
| 0.30 | rule / linear / llm | +0.328 [+0.266, +0.391] / +0.323 [+0.260, +0.385] / +0.229 [+0.182, +0.271] |

A second seed (1) at effect 0.20 reproduces the pattern (rule +0.156,
linear +0.167, llm +0.156), and `--assignment requested` at effect 0.30 with
16 pairs yields +0.289 to +0.312 with CIs excluding 0. Delay deltas are
`+0.000` everywhere by construction. The null arm (`--uptake-effect 0.0`) is
the pipeline's calibration reference: the analyzer reports exactly zero, not
approximately zero.

**These are synthetic numbers.** The cue effect is a programming choice, not
a measurement of people. Use this arm to test the analyzer (detection power,
pairing correctness, CI sanity), never to estimate how cues affect a human
speaker. Every synthetic report and manifest carries `synthetic: true` and a
note saying so.

## Training the fast intervention policy

Every intervention log now contains the observation features used by the
decision model, and every session emits a `carryover` record. The linear policy
is trained from these tuples:

```bash
scripts/train_policy.py logs/*.jsonl --label carryover \
  --output artifacts/sync-policy.json --report artifacts/sync-policy-eval.json
scripts/evaluate_policy.py logs/*.jsonl --model artifacts/sync-policy.json
cueloop-sync ... --decision linear --policy-model artifacts/sync-policy.json
```

Training holds out a stratified fraction of the samples and reports accuracy,
precision, recall, F1, ROC-AUC, and Brier score for both splits. Because the
held-out set is small with a single session, treat the metrics as a regression
check, not as generalisation evidence; `evaluate_policy.py` re-scores the saved
model on new logs.

Exploration and paired runs are safe to concatenate before training: a CUE
session and its SILENCE twin draw the identical `intervention_id`, so both
`train_policy.py` and `evaluate_policy.py` namespace ids per log file before
joining deliveries and outcomes. Assignment strings are summarized and unknown
ones are reported on stderr and in the training summary under
`assignments` / `unknown_assignments`; nothing is silently discarded. The
reward always attaches to the operation actually delivered, never to
`requested_operation`. Assignment is deliberately not a model feature; if it
ever becomes one, bump `MODEL_VERSION` and re-validate like any other schema
change.

The model predicts carryover for a candidate operation and stays silent below
its threshold. It is deliberately a reward model over the actions that were
actually logged. That creates selection bias: untried operations have no
outcome. Exploration logs train on the operation that was actually delivered,
so they widen coverage but still cannot identify the counterfactual for
operations that were never drawn. For causal claims, run the exploration arm
(`--assignment random-operation`), keep the SILENCE control, and do not present
the fitted model's score as an off-policy performance estimate. The local LLM can remain the semantic prior or the
exploration policy; the linear model is the fast exploitation path.

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
