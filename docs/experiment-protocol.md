# Experiment Protocol

Two distinct claims require two distinct experiments. The automated replay validates software policy behavior. Only a preregistered human study can test whether the intervention improves retained reasoning.

## Experiment A: deterministic policy replay

Hypothesis: controller suppression reduces harmful interruptions relative to delivering analyzer proposals directly, without producing excessive missed interventions.

Run:

```bash
cueloop-experiment \
  --dataset experiments/policy-eval.jsonl \
  --report artifacts/policy-experiment.json \
  --policy-out artifacts/policy.json
```

The dataset has separate `tune` and `eval` sessions. Candidate settings are selected on `tune`. A setting is adopted only if it improves tuning loss, does not increase harmful interruptions, and does not regress held-out evaluation loss or harmful interruptions. A harmful interruption has weight 3, a wrong intervention weight 2, and a missed intervention weight 1.

This test checks implementation behavior against human-authored labels. It cannot demonstrate cognitive enhancement.

## Experiment B: within-person retained-reasoning pilot

Primary question: after assistance is removed, does minimal intervention improve the participant's retained reasoning structure compared with silent capture?

### Conditions

- `CONTROL`: transcript capture; the system always stays silent.
- `CUELOOP`: the same capture path with the deterministic controller enabled.

Use matched tasks of comparable difficulty and counterbalance condition order. Keep ASR, duration, environment, and source material constant. A practical pilot is six sessions per participant, three per condition, on separate matched tasks.

### Primary outcome

Ten minutes after each session, with CUELOOP removed, ask the participant to reconstruct the argument or decision. Blindly score a fixed rubric:

- explicit claims
- relevant premises
- valid dependencies
- counterexamples considered
- calibrated unresolved questions
- unsupported factual claims

The primary endpoint is the difference in delayed rubric score between conditions. Do not substitute the assisted-session output for this endpoint.

The analyzer reports each participant's paired difference, a seeded bootstrap 95% interval, and a paired sign-randomization p-value. It marks the primary endpoint confirmatory only when every enrolled participant has at least three sessions per condition, session duration is 9.5–10.5 minutes, recall delay is 9–11 minutes, every scorer was blinded, nobody requested a stop, and no ASR/analyzer failure occurred. Smaller or deviating runs remain exploratory and retain explicit deviation reasons.

### Safety and interaction outcomes

- interruptions during productive speech
- interventions rated disruptive
- intervention count per minute
- time to resume self-generation
- participant-requested stops
- ASR and analyzer failures

### Stop rules

Stop the session on participant request, privacy uncertainty, repeated interruption during speech, or any indication that the tool is being interpreted as diagnosis or therapy. Stop policy rollout if harmful interruptions increase on held-out evaluation.

### Interpretation

A small personal pilot can identify failure modes and estimate variance. It cannot establish general cognitive improvement. Report null and negative results, preserve raw task-level outcomes, and distinguish exploratory changes from preregistered analyses.

## Pilot tooling

Create a seeded, counterbalanced assignment:

```bash
cueloop-pilot assign \
  --participants p1 p2 \
  --tasks task-a task-b task-c task-d \
  --output artifacts/pilot-assignment.json
```

Record one JSONL object per completed session using the fields demonstrated in `experiments/pilot-synthetic.jsonl`, then analyze it:

```bash
cueloop-pilot analyze experiments/pilot-results.jsonl \
  --assignment artifacts/pilot-assignment.json \
  --output artifacts/pilot-report.json
```

The included synthetic file only verifies the analysis pipeline. Its numbers are fabricated and must never be reported as evidence of efficacy. A report cannot be marked confirmatory unless the supplied assignment file matches every recorded participant/task/condition row.

Run both conditions through the same integrated event and analysis path; only delivery differs:

```bash
cueloop-session run --condition CONTROL --event-command-json '[".build/macos-speech-events","--duration","600"]' --goal "task prompt" --consent-acknowledged
cueloop-session run --condition CUELOOP --event-command-json '[".build/macos-speech-events","--duration","600"]' --goal "matched task prompt" --tts-command say --consent-acknowledged
```

After the ten-minute unassisted recall and blind scoring, append a verified session record:

```bash
cueloop-pilot record data/sessions/SESSION_ID \
  --participant p1 --task task-a \
  --claims 1 --premises 1 --dependencies 1 \
  --counterexamples 1 --questions 1 \
  --unsupported-claim-calibration 1 \
  --recall-delay-minutes 10 --scorer-blinded
```

The record command derives condition, duration, session ID, participant-stop status, and intervention count from the integrity-checked manifest rather than accepting those fields from the scorer. The report marks the primary endpoint exploratory when the recall delay is outside 9–11 minutes or the scorer was not blinded.
