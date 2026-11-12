# Synthetic Benchmark (calibration, not human evidence)

CUELOOP cannot replay what a human would say next, so static fixtures always
produce exactly `CUE - SILENCE = 0`. This benchmark exercises the whole
analysis and training pipeline on data with a **programmed** effect and known
ground truth. It checks that:

1. the paired analyzer reports exactly zero on null data,
2. the paired analyzer detects a programmed effect with a CI excluding zero,
3. the linear policy trainer recovers a reward structure planted in the data.

Everything below is synthetic. Nothing here says anything about whether cues
change human reasoning. Only paired live sessions can do that.

Regenerate everything with one command (deterministic, seeded):

```bash
scripts/run_synthetic_benchmark.sh            # writes artifacts/benchmark/
```

## End-to-end runtime simulation (real runtime, synthetic speaker)

`scripts/simulate_sessions.py` generates Japanese clause sequences, feeds them
through the real `SyncRuntime` (cycle predictor, rule decision, canonical
assignment, silent TTS stub), and emits analyzer-ready reports and JSONL logs.
Cue and SILENCE arms share a skeleton and the same uniforms (common random
numbers); only the audible cue can cause uptake/carryover — the programmed
effect is exactly the difference the analyzer should find.

24 sessions × 8 clauses per cell, effects 0 / 0.2 / 0.4, two seeds each
(`artifacts/benchmark/simulated-*.json`):

| programmed effect | seed 0 | seed 1 | seed 2 | seed 3 |
| --- | --- | --- | --- | --- |
| uptake 0.0 | +0.000 [±0.000] | +0.000 [±0.000] | +0.000 [±0.000] | +0.000 [±0.000] |
| uptake 0.2 | +0.097 [+0.043, +0.152] | +0.105 [+0.048, +0.174] | +0.095 [+0.031, +0.156] | +0.050 [+0.001, +0.099] |
| uptake 0.4 | +0.127 [+0.060, +0.199] | +0.163 [+0.101, +0.226] | +0.122 [+0.050, +0.190] | +0.130 [+0.043, +0.214] |
| carryover 0.2 | +0.044 | +0.028 | +0.021 | +0.015 |
| carryover 0.4 | +0.073 | +0.109 | +0.071 | +0.081 |

The null arm here is exact (+0.000 everywhere); the analyzer recovers the
programmed effect with CIs excluding zero at 0.2 and 0.4.

## Analyzer grid: programmed uptake effect × seed

24 clause-pairs per cell, 8 clauses per session, three backends
(rule / linear-policy / local-llm), seeded percentile bootstrap (2000
resamples, 95% CI). Metric: paired uptake delta (CUE − SILENCE).

| programmed effect | seeds | backends | uptake delta (point range) | CI excludes 0 | null violations |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0, 1, 2 | rule, linear, llm | +0.000 (all 9 cells) | 0/9 (correct) | 0 |
| 0.10 | 0, 1, 2 | rule, linear, llm | +0.073 … +0.130 | 9/9 | — |
| 0.20 | 0, 1, 2 | rule, linear, llm | +0.156 … +0.240 | 9/9 | — |
| 0.30 | 0, 1, 2 | rule, linear, llm | +0.229 … +0.328 | 9/9 | — |

The null arm is exactly zero by construction (common random numbers shared by
CUE/SILENCE). The effect arms are detected in all 27 cells at this sample
size. This verifies the analyzer, the pairing, and the CI machinery — nothing
else.

## Log generator and policy training at scale

`scripts/synthesize_logs.py` emits session logs in the real log schema
(intervention / delivery / outcome / carryover records) with a reward structure
planted on three features plus a per-operation bias:

```text
carryover_logit = -1.1
    + 1.3 * (prediction_confidence - 0.75) / 0.2
    + 0.5 * (prediction_length - 20) / 20
    + OP_BIAS[operation]     # ground +1.5 … conclusion -0.7
```

Training corpus: 600 sessions × 8 clauses (18,004 records, ~4,800
interventions, 4,202 usable samples after delivery/divergence/missing-label
filters). Held-out corpus: 150 sessions × 8 clauses with a different seed,
including a random-operation arm (9,028 records, 2,114 usable samples).

Results (`artifacts/benchmark/sync-policy-eval.json`,
`sync-policy-holdout.json`):

| set | samples | positives | ROC-AUC | accuracy | Brier |
| --- | --- | --- | --- | --- | --- |
| training split | 3,151 | 1,245 | 0.696 | 0.664 | 0.229 |
| training holdout split | 1,051 | 415 | 0.695 | 0.676 | 0.228 |
| independent holdout (seed 999) | 2,114 | 837 | 0.664 | 0.650 | 0.236 |

The ceiling (AUC of the true logit against the sampled reward on the same
data) is ≈ 0.73, so the trainer recovers most of the available signal and
does not collapse on the random-operation arm. mean_score_positive >
mean_score_negative on every split.

## Limits

- These numbers verify the *pipeline*, not the *hypothesis*. The synthetic
  speaker is a coin flip conditioned on a logit we chose; humans are not.
- The `llm` backend in the analyzer grid is a generator, not the real local
  model on the wire.
- Policy metrics on logged data are not off-policy estimates of the deployed
  policy. Causal claims still require the exploration arm and SILENCE.
- 0.3.0 ships this calibration as part of the release; human paired sessions
  are still outstanding.
