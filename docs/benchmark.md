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

24 sessions × 8 clauses per cell, effects 0 / 0.2 / 0.4, four seeds each
(`artifacts/benchmark/simulated-*.json`):

| programmed effect | seed 0 | seed 1 | seed 2 | seed 3 |
| --- | --- | --- | --- | --- |
| uptake 0.0 | +0.000 [±0.000] | +0.000 [±0.000] | +0.000 [±0.000] | +0.000 [±0.000] |
| uptake 0.2 | +0.081 [+0.033, +0.132] | +0.114 [+0.064, +0.169] | +0.129 [+0.079, +0.175] | +0.074 [+0.019, +0.129] |
| uptake 0.4 | +0.115 [+0.051, +0.177] | +0.176 [+0.114, +0.243] | +0.180 [+0.106, +0.256] | +0.190 [+0.099, +0.281] |
| carryover 0.2 | +0.029 | +0.045 | +0.033 | +0.010 |
| carryover 0.4 | +0.026 | +0.077 | +0.092 | +0.055 |

## Extended effect grid (detection curve at smaller effects)

Replicating the in-the-loop calibration at smaller effects proves the
detection machinery at the low end. Two seeds shown, 24 paired sessions each:

| effect | seed 0 CI | seed 1 CI |
| --- | --- | --- |
| 0.10 | +0.038 [+0.004, +0.077] | +0.064 [+0.027, +0.106] |
| 0.15 | +0.075 [+0.033, +0.116] | +0.075 [+0.033, +0.119] |
| 0.20 | +0.081 [+0.033, +0.132] | +0.114 [+0.064, +0.169] |
| 0.25 | +0.095 [+0.038, +0.154] | +0.140 [+0.090, +0.194] |
| 0.30 | +0.095 [+0.040, +0.153] | +0.155 [+0.105, +0.210] |
| 0.40 | +0.115 [+0.051, +0.177] | +0.176 [+0.114, +0.243] |

Detection clears CI-excluding-zero at 0.10+, comparing group-level paired
differences through the same bootstrap CI. The 0.05 boundary is at the noise
floor of the default `simulate_sessions` model and was deliberately left out.


## Sample size for the human study (power analysis)

`scripts/analyze_power.py` re-runs the event-level simulator across 16 random
seeds per (effect × pairs) cell and reports how often the analyzer's 95% CI
excludes zero. Output (`artifacts/power-analysis.json`):

| effect | pairs=8 | pairs=16 | pairs=24 | pairs=48 |
| --- | --- | --- | --- | --- |
| 0.0 | 0/16 | 0/16 | 0/16 | 0/16 |
| 0.1 | 9/16 (56%) | 12/16 (75%) | 14/16 (88%) | 16/16 (100%) |
| 0.2 | 12/16 (75%) | 16/16 (100%) | 16/16 (100%) | 16/16 (100%) |
| 0.3 | 14/16 (88%) | 16/16 (100%) | 16/16 (100%) | 16/16 (100%) |

Type-I rate is exactly zero at every cell size (the null never excludes
zero; detecting what isn't there is the noise floor and stays at 0%). At an
uptake effect of 0.1, 24+ pairs are needed for consistent detection; at
0.2+, 16+ pairs reach 100%. **For the human study, plan 24–32 pairs per
cell** — more if you want to stratify (word vs reading task) or add a
mechanism (e.g., uptake at single-boundary cost).

The null arm here is exact (+0.000 everywhere); the analyzer recovers the
programmed effect with CIs excluding zero at 0.2 and 0.4.

## Stem-cue arm (end-to-end with stems)

The same simulator with `--cue-style stem` speaks a complete, context-grounded
"initial-condition" sentence rendered from the deterministic span template
(stems generated from unreachable test server fail over to template, so the
arm is reproducible offline). Effect recovery matches the bare arm exactly,
and the null arm is exactly zero:

| stem effect | seed 0 | seed 1 |
| --- | --- | --- |
| 0.0 | +0.000 [±0.000] | +0.000 [±0.000] |
| 0.2 | +0.081 [+0.033, +0.132] | +0.114 [+0.064, +0.169] |
| 0.4 | +0.115 [+0.051, +0.177] | +0.176 [+0.114, +0.243] |

This validates that `cue_style` is only a presentation knob — the
intervention assignment (when, what operation) is unchanged.

## Real-speech replay (Lux corpus and ReazonSpeech)

`scripts/simulate_lux_replay.py` replays real Japanese audio through the
actual `SyncRuntime` with per-word `transcript_partial` timestamps produced
by `whisper-base-mlx`. Each file is one session, paired with itself under
CUE and SILENCE. Two corpora are covered — the Lux read corpus (this
section) and the ReazonSpeech YouTube-discourse sample (~200 files, 30
operation-tagged) in `docs/real-speech.md`, where the rule backend delivered
455 cues and the local-LLM backend delivered zero. The Lux replay:

Speaker output is held; outcome scoring, boundary detection, and analyzer
pairing all run end-to-end.

Full run live at `artifacts/lux-rule-328/` (328 CUE / 328 SILENCE reports):

| arm | sessions | delivered | outcomes | uptake | carryover | growth | delay ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CUE | 328 | 276 | 276 | 0.00 | 0.00 | 0.04 | 0 |
| SILENCE | 328 | 0 | 276 | 0.00 | 0.00 | 0.04 | 0 |
| delta | — | — | — | +0.000 | +0.000 | +0.000 | +0.000 |

405 boundary\_predictions fired across the 328 files (1.23 per file).
Operations chosen by rule (after fatigue tuning): concretize 194, dependency 50,
mechanism 13, conclusion 8, invert 6, ground 4, comparison 1 — cues used:
`具体は？` 194×, `何に？` 50×, `仕組みは？` 13×, `結論は？` 8×, `逆は？` 6×,
`根拠は？` 4×, `何と比べる？` 1×. The rule fires on the embedded n-gram's
predictions at the small pauses in the TTS read-through — not at any
announced operation the speaker is producing. The rule backend now applies
per-session repetition fatigue (`fatigue_penalty=2.0`): without it the
delivery mix was concretize 264 (96%); fatigue rotated the mix while
preserving identical counts (276 delivered, 405 predictions) and identical
+0.000 deltas.

This is the expected result: the Lux recordings are single utterances of
foreign-language TTS proofreading, not live reasoning, so there is no
mechanism for induced uptake and no cue to pick up. The value of this arm is
that the full live pipeline — ASR event gating, prediction, decision, leak
suppression, dispatch, outcome scoring, analyzer pairing — is exercised on
real Japanese audio without any microphone or speaker hardware.

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
