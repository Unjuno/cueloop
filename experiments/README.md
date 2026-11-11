# Experiment fixtures

- `sync-smoke.jsonl`: deterministic replay fixture for the sync loop. Its
  clause boundary is at 900 ms; the default 100 ms insertion delay is why the
  smoke cue lands at 1000 ms. That is a fixture property, not a latency target.
- `sync-carryover-smoke.jsonl`: same boundary structure as `sync-smoke.jsonl`
  but keeps the utterance going past the outcome window. Use it with
  `--outcome-window-ms 1000` so the uncued speech after the window is scored as
  carryover. The cued operation is not in the continuation window, so uptake
  stays 0 while carryover is non-zero: that separates the mechanism check from
  the persistence measure.
- `policy-smoke.jsonl`: schema and smoke fixture for the linear policy trainer.
  It is synthetic and exists only to exercise intervention/delivery/outcome/
  carryover joins. Do not use its learned weights as experimental evidence.
- `analysis-smoke/`: nine synthetic report JSON files that exercise the paired
  analyzer end to end. Six paired `rule` sessions show a non-zero CUE - SILENCE
  difference, one `linear` pair shows `"insufficient"` (a single pair cannot
  support an interval), and one `llm` cue has no partner and is reported as
  `"not_comparable"`. These are hand-written fixtures, not recordings. They
  exist to prove the analyzer distinguishes those three cases.

Everything under `experiments/` is synthetic. None of it is evidence about
human speech, and no result computed from these files may be presented as one.

## Regenerating the comparison report

```bash
scripts/run_experiment.py --mode replay --decision rule \
  --events experiments/sync-carryover-smoke.jsonl --outcome-window-ms 1000 \
  --output-dir logs/experiment-carryover --session carry-rule
scripts/run_experiment.py --mode replay --decision linear \
  --policy-model artifacts/sync-policy.json \
  --events experiments/sync-carryover-smoke.jsonl --outcome-window-ms 1000 \
  --output-dir logs/experiment-carryover --session carry-linear
scripts/run_experiment.py --mode replay --decision local \
  --events experiments/sync-carryover-smoke.jsonl --outcome-window-ms 1000 \
  --output-dir logs/experiment-carryover --session carry-llm
scripts/compare_policies.py logs/experiment-carryover/reports \
  --json artifacts/policy-comparison-carryover.json \
  --markdown artifacts/policy-comparison-carryover.md
```

All of this is a pipeline check. The numbers come from a synthetic fixture and
are not evidence about human speech.

## Running the exploration arm with a SILENCE control

`--assignment random` randomizes wording only. The exploration arm draws the
operation, so the delivered operation can differ from what the policy wanted:

```bash
scripts/run_experiment.py --mode replay --assignment random-operation \
  --condition CUE --decision rule --seed 1 \
  --events experiments/sync-carryover-smoke.jsonl --outcome-window-ms 1000 \
  --output-dir logs/experiment-explore-01 --session rule-cue-1
scripts/run_experiment.py --mode replay --assignment random-operation \
  --condition SILENCE --decision rule --seed 1 \
  --events experiments/sync-carryover-smoke.jsonl --outcome-window-ms 1000 \
  --output-dir logs/experiment-explore-01 --session rule-silence-1
scripts/analyze_experiment.py logs/experiment-explore-01/reports \
  --json artifacts/experiment-analysis.json \
  --markdown artifacts/experiment-analysis.md
```

Both runs share `pair_key=rule-seed1-sync-carryover-smoke`. Replay is a fixed
transcript, so cueing cannot change the next words and every paired delta is
zero. Zero deltas here mean the pairing and delivery plumbing work; they say
nothing about effect size. Only live sessions can estimate an effect.

To check the analyzer's detection power without a microphone, generate paired
reports from `scripts/synthesize_experiment.py`, which programs a known cue
effect into a generative speaker and writes `synthetic: true` reports plus a
manifest of its parameters (see `docs/experiment-protocol.md`). Synthetic
numbers calibrate the pipeline; they are not evidence about human speech.

To check the analyzer's non-zero and insufficient paths without a microphone,
run it on the synthetic reports:

```bash
scripts/analyze_experiment.py experiments/analysis-smoke \
  --json artifacts/analysis-smoke.json \
  --markdown artifacts/analysis-smoke.md
```


## Live acoustic smoke logs

`logs/live-explore-01/` holds eight live sessions recorded with a `say` voice
playing a Japanese passage while the adapter transcribed it (seeds 0-1,
rule/linear, CUE/SILENCE). These validate the acoustic pipeline and the
paired analysis plumbing. The talker is inanimate, so all outcome deltas are
zero and arms are not perfectly paired; do not cite these numbers as evidence
about anything.
