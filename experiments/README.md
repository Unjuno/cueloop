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
