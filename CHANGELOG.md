# Changelog

## Unreleased

### Added
- **Insertion benchmark extended to Common Voice 8.0** (4,483-file ja
  sample + 120-file replay subset). On the full sample: context-fit
  ranking 8.1% vs RANDOM-order PRIORITY 9.5% — the only corpus where
  PRIORITY inverts the ranking, because read prose has near-zero
  reasoning structure. Anchor decomposition (op present in prefix) locates
  the inversion *inside the anchor cell* (CV8 ranked 10.3% vs PRIORITY
  12.1%, n=58) while the no-anchor cell is 0.0% everywhere; reasoning
  corpora keep a 17-29 point margin inside the anchor cell. The
  JSON export bug collapsing same-basename corpora under one key is
  fixed (`<dir>/<file>` labels); `Next steps` in docs/real-speech.md
  re-prioritized by insertion evidence (selection axis first).
- **Analyzer reads simulator output directly**: `scripts/analyze_experiment.py`
  now discovers report JSONs one directory level deep, so
  `analyze_experiment.py artifacts/<arm>` works on `<arm>/cue/*.json` +
  `<arm>/silence/*.json` without manual flattening. Regression test covers
  the two-level arm layout (140 tests total).
- **Offline insertion benchmark** (`scripts/benchmark_insertion.py`): measures
  insertion appropriateness of the production path on real Japanese
  transcripts without any live microphone. Two axes per clause boundary:
  (a) prediction axis — ngram(3) span op tags vs the human's next-op;
  (b) selection axis — the production missing-op ranking
  (`PolicyObservation.ranked_missing_operations`, the rule decision) vs the
  op actually present in the human's next clause. On lux / reazon-200 /
  podcast-replay / jsut-replay: context-fit ranking picks the human's next
  op 0%–24.1% of the time; PRIORITY (context-blind) picks it 0.0–1.9%; a
  deterministic adjacency-chain selector picks it 0–7.7%; an oracle inside
  the missing set covers it 66.7–100%. H1/H2 remains
  untested — issue #6 stays open; insertion ≠ uptake.
  (Added 2026-11-13: bandad-kw arm, 2,695 boundaries; context-fit 18.9%.)
- **Insertion benchmark — anchor decomposition & gate rejection**:
  selection axis split into anchor/no-anchor cells; CV8's inversion was
  localized inside the anchor cell (ranked 10.3% vs PRIORITY 12.1%,
  n=58). Two signal-gated selectors (`sel_*_clausegated`: ranking only
  with clause ops; `sel_*_weightgated`: ranking only with anchor weight
  ≥ 2.0) measured on all 7 corpora are **both strictly worse than
  production ranking everywhere** (e.g. kw-ja 3.8% / 4.7% vs 18.9%;
  podcast 3.4% / 6.9% vs 24.1%; reazon 0.0% / 7.7% vs 23.1%). Production
  constants unchanged; gate arms exist only in the benchmark.
- **JSUT local-LLM arm (100 files, paired)**: `gemma3:4b-it-qat` predictor
  + local decision. CUE delivered 136/136 interventions (dependency 58,
  boundary 41, ground 26, comparison 10, mechanism 1; 1.36/file; 0 cancelled
  /expired); SILENCE arm HELD all 139 by design. Paired n=100: every delta
  +0.000 CI [+0.000, +0.000]. Zero uptake — the canned-audio asymptote on
  the local-LLM path.
- **Podcast local-LLM arm extended 3 → 15 clips (paired)**: CUE delivered
  443/443 (ground 119, boundary 112, dependency 105, comparison 99,
  mechanism 6, conclusion 2); SILENCE HELD 439/439. Paired n=15: uptake
  −0.002 [−0.006, +0.000] (canned-token coincidence, CI upper-bound zero),
  all other deltas +0.000. No effect measured.
- **bandad kw-ja local-LLM arm (40 clips, paired)**: CUE delivered
  117/117 (dependency 41, boundary 35, ground 22, comparison 15,
  mechanism 3, conclusion 1); SILENCE HELD 115/115. Paired n=40: all
  deltas +0.000 CI [+0.000, +0.000]. With JSUT (n=100) and podcast
  (n=15), the canned-audio asymptote on the local-LLM path now stands at
  **n=155 paired sessions across three corpora — every CI zero**.
- Real-speech replay across 8 Japanese corpora (see docs/real-speech.md) —
  each replayed on real audio end-to-end with local-LLM and rule backends;
  the 8th (`bandad/asr-testset-kw-ja-v1`) shows the highest op-tag density
  of any corpus measured (21.8% of 510 clips tag ≥ 1 operation; 1638 predictions,
  998 delivered cues). The 7th arm, `PlayMaker13/japanese-podcast`, contains
  conclusion-heavy native podcast monologues (20 paired sessions, 474
  delivered, uniform operation distribution). benchmarks+docs updated with
  the local-LLM arm results and reproduction blocks. Human transfer (H1/H2)
  remains untested — issue #6 is open.

### Fixed
- **`scripts/record_self_pair.sh`**: staging step copied two source files
  into a target ending in `.jsonl`, crashing every paired run at the last
  step ("Not a directory"). Now writes `cue/silence` with distinct basenames
  in a paired dir, tolerates a totally silent arm (missing log file), and
  prints an explicit WARNING when SFSpeechRecognizer produced no clause
  text at all — empty "success" can no longer masquerade as paired data.
  (First cut of the WARNING matched the wrong signature — separate
  transcript events — and false-tripped on a genuinely spoken CUE session;
  it now scans `.jsonl` `"clause"` fields, verified on three live files.)
- **Prediction parser accepts non-standard model outputs**: markdown-fenced
  JSON (```json ... ```) and synonym span keys (`next_segment`,
  `next_sentence`, `continuation`, `next_text`, `next`) are now parsed
  instead of failing as "prediction response did not contain a JSON
  object". Regression tests cover both paths. This lets reasoner models
  answer at all; it does not change what they predict.
- **Runtime delivery clamp**: synchronous backends (fresh prediction created
  in the same event) no longer have their intervention pushed out by
  prediction+decision wall latency. Before, the shift landed the cue into an
  already-diverged window and the synchronizer cancelled it — so every
  local-LLM intervention on the ReazonSpeech sample and in live replays was
  cancelled (0 delivered). With the fix the local-LLM backend delivers at
  the boundary event, like the rule backend: 668/668 on the full 200-file
  ReazonSpeech sample (gemma3:4b-it-qat), 138/138 on the
  operation-tagged 30-file subset, 324/324 on the full Lux 328-file corpus
  (gemma3:4b-it-qat), and 327/327 on the Reazon qwen2.5:1.5b-instruct full
  sample; synthetic calibration unchanged (null cells +0.000, 27/27 effect
  cells CI-excluding-zero). On the live mic loopback (EMOTION100_011 /
  _066), cues now reach `status SILENT` delivery instead of being cancelled;
  EMOTION100_072 vets out at the local decision model (no tagged operations
  in the utterance; 0 interventions, as designed).

### Added
- Per-session repetition fatigue in `RuleDecisionModel` (`fatigue_penalty`,
  default 2.0): each chosen operation accrues fatigue so the rule backend
  stops delivering a near-monotone stream of one cue on discourse Japanese.
  Cue diversity on the ReazonSpeech arm rose from 94% `concretize` to a
  7-operation mix (concretize 46%); delivery counts, prediction counts and all paired analyzer deltas
  (+0.000) are unchanged; synthetic null/effect cells are unchanged
  (27/27 effect cells CI-excludes-zero).
- `scripts/simulate_lux_replay.py` accepts `--corpus`/`--whisper-cache`/
  `--source-name` and replays any compatible corpus (Lux or ReazonSpeech).
  Fixes a whisper-cache subset bug (warm cache used to return the whole
  file regardless of the requested corpus) and a relative-path corpus
  resolution bug that created duplicate caches; both covered by tests.
- `scripts/bench_predictors.py`: one-command benchmark of local-LLM next-span
  agreement on any compatible `(metadata.csv + whisper-cache)` corpus;
  reports exact-prefix rate, average longest-common-prefix, operation-tag
  hit rate, and parse failures per model. On the Reazon sample every
  model tested scores 0 exact-prefix; operation-token hit rate is ~0 for
  every tested model. This measures what the local LLM can do on real
  discourse, not whether cues change reasoning.
- Real-speech replay completed end-to-end on the Lux and Reazon corpora for
  the local-LLM decision path, pre-fix and post-fix; post-fix measurements
  now dominate the docs (pre-fix numbers retained for transparency, clearly
  labeled "pre-runtime-fix"): local-LLM delivers at every decision boundary
  under the fixed runtime — 668/668 interventions on the full 200-file
  ReazonSpeech sample (gemma3:4b-it-qat), 138/138 on the operation-tagged
  30-file subset, 327/327 on the Reazon qwen2.5:1.5b-instruct full sample,
  and 324/324 on the full Lux 328-file corpus (gemma3:4b-it-qat).

## 0.3.0 (2026-11-12)

### Added
- Synthetic session-log generator (`scripts/synthesize_logs.py`):
  real-schema logs with a planted, recoverable reward structure for training
  the linear policy at scale (600-session corpus → ROC-AUC ≈ 0.70, close to
  the ≈ 0.73 Bayes ceiling of the generative model).
- End-to-end runtime simulator (`scripts/simulate_sessions.py`): generates
  Japanese clause sequences and feeds them through the real `SyncRuntime`
  (cycle predictor, rule decision, assignment, canned speaker), emitting
  analyzer-ready reports and JSONL logs. Its null arm is exactly +0.000;
  programmed effects of 0.2/0.4 are recovered with CIs excluding zero.
- Real-speech replay (`scripts/simulate_lux_replay.py`): plays the local Lux
  corpus back through the same `SyncRuntime`, emitting 656 analyzer-ready
  reports (328 CUE + 328 SILENCE) with zero uptake as expected for read
  speech. Confirms the live measurement path runs on real Japanese audio.
- One-command calibration benchmark (`scripts/run_synthetic_benchmark.sh`,
  documented in `docs/benchmark.md`): analyzer grid over effects × seeds ×
  backends (null cells exactly zero, 27/27 effect cells CI-excludes-zero), the
  end-to-end runtime simulation (null exactly zero; 0.2/0.4 effects recovered),
  plus policy training/evaluation at scale. `scripts/analyze_power.py`
  computes the paired design's power curve (16 seeds/cell; type-I stays at 0%;
  effect 0.1 detection hits 88% at 24 pairs and 100% at 48).
- `scripts/tune_ngram.py`: n-gram evaluation harness for any `metadata.csv`-shaped
  speech corpus. Used for calibration against the Lux Speech Corpus.
- `tools/file_asr.swift`: one-shot voicing recognizer using the same
  SFSpeechRecognizer the live adapter uses; benchmarks CUELOOP's production
  ASR engine per file when run on a license-accepted macOS machine.
- Operation-level exploration arm (`--assignment random-operation`): the timing
  policy still decides *when* to fire; the delivered operation is drawn
  uniformly from the missing-operation set so `requested_operation` and
  `operation` diverge. Seed-reproducible.
- Stem cues (`--cue-style stem`, the CLI default): instead of a bare
  interrogative word ("根拠は？"), the intervention is a complete,
  context-grounded "initial-condition" sentence that embodies the operation
  ("根拠は、観測データでは弱いからです。"). For `local`/`steer` the stem is
  written inside the decision call; for `rule`/`linear` a bounded synchronous
  LLM call writes it (`--stem-timeout`, default 1.5 s) with a predicted-span
  template and then bare wording as fallbacks. `--max-stem-characters` (default
  48) caps the sentence; stems take seconds to speak, so echo suppression
  windows are longer. Under `--async-decision` (default) the stem writer runs
  speculatively in the background for all backends, so boundaries do not block
  on the stem LLM; an unready precompute falls back to a bare cue on time.
  `--cue-style bare` restores the previous behaviour; experiment runs should
  use bare or length-match NEUTRAL to stem.
- SILENCE control (`--silent`): decisions and timing are logged, no cue is
  spoken, delivery is recorded as `SILENT`. Both conditions are comparable
  under the paired analysis.
- Paired CUE/SILENCE analysis (`scripts/analyze_experiment.py`): sessions are
  clusters, percentile bootstrap with fixed seed, `insufficient` below two
  complete pairs, `not_comparable` when no pair exists.
- Context-fit operation ranking (default): candidates are ordered by a
  hand-authored adjacency prior anchored to the spoken clause. The rule stays
  silent when the transcript carries no operation signal. `--no-context-fit`
  restores the legacy static priority order.
- Steering backend (`--decision steer`, experimental): a local LLM proposes
  the one move that best redirects the reasoning trajectory instead of filling
  the next missing vocabulary item. Logged under the `steering` family.
- `--adapter-args-json` on the experiment runner (e.g. a lower `--speech-db`
  for weak-signal acoustic tests).
- Synthetic generative calibration arm (`scripts/synthesize_experiment.py`):
  paired reports with a programmed cue effect so the analyzer's detection
  power can be checked without human data.
- CI: ruff and pytest run on every push/PR.

### Changed
- Default rule backend now ranks candidates by context fit rather than global
  PRIORITY. Fixture replays now produce `具体は？`/`逆は？`-style cues instead
  of `根拠は？` depending on the anchored move.
- Boundary-to-delivery delay is computed over delivered *and* silent
  decisions.

### Fixed
- Paired CUE/SILENCE exploration sessions digested identical
  `intervention_id`s; log files are namespaced before joins so SILENCE
  deliveries no longer drop CUE training samples.
- Unknown `assignment` strings are reported by `train_policy.py` instead of
  being silently ignored.

### Notes
- All numbers measured on fixtures are synthetic. No human session exists
  yet; claims about human reasoning outcomes are unsupported until paired
  live recordings exist.
