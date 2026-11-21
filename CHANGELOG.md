# Changelog

## Unreleased

### H1 supported — first clean human paired result (n=24)
- Recorded **24 clean paired human CUE/SILENCE sessions** with the fixed
  runtime (after the PTY event-capture fix), using `scripts/record_self_pair.sh`
  in batches via the new `scripts/run_accuracy_batch.sh`. The protocol's
  24-pairs/cell cell is met.
- `scripts/evaluate_h1h2.py` selects only fixed-runtime pairs (CUE outcomes>0
  and SILENCE outcomes==0) and runs the seeded paired bootstrap. Result:
  **uptake delta +0.254, 95% CI [+0.176, +0.335]** (excludes zero), novel
  uptake +0.265 [+0.180, +0.354], carryover +0.250 [+0.170, +0.338].
  CUE arm: 165 deliveries / 165 outcomes; SILENCE arm: 0 outcomes (clean
  control). H1 (cue raises operation-word uptake in the next clause) is
  supported on human Japanese reasoning speech. H2 remains unmeasured — it
  needs the blinded delayed-reconstruction rubric.
- This supersedes the historical contaminated-control estimate (+0.035).

### Fixed — live microphone capture was silently broken
- **Event commands now read through a PTY.** `cli._event_lines` launched the
  speech adapter with `stdout=PIPE`. Swift's `print` block-buffers on a pipe
  (its internal buffer is independent of `fflush(stdout)`), so
  `transcript_partial` events never reached the runtime on live sessions:
  predictions stayed ~0 and no cues were inserted. Reading from a PTY
  (`pty.openpty`) restores line buffering; a pipe fallback remains for
  platforms without PTYs. Evidence: a clean paired session now records
  predictions=59, delivered=4, uptake=1/4 in the CUE arm.
- The Swift source `emit()` was also changed to write via
  `FileHandle.standardOutput.write` (effective once the Xcode license is
  accepted and the adapter is rebuilt); the PTY fix works with the existing
  prebuilt binary without needing that rebuild.
- **`--min-outcomes` no longer gates the SILENCE control arm.** Controls
  intentionally score zero outcomes, so gating them removed the control and
  left the CUE arm unpaired (analysis reported "not comparable"). Now only
  the CUE arm is quality-gated; the control is always retained.
- `record_self_pair.sh` now uses a 3 s warm-up after each spoken announcement
  (SFSpeechRecognizer's first result can lag ~20 s) and prompts for operation
  words (結論は／なぜなら／逆に) so cues can actually be triggered.
- First clean paired human session with the fixed stack:
  `logs/self-accuracy-pty-paired-20261121-234611`, CUE uptake=1/4, SILENCE
  outcomes=0, paired uptake delta +0.250 (n=1, CI insufficient).

### Added — analyzer quality gate
- `scripts/analyze_experiment.py --min-outcomes N` drops sessions with fewer
  than N finalized outcomes before pairing. Empty/silent-mic arms can no
  longer masquerade as valid "zero effect" measurements. A regression test
  (`test_min_outcomes_drops_empty_recording_failure`) proves empty pairs are
  excluded. On the historical contaminated dataset, `--min-outcomes 1`
  reduces 77 pairs to 37 (all with at least one CUE outcome).

### Fixed — observation bug in the SILENCE control arm
- **`SyncRuntime._deliver()` scored an Outcome for `SILENT` deliveries.**
  `--silent` logged a SILENT delivery (timing parity) but then created an
  Outcome and scored `target_operation_seen` on text the speaker never heard
  a cue for. Over the 77 stored pairs this produced 12 spurious "uptake"
  events in SILENCE arms (12/176) vs 21/165 in CUE arms, biasing every
  paired CUE-SILENCE contrast toward zero. Now only audible (`DELIVERED`)
  deliveries create Outcomes; SILENT deliveries are still logged so counts
  and timing stay paired, but the control arm no longer invents uptake.
- Regression tests updated: `SilentConditionTests` now asserts no outcomes
  and zero novel uptake; `test_silence_never_speaks` and the effect-zero
  calibration cell were aligned with the corrected control semantics.
- Re-scoring with the fixed observation: a simulator arm with a programmed
  +0.40 effect now reads +0.368 uptake delta (previously diluted by the
  control arm's fake uptake). All 141 tests passed at that point; the suite
  is now 142 tests after the quality-gate regression test was added.

## [0.4.0] - 2026-11-21

### Fixed
- **CI `unit` job format check** — `tests/test_pair_audit.py` was introduced
  unformatted in 2eeed8b, so `ruff format --check` failed on the latest
  main-branch run (44475798168). Reformatted with `ruff format`; local
  `ruff check`, `ruff format --check`, 141 pytest tests and unittest discover
  all pass (2026-11-21). Re-run 44825622170 is green on all three jobs.
  CI actions were also bumped to `checkout@v5` / `setup-python@v6` to clear
  the Node 20 deprecation annotations.

### Added
- **`scripts/pair_audit.py`** — data-integrity audit across live paired
  sessions. Scans every `logs/self-design-choice-paired-*` (or any roots
  given), classifies each pair into EMPTY / SILENT / LIGHT / OK /
  NON-LIVE, and reports per-pair clause-count, delivery count, and a
  summary line. Catches the common silent-mic case where a session looks
  like it "succeeded" but produced zero Japanese clause bytes. Runs in
  ~0.1 s; `--json` output for automation.
- **CI macos-adapter cancellation issue resolved.** Earlier main-branch runs
  were cancelled mid-compile (~15 min infra limit, not a compile limit); the
  Swift build always completed locally. `timeout-minutes: 25` was added at
  commit 16e62be, and subsequent runs (44350984159, 44825622170,
  44825852155) were green on both jobs. Actions were bumped to
  `checkout@v5` / `setup-python@v6` to clear Node 20 deprecation warnings.
  Workarounds if it recurs: (a) skip macos-adapter in CI, (b) commit the
  prebuilt adapter binary (`macos-speech-events-ci`), or (c) pin macos-latest.
  See issue #6.
- **2026-11-18 restricted-ops arm (A — n=7)**: pairs 234729/235451/000203/001419/002521/004038/005156.
  Mostly sparse/tiny sessions; some language fragments only ('パイロット'/'PNとか使うの').
  Aggregate A: delivered=6 (ground×2, invert×3, comparison×1) uptake=0; paired CI [+0.000,+0.000].
  Cumulative restricted arms (B + A): n=10 sessions live, uptake=0 — no effect detected in any restricted configuration.
- **`docs/gpt-live-protocol.md`** — precise protocol for pairing GPT Live
  (the ChatGPT app's Japanese voice reasoning) with CUELOOP on this Mac:
  topology (speakers-to-mic or BlackHole 2ch routing with owner consent),
  recording procedure (paired CUE/SILENCE with shared pair_key), analysis,
  caveats (agent-side uptake ≠ human-transfer).
- **`scripts/run_agent_live_pair.sh`**: fully autonomous Japanese reasoning
  monologue generation via local LLM (qwen2.5:7b-instruct) → Kyoko TTS →
  SFSpeechRecognizer ingest. Keeps end-to-end pipeline exercised while the
  owner is not at the mic; still produces agent-side data only — local LLM
  cannot hear cues, uptake is structurally zero.
- **First agent-live pair completed 2026-11-16** (logs/agent-live-なぜdesign-
  choiceを選ぶのか-paired-20261116-005151): CUE delivered=4
  (invert/mechanism/conclusion/concretize), uptake=0 on each; SILENCE arm
  delivered=0 by design; paired analyzer ran end-to-end. Not H1/H2.
- **`scripts/record_gpt_live_pair.sh`** — one-command paired recorder for
  the GPT Live setup (300 s arms, Kyoko announce). Smoke-tested today;
  full pipeline works end-to-end, uptake=0 under ambient conditions.
- **2026-11-15 absolute live-pair count 28 (204315 — fifth consecutive
  empty)**: utterances CUE=19 SILENCE=1, zero Japanese clause text.
  Aggregated 28 pairs: uptake +0.045 CI [-0.027, +0.132] (includes zero).
  5 consecutive silent pairs confirmed owner absent; live H1/H2 could not
  advance. See issue #6.
- **2026-11-17 op-spectrum**: `scripts/op_fire_spectrum.py` added; artifacts/
  live-op-spectrum.md generated (gitignored). speaker-natural ops
  (boundary/conclusion/concretize/dependency) fire on both arms; cue-
  divergent (mechanism/ground/invert/comparison) appear only under CUE.
  This is an op-selective hint, not evidence of causal uptake.
- **2026-11-15/17/18 live pairs #36–#69**: #57-#67 near-; #182933 (#68) sparse;
  #010756 (#69) EMPTY. Aggregated 69 live pairs: uptake +0.035 CI [-0.004, +0.080] — zero straddles. See issue #6.
- **2026-11-15 17:09 design-choice attempt 4 (genuinely live)**: pair 3
  of the corpus's live history. CUE arm: utterances=1, predictions=23,
  4 deliveries (concretize/mech/ground/conclusion) — mechanism outcome
  target_operation_seen=True (continuation "同一方向に改善する仕組み"),
  2/6 delivered-ops across live pairs. SILENCE twin: 12 interventions
  HELD by design; carryover seen 2/5 with zero cues. Analyzer n_pairs=1
  per pair, growth/carryover favour SILENCE — CI insufficient; n_pairs=2
  total; not H1/H2 evidence.
- **2026-11-15T17:00 design-choice attempt 3 (genuinely live)**: after
  the playback reclassification, a truly live run captured the owner
  speaking Japanese reasoning through the mic — utterances=1,
  predictions=11, interventions=3 (2 DELIVERED: '仕組みは？' @149,005
  ms +102 ms with outcome target_operation_seen=True → uptake on
  mechanism; '根拠は？' @151,703 +171 ms uptake=False; 1 CANCELLED
  '具体は？'). SILENCE twin kept silence by design. Analyzer n_pairs=1:
  uptake delta +0.500 CI insufficient — not H1/H2 evidence; pending
  24-32 pairs per cell.
- **`--early-lead-ms` plumbed through the Lux simulator** (default 0,
  off): plan cues up to N ms before the pause threshold — executes only
  on TICK-bearing input. On Lux replay (segment feeds without TICKs)
  0 vs 300 produces 30/30 identical sessions; knob exists for future
  TICK-driven studies. Production default unchanged.
- **Lux pause_ms sensitivity sweep (200/350/500/750)**: predictions=36,
  delivered=25 identical at all settings — replay boundaries bind on
  clause-ender punctuation, not on millisecond gaps; pause_ms is a
  live-path-only knob.
- **2026-11-15 playback reclassification (docs/real-speech.md)**: every
  "live" paired capture in this repository is replay of the owner's own
  pre-recorded Japanese monologue into the mic (user-confirmed).
  Consistent with quantitative evidence: byte-identical
  boundary/timestamps across consecutive arms (MD5 `1945b967…` shared
  by Nov 14 20:01 and Nov 15 20:34 runs at ms 54375/55497/56072) — impossible for
  spontaneous speech. All such sessions are canned-class:
  delivery/boundary/prediction all logged, but the speaker cannot react
  to cues; H1/H2 n_pairs = 0. Insertion benchmark and synthetic
  calibration axis survive the reclassification unchanged.
- **2026-11-15 "live pair" (180 s arms, audible Kyoko announce) — historical,
  reclassified**: continuous Japanese monologue replay was captured
  end-to-end. Not live human interaction. Kept as an end-to-end pipeline
  trace (CUE 2 deliveries: mechanism @9.5 s +181 ms, conclusion @176.7 s
  +101 ms; SILENCE arm 12 interventions HELD by design). Do not cite as
  uptake evidence.
- **Recorder announces audible start + defaults to 180 s window**:
  scripts/record_self_pair.sh voices "これから話してください" (CUE) /
  "もう一度、同じ話題で話してください" (SILENCE) before each arm and the
  default arm length is now 180 s, closing the structural race where the
  window expired before the speaker reacted.
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
  canned audio, no effect claimed. Highlights: the 8th (`bandad/asr-testset-kw-ja-v1`) shows the highest op-tag density
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
