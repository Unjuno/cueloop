# Real Speech Corpus Evaluation

CUELOOP's live path depends on (a) the macOS microphone adapter for ASR and
(b) the n-gram predictor for low-latency prediction with the local LLM as
semantic prior. This document records what we found when we put publicly
available Japanese speech corpora through that pipeline.

**Corpora.**

- *Lux Japanese Speech Corpus* (Lami Inc., CC-BY-4.0) — the `ita_recitation`
  clean read subset: 324 files, ~230 MB, 96 kHz/16-bit WAV. Read prose;
  no CUELOOP operation-vocabulary overlap. Used for the first real-speech arm.
- *ReazonSpeech* (Reazon Holdings, CDLA-Sharing-1.0; mirror
  `osanpo/reazonspeech`) — YouTube-derived Japanese speech (news, sports,
  talk, cooking — real speaker discourse). `metadata-sample-200.csv` draws the
  200 longest/most-content-dense utterances with at least 6 s and 15
  characters from chunk `000` (matches `efwkjn/reazonspeech_metadata` for
  transcriptions; YouTube-derived; audio is not redistributed in this repo —
  `data/` is git-ignored). Average content density ~4.5× the Lux set; 30/200
  lines tag at least one CUELOOP operation. Used for the vocabulary-domain
  arm.

These corpora are real recorded Japanese speech but they are not CUELOOP's
live target either: neither contains human-against-cue interaction (canned
audio cannot respond). They remain honest calibration targets, just ordered by
proximity to the operational discourse register.

## Predictor capacity across all Japanese corpora (refined)

For every Japanese speech corpus we ran the n-gram predictor sweep. Zero
exact-prefix everywhere — not one prediction matched even one character:

```text
corpus                              exact_prefix (best config)
lux (order 4/5, default/merged)     0.0 / 0.0 / 0.0
reazon (order 3, all variants)      0.0 / 0.0 / 0.0
common voice 8.0 ja                 0.0 / 0.0 / 0.0
common voice 17.0 ja                0.0 / 0.0 / 0.0
jsut-basic5000                      0.0 / 0.0 / 0.0
```

Local-LLM delivery counts, expanded across eight corpora:

```text
jsut local-LLM (50 files): 78 predictions, 75 interventions, 75 delivered
  (ops: dependency 27, boundary 27, ground 14, comparison 5, mechanism 2)
  files with >=1 delivery: 41/50
  analyzer n_pairs=50, all deltas +0.000 CI [+0.000, +0.000]

jsut local-LLM (30 files, extended): 49 predictions, 48 interventions, 48 delivered
  (ops: dependency 21, boundary 13, comparison 6, ground 6, invert 1, mechanism 1)
  files with >=1 delivery: 25/30; analyzer n_pairs=30, all +0.000 CI [+0.000, +0.000]

bandad/asr-testset-kw-ja-v1 local-LLM (10 files): 29 predictions, 28 interventions, 28 delivered
  (ops: dependency 10, ground 7, comparison 7, boundary 4)
  files with >=1 delivery: 5/10; analyzer n_pairs=10, all +0.000 CI [+0.000, +0.000]

lux local-LLM (post-fix): 324 / 324 delivered, all metrics +0.000 CI
reazon local-LLM (full 200): 668 / 668 delivered, all metrics +0.000 CI
dialog (A0022_S0003_0_G0029): 387 / 387 delivered
kw-ja 510-clip arm: 998 delivered (rule backend)
```

The local-LLM backend is consistent: it produces downstream-acceptable
predictions, but on all eight real Japanese corpora the paired analyzer
finds no measurable uptake. This is structural: replay is canned speech.
The only path to uptake evidence is a human speaker (issue #6).

## bandad/asr-testset-kw-ja-v1 (eighth corpus arm; closest to Japanese public oral reasoning)

`bandad/asr-testset-kw-ja-v1` — 540 clips of real Japanese YouTube speech,
human-transcribed and human-confirmed (lawyers, sales presentations,
cooking, antique dealers, students, wedding speeches, electrical
contractors). After filtering out excluded / unconfirmed segments, 510
clips remain. This is the richest Japanese oral corpus in our sweep for
operation-tag coverage (21.8% of clips tag at least one op).

```text
clips with JP content        510
clips with tagged ops        111 (21.8%)
op counts: conclusion 48, concretize 31, comparison 13, boundary 10,
           mechanism 10, invert 8, dependency 5, ground 4
files             510 / CUE+SILENCE paired
predictions       1638
interventions    998
delivered        998
operations delivered  concretize 425, dependency 276, conclusion 159,
                       mechanism 47, ground 45, invert 42, comparison 4
paired delta     +0.000 CI [+0.000, +0.000] (every metric)
n_pairs          510
```

CUE vs SILENCE outcome is +0.000 — same as everywhere else. The point is
that this corpus is rich enough to fire cues emphatically (1638
predictions across 510 clips, each clip averaging ~3 predictions and ~2
interventions).

## What was measured

### 1. ASR quality on the corpus (whisper-base-mlx, not the production path)

`mlx-community/whisper-base-mlx` (multilingual base checkpoint, untuned)
transcribed all 324 files. Character accuracy vs the ground truth:

```text
file count:        324
total characters:  ~
character accuracy: 73.8%
```

Errors concentrate in two clusters:

- **Unknown names / proper-reading mis-transcriptions** (e.g. `ツァツォ`
  becomes `サツ`, `チューティー` variants), which is expected for a
  96 kHz professional recording read at pace with rich vocabulary.
- **Approximately 22 files under 50%**, which are either whisper-blended
  noise or content of overwhelmingly foreign proper-noun density.

This is not a criticism of whisper-base. A larger model would close most of
the gap. The point is the **CUELOOP pipeline is only as good as its ASR
front-end**, and the macOS SFSpeechRecognizer on-device model (the one we
actually use) is a different model with different failure modes. Both share
the property that they will mis-transcribe rare Japanese names and loanwords;
they will not share the errors on common words. We have not yet measured the
on-device recognizer on this corpus (it requires a live microphone or a
Swift-file recognizer build, both of which are platform-limited).

### 2. N-gram predictor on the corpus

The n-gram predictor (order 3/4/5, span 4/8/16/24/40) achieves **zero**
exact-prefix matches on held-out lines from this corpus, under all
configurations, in both train-on-Lux-only and train-on-embedded+Lux. This
does not mean training on Lux is useless; it means the embedded corpus
does not transfer to human Japanese, and Lux-only training does not
generalize across genres of Japanese prose.

The implication:

- The CUELOOP n-gram predictor is **not useful on real Japanese speech**.
- The local LLM (`--decision local`) is the only model that performs
  prediction at all on this corpus.
- The fallback n-gram is a *very* weak signal and should not be treated
  as meaningful input — `min_confidence=0.55` is a low bar.

**We have not fixed this yet.** Options include (a) expand the embedded
corpus on Japanese-software-reasoning prose, (b) always launch with
`--predictor local` for live sessions, (c) retire the n-gram fallback
entirely.

### 3. Local-LLM predictor on the corpus

The local model is the workhorse (`--predictor local`, the intended live
path). Using the exact `LocalLLMPredictor` code CUELOOP ships with, the
same prefix task on the same corpus:

```text
model: qwen2.5:7b-instruct (Ollama)
corpus: Lux ita_recitation, 12-char prefix
exact_prefix_rate: 0/30
```

The local model *never replicates* the human-spoken continuation. This is
unsurprising: the Lux ita_recitation is translations of European literature
and philosophy — far from own-domain Japanese software-engineering speech,
which is what CUELOOP was built around. Topicality is much better (it
writes `笑顔で手を振っている` for `女の子がキッキッ嬉しそう`), but no model
knows which specific token a writer picked.

The signal we can draw here is narrower than it first appears:

- Out-of-domain Japanese: no predictor helps. This does not indict CUELOOP's
  mechanism; it means we have no proof either way for the target domain until
  live sessions exist.
- The decision-model latency to consider: local LLM is still the only model
  with any signal; n-gram does nothing at all beyond substring recall.

## What was not measured

- **The CUELOOP arbitration pipeline as a whole**: we cannot yet test
  uptake/carryover from real Lux audio because:
    - To trigger an intervention you need a cue *during* the speaker's turn.
      Lux reads are single-shot utterances; there is no live turn to react to.
    - To test uptake/carryover on real humans you need headsets/microphones.
- **The macOS SFSpeechRecognizer on Lux files**: not run locally (Xcode
  license not accepted; the Swift file-asr tool in `tools/file_asr.swift`
  waits on CI or this machine accepting the license).  The microphone path
  has been driven against clean recordings of Lux files played back from a
  speaker, post-fix, with `--no-async-prediction --decision local
  --predictor local --silent` (local LLM predictions, bare cues, interventions
  held for suppression-consistent measurement). Live loopback deliveries
  (SILENT arm = `--silent`; all detections vs. built-in speaker output):

  | file | ASR headline | predictions | interventions | delivered | note |
  | --- | --- | --- | --- | --- | --- |
  | EMOTION100_011 | `イタリア旅行で彼は…` | 4 | 1 | 1 (何と比べる？, comparison) | delivery 4808 ms (+677 ms); pre-fix build cancelled |
  | EMOTION100_054 | (none — clip too quiet for the on-device recognizer at -50 dB) | 0 | 0 | 0 | repertoire grew boundary/invert without cue |
  | EMOTION100_066 | `時間はあるんだから…` | 1 | 1 | 1 (根拠は？, ground) | boundary trigger `だから`, delivery 4498 ms (+184 ms) |
  | EMOTION100_072 | `1つには…` | 8 | 0 | 0 | decision model vetoed every boundary (no tagged operations present) |
  | RECITATION324_001 | `女の子が…` | 1 | 1 | 1 (根拠は？, ground) | delivery 4107 ms (+483 ms) |
  | RECITATION324_004 | `愛知共和国でトゥーサンルーベルトゥールが…` | 1 | 1 | 1 (根拠は？, ground) | delivery 7220 ms (+456 ms); ASR mis-transcribes but boundary/delivery fires |

  No external headphones are connected on this machine (`system_profiler
  SPAudioDataType` reports only MacBook Pro Microphone + MacBook Pro
  Speakers) — these loopback runs are raw speaker→mic with algorithmic
  suppression only, and `--silent` holds cues so no acoustic effect can
  leak in via the speaker.

  So the local-LLM decision path now delivers cues on real mic audio end
  to end, and the rule backend is complementary — same battery, alternating
  coverage:

  | file | backend | predictions | interventions | delivered |
  | --- | --- | --- | --- | --- |
  | EMOTION100_011 | local-llm | 4 | 1 | 1 (何と比べる？, comparison) |
  | EMOTION100_066 | local-llm | 1 | 1 | 1 (根拠は？, ground) |
  | EMOTION100_072 | rule | 4 | 1 | 1 (具体は？, concretize) |
  | EMOTION100_072 | local-llm | 8 | 0 | 0 (decision model vetoed) |
  | RECITATION324_001 | rule | 5 | 0 | 0 (no boundary matched) |
  | RECITATION324_001 | local-llm | 1 | 1 | 1 (根拠は？, ground) |
  | RECITATION324_004 | local-llm | 1 | 1 | 1 (根拠は？, ground) |

  All arms held silent with `--silent`, so these are mechanism-completion
  proofs, not uptake evidence. CUE (unscripted human response) is impossible
  on canned audio.
- **A Japanese-software-reasoning speech corpus**: none publicly available
  short of paying for it. The synthetic CUELOOP corpus in
  `experiments/`/`docs/benchmark.md` is the closest substitute.

## Predictor capacity on the Reazon sample (and what it buys)

We benchmarked the production local-LLM predictor (`LocalLLMPredictor`) on
the 24-sentence slice used by `scripts/bench_predictors.py`:

```text
model                   exact-prefix  avg longest-common-prefix  op-match  failures
gemma3:4b-it-qat        0/24          0.08                       0/24      0
qwen2.5:7b-instruct     0/24          0.08                       0/24      0
qwen2.5-coder:7b        0/24          0.21                       1/24      0
qwen3:8b (max_tokens=2000)  0/24     0.00                       0/24      0
```

No local model we tested can reproduce the *verbatim* next character of
real Japanese discourse — as expected. The token-level predictability that
the live path actually needs (`prediction_match`) is what the rule backend
already gets for free from the n-gram predictor's memorisation of the
embedded corpus; the local LLM's apparent advantage there did not hold on
this corpus either.

**Parser compatibility fix.** While running the benchmark we found the
shipped parser rejected legal Ollama responses that non-standard models
produce (`next_segment`, `next_sentence`, `continuation`, `next_text`,
`next` in place of `span`, or markdown-fenced JSON). `parse_prediction_payload`
now unwraps markdown fences and accepts those synonym spans; regression
tests cover both paths (`tests/test_sync_core.py`). This does not change
what models predict — it lets them answer at all.

`qwen3:8b` returned an empty string at `max_tokens=400` (its thinking
preamble consumed the budget); at `max_tokens=2000` it answered but scored
identically to the other models. Stand-ins did not buy predictive power.

### 4. N-gram predictor on the Reazon sample

`scripts/tune_ngram.py` sweeps the predictor variants described above against
the 200-file Reazon sample (180 train / 20 eval), prefix 8 chars, order 3 / 4
/ 5, span 40:

```text
model                                  exact_prefix
default (ngram(3) embedded corpus)     0/15
lux-only (trained on 180 train lines)  0/20
merged (embedded + lux-only)           0/20
order 5 n-gram                          0/20 (order 3 / 4 / 5 all zero)
```

Zero exact-prefix anywhere. The embedded corpus contains 12 JP software-
generated lines; train-on-Lux cannot rescue it because natural Japanese is
far too sparse at char-order 5. This is not evidence against the mechanism —
it means no offline corpus substituting for human Japanese reasoning exists
that would make a ppm-matching predictor possible, so predictor quality
cannot be bought for this domain without collecting real sessions.

## Common Voice 8.0 Japanese (fourth corpus arm)

`japanese-asr/ja_asr.common_voice_8_0` — 4,483 Japanese utterances from
many native speakers reading short sentences (CC0 source; the
`japanese-asr` mirror inherits the public license). We ran a 120-file
sample under the rule backend:

```text
files                    120 (CUE + SILENCE paired)
deliveries CUE           84 / 120 (63 files with >=1 delivery)
predicted+intervened     117 / 84 / 0 cancelled
operations delivered     concretize 62, dependency 11, mechanism 6,
                            invert 4, conclusion 1
paired delta             +0.000 CI [+0.000, +0.000] every metric
n_pairs                  120
```

Common Voice is read speech (as is JSUT and Lux), so its boundaries are
sentence-aligned and saturated with punctuation — the trigger-firing
behavior is dominated by the same `concretize` heuristic from the embedded
corpus. Uptake is +0.000 (canned). The corpus proves the pipeline works
end-to-end on a completely unrelated voice/borrowed accent set, not any
human-transfer.

## JSUT-basic5000 (fifth corpus arm; read prose, single female)

`FluidInference/JSUT-basic5000` — 500 utterances of read Japanese (text
CC-BY-SA 3.0 / CC-BY-SA 4.0 / CC-BY 2.0 depending on source sentence;
audio: non-redistribution, research use permitted). Single female
Japanese voice reading newspaper-length sentences — pure read prose with
zero operation vocabulary overlap. Replay with the rule backend:

```text
files                    500 (CUE + SILENCE paired)
predictions              629
interventions delivered  477 (378 files with >=1 delivery)
operations delivered     concretize 376, dependency 58, mechanism 20,
                            invert 13, conclusion 9, ground 1
paired delta             +0.000 CI [+0.000, +0.000] every metric
n_pairs                  500
median delay ms          0 / n_pairs=378
repertoire_growth        0.01 both arms
```

Same story: read-speech Japan of the highest signal-to-noise ratio and
the largest N (500 pairs) produces high firing but zero uptake — exactly
as if the corpus were synthetic, because it's in fact canned read text.

## Common Voice 17.0 Japanese (sixth corpus arm)

`kotoba-speech/common_voice_17_0` — a newer Japanese mirror of Common
Voice 17 with cleaner transcriptions and sentence-level segments. Our
example: 150-file sample replayed under the rule backend; 4 files had no
usable ASR output (probably near-silent) and were counted as `none`, the
rest paired by `pair_key`.

```text
files                    150 (CUE + SILENCE paired)
predictions              149 (4 files with no usable transcript — counted separately)
interventions            69
delivered                69 (45 files with >=1 delivery)
operations delivered     concretize 47, dependency 13, conclusion 5,
                            mechanism 2, ground 2
paired delta             +0.000 CI [+0.000, +0.000] (every metric, on the 146 paired rule arm)
n_pairs                  146
median delay ms          0
```

Same verdict: canned audio signaled over the entire test set. This arm
adds yet another voices corpus; across all six Japanese corpora the same
result surfaces, which rules out an unlucky single-corpus artifact.

## Japanese vlog podcast (seventh corpus arm; closest to native reasoning)

`PlayMaker13/japanese-podcast` — 10,039 clips of a native Japanese
podcaster giving their opinions on common topics (food, vacations,
relationships, language learning). 20 clips sampled, ~17,000 chars total.
Podcasts are natural Japanese monologues with explicit reasoning
announcements (`つまり`, `だから`, `なので`, `なぜなら`) and is the only
corpus in our sweep where **conclusion ops outweigh concretize**:

```text
utterances per id                   729
utterances tagged (tag>=1)         71/729 (9.7%)
op hits: conclusion 31, concretize 16, mechanism 12,
         dependency 6, comparison 5, boundary 2, invert 1
files                     20 / CUE+SILENCE paired
predictions              905
interventions delivered  474
operations delivered     concretize 81, dependency 73, conclusion 67,
                            ground 61, comparison 56, invert 53,
                            mechanism 50, boundary 33
paired delta              +0.000 CI [+0.000, +0.000] every metric
n_pairs                  20
```

The corpus content skews toward Japanese first-person narrative with
explicit opinion-marking language. The speaker voice is close-mic iPhone
grade, slow, consciously paced — a clean whisper transcription target.
What this arm adds is clear: over the seven corpora tested, this is the
closest one whose ASR output has both high enough quality and enough
operation-tagged segments to exercise the runtime on most utterances.

CUE vs SILENCE paired delta is +0.000 CI [+0.000, +0.000] — same as
everywhere, because the speaker is canned. What differs here is that the
corpus engages the predictor heavily: each whisper segment looks like
real reasoning discourse, not idiomatic or read prose.

## local-LLM backend on the PlayMaker13 podcast subset (3 clips)

The same corpus was re-run under `--predictor local --decision local` with
`gemma3:4b-it-qat` on the first 3 clips, paired CUE/SILENCE:

```text
files                3 (podcast_000/001/002)
predictions          83
interventions delivered  83
operations delivered  comparison 30, ground 19, boundary 18, dependency 16
analyzer n_pairs=3, uptake/carryover delta +0.000 (canned audio)
```

Same story: every CUE cue delivered, uptake still +0.000 because audio
is canned. What's new here is that even the language model which
correctly fails at next-token prediction textually (exact-prefix=0
throughout) produces downstream-usable cue events — the predictor's
output is evaluated for operation signals, not token identity.

## What was added to CUELOOP from this evaluation

- `scripts/bench_predictors.py` — one-command local-LLM next-span
  benchmark on any compatible corpus; covers exact-prefix rate, average
  longest-common-prefix, operation-tag hit rate, and parse failures
  (tested at 24-token default; qwen3 reasoner needs `max_tokens ≥ 400`
  or it returns an empty string).
- `scripts/tune_ngram.py` — offline n-gram evaluation/tuning harness on any
  `metadata.csv`. Useful once a proper corpus appears for the target domain.
- `tools/file_asr.swift` — Swift one-shot recognizer per Lux-style file,
  for direct SFSpeechRecognizer comparison when the license is accepted.
- `scripts/simulate_lux_replay.py` — replays Lux *or any compatible corpus*
  (CSV with file_name/cleaned_path/transcript) through the actual
  `SyncRuntime`, emitting analyzer-ready paired reports (CUE and SILENCE).
  On ReazonSpeech it delivers 455 cues with the rule backend; under the
  fixed runtime local-LLM deliveries complete at the boundary event as well
  (668/668 on the full 200-file arm). Also fixes a cache-subset bug where a
  warm whisper cache returned extra files not in the requested corpus
  (regression test included).

Note: the Lux corpus is literary Japanese (read speech) with essentially no
operation-tag vocabulary overlap — both the embedded n-gram and the local
LLM produce predictions whose operation tags never coincide with the real
operations in the next sentence. This is intrinsic to the corpus (it is not
a reasoning dataset) rather than something that can be fixed by tuning the
predictor.

See `docs/benchmark.md` for the synthetic evaluation and
`docs/experiment-protocol.md` for the human study design.

## Rule arm on full Lux replay (328 files)

The `rule` (default n-gram prediction-gap) pipeline was fired over every
Lux file. Counts (from the analyzer,`artifacts/lux-rule-328-analysis.md`):

```text
files                  328
predictions fired      405
interventions delivered 276 (held on SILENCE: 276)
cancellations          0
outcomes               276
uptake / carryover     0.00 / 0.00 (both arms identical; read speech)
repertoire_growth      0.04
```

Distribution of chosen operations (after fatigue tuning): `concretize 194`,
`dependency 50`, `mechanism 13`, `conclusion 8`, `invert 6`, `ground 4`,
`comparison 1`. Cues used: `具体は？` 194×, `何に？` 50×, `仕組みは？` 13×,
`結論は？` 8×, `逆は？` 6×, `根拠は？` 4×, `何と比べる？` 1×. The rule backend
now applies per-session repetition fatigue (`fatigue_penalty=2.0`, see
`RuleDecisionModel`): before it, concretize took 264/276 (96%); after it the
mix spans 7 distinct operations while the delivery count (276), prediction
count (405) and every paired delta (+0.000) are unchanged.

The rule fires on the embedded n-gram's predictions at the small pauses in
the TTS read-through — both arms receive an identical count of
interventions so the paired delta is `+0.000` on every metric. The
trigger is *not* meaningful announcement detection — Lux is a TTS
proofreader script. Before fatigue tuning, most cues fired on `concretize`
because the embedded corpus is dominated by example-language (`例えば`,
`具体的`); the fatigue penalty now rotates the mix (see above).

## Local-LLM end-to-end on the Lux corpus (post-fix)

> The pre-fix Lux measurements were invalidated by the runtime latency clamp:
> a synchronous local-LLM decision was pushed beyond the next audio event and
> then cancelled by the synchronizer. Lux was re-measured under the fixed
> runtime (`artifacts/lux-llm-328-full-v2/`):

```text
file                              328 CUE + 328 SILENCE (reports at artifacts/lux-llm-328-full-v2/)
predictions                      333 CUE / 332 SILENCE (201/328 files with predictions both arms)
interventions                    324 CUE / 325 SILENCE
delivered                        324 CUE / 0 SILENCE (all delivered, 0 cancelled)
files with >= 1 prediction       201 / 328
operations delivered             dependency 122, boundary 107, ground 47,
                                 comparison 37, mechanism 11
uptake / carryover               0.00 / 0.00 (identical; read speech)
pair delta                       +0.002 CI [+0.000, +0.005] (not significant)
n_pairs                          328
median boundary->delivery delay   0 both arms (n_pairs=198)
repertoire_growth                0.03 both arms
```

Earlier pre-fix run: 480 predictions, 0 delivered (both arms) — recorded for
transparency below this section.

```text
file                              328 CUE + 328 SILENCE
predictions                      480 (identical in both arms)
interventions                    457 CUE / 466 SILENCE
delivered                        0  (both arms; all interventions cancelled/held)
files with >= 1 prediction       201 / 328
predictions not intervened        23 CUE / 14 SILENCE (decision model vetoed
                                 its own prediction)
uptake / carryover               0.00 / 0.00 (identical; read speech)
paired delta                     +0.000 on every metric
n_pairs                          328
median boundary->delivery delay   insufficient (no deliveries)
repertoire_growth                0.06 (identical both arms; reflects operation
                                 tags present in the transcript, not an effect
                                 — nothing was delivered)
```

CUE and SILENCE predictions are identical because the predictor and the seed
are deterministic; interventions differ by 9 because the local-LLM decision
is not bit-deterministic across processes (Ollama batching/scheduling).

Idempotent `--limit` runs: the earlier `--limit 50` run produced
`files=50, predictions=60, interventions=60, delivered=0, cancelled=60`,
fully consistent with the complete 328-file run.

Every intervention is cancelled (CUE) or held (SILENCE) because the actually
spoken text diverges from the prediction before the speaker pauses long
enough for a cue to land. This is the honest state: **the pipeline decides
when to speak, but never manages to say it during read speech** — the corpus
contains single-shot literary sentences, not an unfolding reasoning stream.
What the run *does* verify is that the whole path — per-word ASR events →
prediction → prediction-gap decision → SILENCE hold / CUE cancel — composes
on real Japanese audio in both arms.

The most frequent predictor engagements are on `EMOTION100_072` (13
predictions) and `EMOTION100_011` (7) — driven by surface trigger vocabulary
(`例えば`, `理由`), not by announced reasoning operations.

To improve:

- (a) a Japanese software-reasoning speech corpus in CUELOOP's operation
  vocabulary — not available publicly;
- (b) real software-reasoning recordings — not available publicly;
- (c) accept that the reliable arm on this corpus is the `rule` backend with
  the bare cue (276 delivered, all `具体は？`-heavy), not the local LLM.

## ReazonSpeech sample: vocabulary-domain arm (YouTube Japanese)

YouTube-derived ReazonSpeech audio is the closest publicly available corpus
to the register CUELOOP was built for, and it does contain CUELOOP's
operation vocabulary:

```text
sample:         200 files, ≥6 s and ≥15 chars, total ≈ 52 minutes
ops tagged in transcripts:  30 / 200 files (conclusion 10, concretize 7,
                            dependency 6, mechanism 5, boundary 3,
                            invert 1, ground 1, comparison 1)
whisper-base-mlx vs dataset whisper-turbo pseudo-labels: 78.0% char accuracy
                            (0/200 exact; both are machine transcriptions,
                            so this is an agreement bound, not an accuracy bound)
```

### Rule backend, 200-file sample

```text
files                    200 (CUE and SILENCE paired)
predictions fired        894  (vs 405 on the Lux read corpus; 2.2×)
interventions delivered  455  (CUE)
held (SILENCE)          455
cancellations            0
files delivering ≥ 1 cue 133 / 200
max cues per file         9
operations delivered     concretize 207, dependency 134, conclusion 60,
                            mechanism 23, ground 18, invert 13
                            (fatigue tuning; before: concretize 427, dependency 2)
paired uptake delta     +0.000 (both arms identical: the continuation text
                            is identical in both arms, so target_operation_seen
                            fired 8/455 times in both)
```

### Local-LLM backend, 200-file full sample (qwen2.5:1.5b-instruct, post-fix)

Post-fix re-measurement:

```text
files                    200 CUE + 200 SILENCE (artifacts/reazon-llm15b-200-v2/)
predictions              741 CUE / 723 SILENCE (142/200 files with predictions
                          in both arms; not bit-deterministic)
interventions            327 CUE / 314 SILENCE
delivered                327 CUE / 0 SILENCE
files with ≥ 1 prediction 142 / 200
operations delivered     ground 133, boundary 122, concretize 58,
                            comparison 13, dependency 1
analyzer n_pairs         200
paired uptake delta     +0.002 CI [-0.006, +0.011] (not significant)
paired carryover delta  +0.000 CI [+0.000, +0.000]
repertoire_growth        -0.025 CI [-0.060, +0.010] both arms
median delay ms          0 / 0.00 (n_pairs=108)
```

Pre-runtime-fix precursor (superseded): 986/973 predictions, delivered 2
(`boundary 範囲は？` on `ミリカ!`, `ground 根拠は？` on `おととい、`),
analyzer n_pairs=200, every metric +0.000. Retained here for calibration
transparency. These were the first ever local-LLM deliveries on real
Japanese; the fix supersedes the framing — the local-LLM arm now delivers
in the same way the rule backend does (post-fix: 327/327, 0 cancelled).

### Local-LLM backend, 200-file full sample (gemma3:4b-it-qat, post-fix)

The full 200-file re-run under the fixed runtime (see CHANGELOG
"Runtime delivery clamp") — both arms, same seed, same whisper cache:

```text
run                     artifacts/reazon-llm-200-v2/ (200 paired CUE+SILENCE reports)
predictions              687 CUE / 687 SILENCE (142/200 files with predictions
                          in both arms)
interventions            668 CUE / 664 SILENCE
delivered                668 CUE / 0 SILENCE (0 cancelled, 0 expired)
files with ≥ 1 delivery  142 / 200
max delivered per file   13
operations delivered     ground 225, boundary 179, dependency 168,
                            comparison 72, mechanism 23, conclusion 1
paired uptake delta     +0.002 CI [-0.000, +0.007]  (both arms 0.01)
paired carryover delta  +0.000 CI [+0.000, +0.000]
analyzer n_pairs         200, report at artifacts/reazon-llm-200-v2-analysis.md
median delay ms          0 / 0.00 in both arms
repertoire_growth        0.14 both arms (nothing carried over)
```

Deliveries now complete at the boundary event itself; 0 were cancelled.
The analyzer shows no measurable uptake, as expected — both arms replay
the same canned transcript, so any uptake excess (7/668 in CUE vs 7/664
in SILENCE) is an alignment artefact, not an effect.

The older (pre-fix) run of the same model had every intervention cancelled
after the decision returned measurably later than the next audio event.
That failure mode is gone.

### Local-LLM backend, 30 operation-tagged files (gemma3:4b-it-qat, post-fix)

The 30 files whose transcripts tag at least one CUELOOP operation — the
domain where a cue has a real target — re-run under the fixed runtime:

```text
files                    30 (CUE and SILENCE paired)
predictions fired        143 CUE / 142 SILENCE (not bit-deterministic)
interventions            138 CUE / 141 SILENCE
delivered                138 CUE / 0 SILENCE (max 12 per file; 25/30 files
                            at least one delivery)
analyzer n_pairs         30
paired uptake delta     +0.022 CI [+0.000, +0.061]
paired carryover delta  +0.000 CI [+0.000, +0.000]
median delay ms          0 in both arms (analyzer CI [+0.000, +0.000], n_pairs=25)
repertoire_growth        0.73 both arms (nothing carried over)
operations delivered     ground 59, dependency 29, boundary 28,
                            comparison 16, mechanism 6
```

The +0.022 uptake delta is a count artefact of the canned arm pairing
(CUE has 28 outcomes vs SILENCE's 33, with the same operations appearing
by construction); the CI touches zero and the result is NOT evidence of
the mechanism working. It proves delivery completes end-to-end.

The local-LLM predictor produces plausible continuations but never the exact
author's wording — so after each boundary the synchronizer marks the speech
diverged and every intervention is cancelled. The rule backend survives the
same data because its echo-like n-gram predictions partially match the
continuation often enough to keep the synchronizer un-diverged long enough
for the delivery window to elapse; that is not foresight, it is substring
echo, and it is why the rule arm can deliver while the local-LLM arm cannot.

**Honest framing.** This corpus changes the *base rate*, not the mechanism:
CUELOOP's operation vocabulary is genuinely present in real Japanese
discourse, the rule backend fires and delivers cues against it (455 times,
concentrated on `具体は？`), and the local-LLM backend fires often but never
completes a delivery. Neither result says anything about whether a human
hearing a cue would shift their reasoning — that remains the open H1/H2
question for live sessions.

### Artifacts and reproduction

```bash
# whisper transcription (already cached in data/real-speech/reazon/)
# full 200-file sample, rule backend (~4 min)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/reazon/metadata-sample-200.csv \
  --source-name reazon --output-dir artifacts/reazon-rule-200 --predictor ngram --decision rule
.venv/bin/python scripts/analyze_experiment.py artifacts/reazon-rule-200/cue \
  artifacts/reazon-rule-200/silence --markdown artifacts/reazon-rule-200-analysis.md

# 30 operation-tagged files, local-LLM backend (~15 min; needs Ollama + gemma3)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/reazon/metadata-sample-tagged.csv \
  --whisper-cache data/real-speech/reazon/metadata-sample-200-whisper-results.json \
  --source-name reazon-tagged-v2 --output-dir artifacts/reazon-llm-tagged-v2 \
  --predictor local --decision local --local-model gemma3:4b-it-qat
.venv/bin/python scripts/analyze_experiment.py artifacts/reazon-llm-tagged-v2/cue \
  artifacts/reazon-llm-tagged-v2/silence --markdown artifacts/reazon-llm-tagged-v2-analysis.md

# full 200-file sample, fast local model (~15 min; needs Ollama + qwen2.5:1.5b)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/reazon/metadata-sample-200.csv \
  --whisper-cache data/real-speech/reazon/metadata-sample-200-whisper-results.json \
  --source-name reazon-llm15b-v2 --output-dir artifacts/reazon-llm15b-200-v2 \
  --predictor local --decision local --local-model qwen2.5:1.5b-instruct
.venv/bin/python scripts/analyze_experiment.py artifacts/reazon-llm15b-200-v2/cue \
  artifacts/reazon-llm15b-200-v2/silence --markdown artifacts/reazon-llm15b-200-v2-analysis.md

# full Lux corpus, local-LLM backend (~10 min; needs Ollama + gemma3)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/lux/metadata.csv \
  --source-name lux-v2 --output-dir artifacts/lux-llm-328-full-v2 \
  --predictor local --decision local --local-model gemma3:4b-it-qat
.venv/bin/python scripts/analyze_experiment.py artifacts/lux-llm-328-full-v2/cue \
  artifacts/lux-llm-328-full-v2/silence --markdown artifacts/lux-llm-328-full-v2-analysis.md

# full 200-file sample, gemma3:4b-it-qat (~80 min total; needs Ollama + gemma3)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/reazon/metadata-sample-200.csv \
  --whisper-cache data/real-speech/reazon/metadata-sample-200-whisper-results.json \
  --source-name reazon-fulllm9 --output-dir artifacts/reazon-llm-200-v2 \
  --predictor local --decision local --local-model gemma3:4b-it-qat
.venv/bin/python scripts/analyze_experiment.py artifacts/reazon-llm-200-v2/cue \
  artifacts/reazon-llm-200-v2/silence --markdown artifacts/reazon-llm-200-v2-analysis.md

# full corpus replay, rule backend
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_dialog_replay.py --output-dir artifacts/dialog-rule \
  --predictor ngram --decision rule --source-name dialog-rule
.venv/bin/python scripts/analyze_experiment.py artifacts/dialog-rule/cue \
  artifacts/dialog-rule/silence --markdown artifacts/dialog-rule-analysis.md

# full corpus replay, local-LLM backend (may take ~10-30 min per arm; needs Ollama + gemma3)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_dialog_replay.py --output-dir artifacts/dialog-llm-1 \
  --predictor local --decision local --local-model gemma3:4b-it-qat --source-name dialog-llm \
  --limit 1
.venv/bin/python scripts/analyze_experiment.py artifacts/dialog-llm-1/cue \
  artifacts/dialog-llm-1/silence --markdown artifacts/dialog-llm-1-analysis.md

# JSUT-basic5000, rule backend
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/jsut-replay/metadata.csv \
  --source-name jsut-basic5000 --output-dir artifacts/jsut-replay \
  --predictor ngram --decision rule
.venv/bin/python scripts/analyze_experiment.py artifacts/jsut-replay/cue \
  artifacts/jsut-replay/silence --markdown artifacts/jsut-replay-analysis.md

# corpus acquisition (audio not in git):
#   huggingface.co/datasets/MagicDataTech/Japanese_Spontaneous_Conversation_Training_Dataset
#   common voice 8.0 Japanese mirror (ja_asr.common_voice_8_0), CC0
#   kotoba-speech/common_voice_17_0 (mirror of the CC0 Common Voice 17 ja test set)
#   huggingface.co/datasets/FluidInference/JSUT-basic5000 (audio no redistribution; research use free)
#   bandad/asr-testset-kw-ja-v1 (license unspecified in card data; used for internal eval only)
#   PlayMaker13/japanese-podcast (license unspecified in card data; public permissive "listen to anyone" podcast)

# podcast 3-clip local-LLM arm (~10 min, needs Ollama + gemma3:4b-it-qat)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/podcast-replay/metadata.csv \
  --source-name podcast-llm --output-dir artifacts/podcast-llm-3 \
  --predictor local --decision local --local-model gemma3:4b-it-qat --limit 3

# extended 30-file JSUT local-LLM arm
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/jsut-replay/metadata.csv \
  --source-name jsut-llm-30 --output-dir artifacts/jsut-llm-30 \
  --predictor local --decision local --local-model gemma3:4b-it-qat --limit 30

# kw-ja 10-file local-LLM arm
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/bandad-kw/metadata.csv \
  --source-name bandad-llm --output-dir artifacts/bandad-llm-10 \
  --predictor local --decision local --local-model gemma3:4b-it-qat --limit 10

# common voice 17.0 Japanese (150-file sample), rule backend
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/cv17-ja/cleaned-metadata.csv \
  --source-name commonvoice17-ja --output-dir artifacts/cv17-replay \
  --predictor ngram --decision rule
.venv/bin/python scripts/analyze_experiment.py artifacts/cv17-replay/cue \
  artifacts/cv17-replay/silence --markdown artifacts/cv17-replay-analysis.md

# common voice 8.0 Japanese (120-file sample), rule backend
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 \
  scripts/simulate_lux_replay.py --corpus data/real-speech/common_voice_sample_replay/metadata.csv \
  --source-name commonvoice8-ja --output-dir artifacts/commonvoice-ja-replay \
  --predictor ngram --decision rule
.venv/bin/python scripts/analyze_experiment.py artifacts/commonvoice-ja-replay/cue \
  artifacts/commonvoice-ja-replay/silence --markdown artifacts/commonvoice-ja-replay-analysis.md
```

## MagicData Japanese spontaneous dialogue (third corpus arm)

`MagicDataTech/Japanese_Spontaneous_Conversation_Training_Dataset`
(Hugging Face dataset tag: Apache-2.0; used here for non-redistributive
evaluation and never committed — the WAVs stay in a git-ignored
`data/real-speech/` cache): 20 two-person smartphone conversations,
~7,477 utterances, 129,707 Japanese characters with publisher-provided
per-utterance timestamps and two-speaker diarization. The audio is
natural Japanese — questions interrupted, false starts, overlapping
speech, backchannels, filler — not read prose or broadcast narration.

Two measurements on this corpus with the rule backend (replaying
utterance chunks spread evenly across the publisher's `[start_ms, end_ms]`):

```text
files                                 20 CUE + 20 SILENCE (paired)
utterances with Japanese content      7477
utterances tagging >=1 operation      337 (4.5%)
operation hit counts                  conclusion 161, comparison 69,
                                       concretize 45, ground 21, boundary 19,
                                       invert 15, mechanism 12, dependency 11
delivered (CUE)                        5527
uptake / carryover                     0.01 / 0.04 (identical in SILENCE;
                                       canned audio cannot respond)
paired delta                           +0.000 CI [+0.000, +0.000] (every metric)
n_pairs                                20
```

The rule backend fires on virtually every boundary in this corpus (411
predictions in the first file alone, 316 delivered). Uptake on any canned
corpus is structurally 0, which it is here; the corpus does, however,
contain real reasoning-with-announcements (e.g., `結論言うと`, `だから、
そういった面では`), so this arm is the first committed one with
genuinely spontaneous reasoning audio. Tagged-operation coverage (4.5% of
utterances) is small but non-zero; that's the same as the Lux and Reazon
arms — no working corpus has dense announcements.

A local-LLM arm on this corpus is possible but expensive — each of the
~400 utterances in a file is its own local-LLM call. We ran a single-file
smoke test to demonstrate the local-LLM end-to-end path on real conversational
Japanese (not just a simulator):

```text
file                    A0022_S0003_0_G0029 (two-person smartphone conv)
predictions              394
interventions            387
delivered                387  (boundary → delivery complete, canned audio)
operations delivered      ground 135, dependency 88, comparison 83,
                          boundary 71, mechanism 9, conclusion 1
analyzer (n_pairs=1)     uptake delta −0.003 (insufficient CI);
                         carryover delta +0.010 (insufficient CI)
```

Its practical value is proving the live local-LLM path composes on real
conversation — not any effect measurement. A full arm is available via
`scripts/simulate_dialog_replay.py --predictor local --decision local` at
the same cost per utterance.

## How to reproduce these numbers

```bash
# ASR evaluation (whisper-base-mlx) — see docs/real-speech.md results above.
.venv/bin/python - <<'PY'
import csv, mlx_whisper, time
t0 = time.time()
rows = list(csv.DictReader(open('data/real-speech/lux/metadata.csv', encoding='utf-8')))
total = 0; match = 0
for row in rows:
    if not row['cleaned_path'].startswith('data/cleaned/ita_recitation/'):
        continue
    wav = 'data/real-speech/lux/' + row['cleaned_path']
    result = mlx_whisper.transcribe(
        wav, path_or_hf_repo='/tmp/whisper-base',
        language='ja', task='transcribe', fp16=False, verbose=None,
    )
    pred = result['text'].strip()
    truth = row['transcript'].strip()
    sm = __import__('difflib').SequenceMatcher(None, pred, truth)
    match += sum(m.size for m in sm.get_matching_blocks())
    total += len(truth)
print(f"{total} chars, character accuracy: {match / total * 100:.1f}%, time {time.time() - t0:.1f}s")
PY

# n-gram evaluation
.venv/bin/python scripts/tune_ngram.py --corpus data/real-speech/lux/metadata.csv \
  --order 3 --prefix-chars 8 --span-characters 8 \
  --eval-mod 10 --eval-remainder 0

# full Lux replay, rule backend (~2 minutes)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 scripts/simulate_lux_replay.py \
  --output-dir artifacts/lux-rule-328 --predictor ngram --decision rule
.venv/bin/python scripts/analyze_experiment.py artifacts/lux-rule-328/cue artifacts/lux-rule-328/silence \
  --markdown artifacts/lux-rule-328-analysis.md

# full Lux replay, local-LLM backend (path as above, ~10 minutes)
PYTHONPATH=.venv/lib/python3.12/site-packages /opt/homebrew/bin/python3 scripts/simulate_lux_replay.py \
  --output-dir artifacts/lux-llm-328-full-v2 --predictor local --decision local \
  --local-model gemma3:4b-it-qat --source-name lux-v2
.venv/bin/python scripts/analyze_experiment.py artifacts/lux-llm-328-full-v2/cue artifacts/lux-llm-328-full-v2/silence \
  --markdown artifacts/lux-llm-328-full-v2-analysis.md
```

Whisper characterization needs `mlx-whisper` installed; the pip experience on
PyPI for MLX is slow (~20 minutes per wheel). The Lux corpus is deliberately
git-ignored — don't commit the WAVs.

## Next steps

1. Accept Xcode license on the developer laptop and run the macOS on-device
   recognizer on Lux/Reazon files (tools/file_asr.swift + a tiny harness),
   comparing error profile vs whisper-base-mlx.
2. **Human sessions remain the only gate to H1/H2**: Japanese reasoning
   speech recorded live with a speaker who can hear and react to cues.
   The Reazon arm shows real discourse with the vocabulary present and cues
   deliver end-to-end, but canned arms replay the same text both rounds, so
   no effect on a human has been measured. Issue #6.
3. With a Reazon-style discourse corpus in hand, the most valuable next
   technical step is *domain-adapted prediction*: measure whether the local
   LLM (or n-gram) can anticipate operations in the 30-tagged-file subset
   with enough fidelity to keep the synchronizer un-diverged past the
   delivery window — that is the only way local-LLM cues can complete a
   delivery on recorded speech.
4. Expand the embedded corpus on Japanese software engineering (code review,
   design discussion) texts; re-evaluate `exact_prefix_rate`. The embedded
   corpus is the single largest source of false-positive predictions.
