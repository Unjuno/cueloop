# GPT Live × CUELOOP protocol (Japanese reasoning, agent-side uptake)

This note defines how to use this machine to pair the ChatGPT app's GPT
Live voice with CUELOOP, so that Japanese reasoning speech produced by
GPT Live is fed into CUELOOP's ASR, and CUELOOP's cues are audited by
both the ASR pipeline and the GPT Live agent.

**Warning: agent-side uptake ≠ human-transfer uptake.** GPT Live is an
LLM-driven voice agent. It can hear CUELOOP's Kyoko cue; it may adopt
it; but that signal does not prove human-subjective resonance — that
would require a real human listener. This protocol measures
*agent-side uptake* only.

## Roles

- `GPT Live` — ChatGPT app on macOS (Japanese voice mode). It speaks
  Japanese reasoning reasoning aloud, listens for replies, and speaks
  its continuation.
- `CUELOOP` — this repo; runs on the same Mac; ASR via SFSpeechRecognizer
  (locale ja-JP); intervention via `cueloop-sync`.
- Mac OS
- Owner — operant at the console; presses start/stop.

## Topology

```text
  ChatGPT app (GPT Live)        CUELOOP (this repo)
  ─────────────────────────     ─────────────────────────
  speaks Japanese reasoning     ASR (built-in mic)
  ▼                              ▼
  near speakers                 macos-speech-events
  ▼                              ▼
  Mac audio output (speakers)   sync loop
  ────────────────────────► CUELOOP ASR
                                 (locale ja-JP)
                                 ▼
                                prediction / boundary
                                 ▼
                                intervene + speak cue
                                 ▼
                                CUELOOP Kyoko voice
                                 ▼
  ┌────────────────────────────→  GPT Live hears cue
  │                              (if speakers are exposed; else
  │                               CUELOOP won't be heard back)
  └────────────────────────────┘
```

GPT Live and CUELOOP must share the same audio environment. The two
reliable options are:

1. **Speakers → microphone.** GPT Live reads through the Mac's speakers;
   the built-in mic picks up; the system audio output level must not clip
   into SFSpeechRecognizer's noise floor.
2. **BlackHole 2ch (free virtual audio device).** GPT Live's output is
   routed to BlackHole 2ch; SFSpeechRecognizer's input is also BlackHole
   2ch. Installation requires a system restart and owner consent — only
   run if the owner explicitly asks for it.

## Recording procedure

```bash
# CUE arm — cues are voiced via Kyoko and audible to GPT Live
CUELOOP_TTS=1 .venv/bin/cueloop-sync \
  --event-command-json '[".build/macos-speech-events","--locale","ja-JP","--duration","300","--speech-db","-50"]' \
  --decision rule --predictor ngram \
  --tts-command-json '["say","-v","Kyoko"]' \
  --condition-label CUE --assignment canonical --seed 3 --cue-style bare \
  --pair-key gpt-live-<topic>-$(date +%Y%m%d-%H%M%S) \
  --session-id gpt-live-<topic>-cue \
  --log logs/gpt-live/<topic>-cue.jsonl --report logs/gpt-live/<topic>-cue.report

# SILENCE arm — no Kyoko, GPT Live still speaks; the matched twin
.venv/bin/cueloop-sync \
  --event-command-json '[".build/macos-speech-events","--locale","ja-JP","--duration","300","--speech-db","-50"]' \
  --decision rule --predictor ngram \
  --condition-label SILENCE --assignment canonical --seed 3 --cue-style bare \
  --silent \
  --pair-key gpt-live-<topic>-<same-stamp-as-cue> \
  --session-id gpt-live-<topic>-silence \
  --log logs/gpt-live/<topic>-silence.jsonl --report logs/gpt-live/<topic>-silence.report
```

Practical refinements:
- Keep the same topic slate across pairs (design-choice / governance / ergonomics / launch-readiness).
- Wait 3–5 seconds between arms to let SFSpeechRecognizer reset.
- `--speech-db -50` is appropriate for speaker→mic;
  if you routed through BlackHole, set `--speech-db -20` instead (no room-noise floor).
- GPT Live should reason aloud about evaluative design choice topics
  (counterpart of the live human study) — same prompt descriptor style.

## Analysis

```bash
# Per-pair analysis (after both arms)
.venv/bin/python scripts/analyze_experiment.py logs/gpt-live-<topic>-paired-<stamp> \
  --markdown logs/gpt-live-<topic>-paired-<stamp>/analysis.md
```

Metrics to watch for (they are NOT directly comparable to the human study):

| Metric | Agent-side meaning | Human-side meaning |
| --- | --- | --- |
| uptake_rate | GPT Live reuses op-token | Human chooses op unaided |
| repertoire_growth | GPT Live picks up new op-family | Human gains op via cues |
| carryover | GPT Live keeps op into next clauses | Human repeats op naturally |
| novel_uptake | first-time op after cue | transfer |

Agent-side uptake is lower-bounded by GPT Live's own op-vocab anchors it
uses (boundary, dependency etc.); expect which ops dominate unpredictably.
Continued observation in the next sessions will surface the pattern.

## Caveats & honesty notes

1. **Agent uptake ≠ human uptake.** Recording the OpenAI assistant
   uptake is not evidence that a human listener would uptake. Results
   here corroborate of ICBLOOP's pipe in interactive stream conditions.
2. **Recognition stability.** GPT Live's voice is high-quality synthetic
   audio but its token stream is untrained-to-our-domain vocabulary;
   some boundary/trigger-op pairings may produce misfires.
3. **Latency.** Kyoko announcement × ~3 s per cue. GPT Live hearing it
   is possible only with acoustically reasonable setup; it may utterly
   ignore the cue without reacting. We record exactly that.
4. **Goals of the experiment stay intact.** This path (a) increases live
   ASR feed, (b) gives a new, different measurement of "agent-side
   uptake" comparatively with the human arm. It does NOT substitute for
   the H1/H2 human study.
5. **Goal-closing is blocked as always.** No result from GPT Live closes
   the goal; the ROI milestone for H1/H2 is still human-talk.

## Open questions / next steps

1. Routing option 2: do we want to install BlackHole 2ch on this Mac?
   (Needs owner consent.)
2. If not, can the mic feed GPT Live voice directly? Empirical test
   with 20-30-second runs before running the full arm.
3. First pair — same topic design-choice for continuity with the
   human-arm data.
4. Generator/Q rings: gpt-live-protocol scenario tuning to abstract
   from the starter, but matching the human study protocol for
   cross-comparison.
