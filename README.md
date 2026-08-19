# CUELOOP

CUELOOP is a voice-first cognitive intervention system designed to help people think without replacing their reasoning.

## What CUELOOP does

CUELOOP listens while a person thinks aloud and chooses among a small set of interventions:

- stay silent
- acknowledge and preserve the user's turn
- ask a minimal question
- mirror the current reasoning structure
- introduce a counterexample or constraint
- retrieve missing knowledge
- mark claims for later verification

The system does not generate a completed answer by default. Its job is to improve the conditions under which the user reasons, while keeping the user as the primary reasoning agent.

The initial product is a 10–30 minute spoken thinking session for:

- research hypotheses
- product and system design
- complex decision-making
- writing and argument development
- technical problem decomposition

After the session, CUELOOP can extract:

- claims
- premises
- evidence
- dependencies
- counterexamples
- unresolved questions
- claims requiring verification

## Why CUELOOP exists

Frontier AI development has largely focused on improving external models through more compute, more data, larger models, and better optimization.

That direction has produced powerful systems, but it usually treats the human as a consumer of model outputs.

CUELOOP starts from a different hypothesis:

> Humans already possess a flexible, grounded, continuously adapting cognitive architecture that current AI interfaces do little to improve.

Pretrained models provide broad prior knowledge, rapid retrieval, and cross-domain patterns. Humans provide continual learning, real-world grounding, goal formation, value judgment, flexible adaptation, and responsibility.

CUELOOP attempts to connect these complementary capabilities.

Instead of using the model primarily to produce completed answers, it converts model knowledge and reasoning into minimal interventions that help the human retrieve, generate, restructure, verify, and update their own thinking.

The objective is not merely better AI-assisted output.

The objective is greater retained human capability after the assistance is removed.

This project exists to address four problems:

1. **Answer substitution**  
   Completed model outputs can replace the user's own generation process.

2. **Weak internalization**  
   A user may perform better while assisted without retaining the reasoning structure after assistance is removed.

3. **Poor updateability**  
   Memorized answers are difficult to revise consistently when a premise or counterexample changes. Structured reasoning makes downstream revision possible.

4. **Over-intervention**  
   Conventional assistants are optimized to respond. In a thinking session, silence or a two-word cue may be more useful than a full answer.

CUELOOP therefore optimizes for minimal, context-dependent intervention, later verification, and eventual cue reduction—not maximum response volume.

For the full origin, architectural thesis, and epistemic limits, see [`docs/origin-and-thesis.md`](docs/origin-and-thesis.md).

## Core hypothesis

> A pretrained model's broad knowledge and intervention capabilities can be converted into minimal, context-sensitive feedback that helps a human update their own reasoning policies and internal representations.

This is a design hypothesis, not an established scientific result.

## Core principles

1. The human generates first.
2. Silence is a valid system action.
3. The system should not produce a completed answer by default.
4. Interventions should be minimal and context-dependent.
5. Knowledge input, generation support, and verification remain separate.
6. Successful use should reduce—not increase—dependence on the system.
7. Real-time performance is insufficient; useful reasoning structures should remain after assistance is removed.
8. Claims about cognitive enhancement must not be made without evidence.

## Initial action space

- `SILENCE`
- `BACKCHANNEL`
- `QUESTION`
- `MIRROR`
- `PERTURB`
- `CONSTRAIN`
- `RETRIEVE`
- `VERIFY`

## Initial human-state model

- `PRODUCTIVE`
- `RETRIEVAL_FAILURE`
- `STRATEGY_FAILURE`
- `FIXATION`
- `STRUCTURE_GAP`
- `OVERLOAD`

These are short-lived operational interaction states, not diagnoses of a person's cognitive ability.

## Architecture

```text
Microphone
    ↓
Voice activity detection
    ↓
Streaming speech recognition
    ↓
Fast path
- turn and pause detection
- intervention suppression
- lightweight state classification
- cached cue playback
    ↕
Shared session state
    ↕
Slow path
- language-model reasoning
- claim and structure extraction
- intervention candidate generation
- knowledge retrieval
- verification
```

The language model proposes hypotheses and intervention candidates. A deterministic controller decides whether an intervention is actually delivered.

## Working prototype

The repository now contains an end-to-end, provider-neutral prototype:

- `cueloop`: interactive text reasoning session
- `cueloop-voice`: streaming VAD/ASR event ingestion, turn boundaries, suppression, optional TTS, and local logs
- `cueloop-structure`: post-session extraction of claims, premises, evidence, dependencies, counterexamples, questions, and verification items
- `cueloop-ground`: local-document retrieval for reviewable source candidates
- `cueloop-experiment`: held-out policy evaluation and bounded automatic tuning
- `cueloop-feedback`: explicit feedback that can only suppress or fade future interventions
- `cueloop-memory`: explicit, expiring retention and forgetting of user-selected reasoning structures
- `cueloop-soak`: deterministic continuous-session load test
- `cueloop-delete-data`: dry-run-first deletion of local session data
- `cueloop-pilot`: counterbalanced assignment and delayed-reasoning pilot analysis
- `cueloop-audit`: executable completion-evidence gate across the full prototype
- `cueloop-session`: atomic end-to-end session with manifest and integrity verification

Install with Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Run the reproducible policy experiment and the voice-pipeline smoke session:

```bash
cueloop-experiment
cueloop-voice --events experiments/voice-smoke.jsonl --policy artifacts/policy.json
```

On macOS, add audible cues with `--tts-command say`. The repository includes an Apple Speech/AVFoundation microphone adapter; the build command, JSONL contract, and privacy procedure are documented in [`docs/voice-mvp.md`](docs/voice-mvp.md).

Run the complete session workflow (voice decisions, structure, grounding, retention metadata, and integrity manifest):

```bash
cueloop-session run \
  --events experiments/voice-smoke.jsonl \
  --goal "検証可能な設計仮説を作る" \
  --sources README.md docs/product-axioms.md \
  --tts-command say \
  --session-id first-session
cueloop-session verify data/sessions/first-session
```

Artifacts are committed atomically: malformed input or downstream failure leaves no partial session directory. Use `--no-transcript-retention` to retain only derived structure and the manifest.

Preview and apply retention expiry:

```bash
cueloop-session purge
cueloop-session purge --confirm
```

Build and ground a post-session structure:

```bash
cueloop-structure logs/voice-session.jsonl
cueloop-ground artifacts/session-structure.json notes/article.txt notes/paper.txt
```

Reasoning structures are not added to long-term memory automatically. Explicitly retain a selected item from an integrity-verified session, inspect it, and forget it with:

```bash
cueloop-memory retain data/sessions/first-session C1 --retention-days 90
cueloop-memory list
cueloop-memory forget structure-ID
```

All recording, transcription, and profile data remains local by default. Generated data lives under ignored `data/`, `logs/`, and `artifacts/` paths. Inspect and then delete all local session data:

```bash
cueloop-delete-data
cueloop-delete-data --confirm
```

The human retained-reasoning experiment, counterbalanced assignment, scoring fields, stop rules, and analysis commands are documented in [`docs/experiment-protocol.md`](docs/experiment-protocol.md). The included synthetic pilot data tests the analyzer only; it is not efficacy evidence.

## Roadmap and implementation status

### M0 — Formal specification

Implemented.

- define states
- define actions
- define intervention prohibitions
- define deterministic state-to-action policy
- define structured model outputs

### M1 — Text prototype

Implemented and covered by contract tests.

- classify the current state
- generate one minimal intervention candidate
- apply deterministic intervention rules
- log decisions and outcomes

### M2 — Voice MVP

The streaming event boundary, pause handling, expiring/cancellable candidates, interruptible TTS, participant stop controls, controller, Apple Speech/AVFoundation microphone adapter, command-based provider boundary, replay test, and 10-minute load test are implemented. A live macOS session requires the user's microphone and speech-recognition permission.

- streaming speech recognition
- pause and phrase-boundary detection
- background language-model processing
- fixed low-latency audio cues
- continuous 10-minute sessions

### M3 — Post-session structure

Implemented as a deterministic, reviewable baseline. Extraction quality—not just schema behavior—still requires human evaluation.

Extract:

- claims
- premises
- evidence
- dependencies
- counterexamples
- unresolved questions
- claims requiring verification

### M4 — Knowledge input and grounding

Implemented for local UTF-8 text sources using lexical candidate retrieval. Matches include the source-file SHA-256 so reviewers can detect source-version drift, and are deliberately labeled as review candidates rather than proof.

- ingest articles, papers, and notes
- connect spoken claims to source passages
- distinguish supported, unsupported, and inferred claims

### M5 — Long-term personalization

Implemented for explicit intervention feedback, expiring operational-state counts, and explicit retention/forgetting of user-selected reasoning structures. Evidence text is represented only by a hash in the long-term profile; no transcript, unselected structure, or stable trait judgment is stored. Adaptation may suppress or fade interventions but cannot bypass safety controls. Long-term cognitive effects remain unvalidated.

- remember effective and ineffective interventions
- track recurring reasoning failures
- retain user-formed structures
- fade cues that appear to have become internalized

## Non-goals

CUELOOP is not initially intended to:

- diagnose cognitive conditions
- provide medical or psychological treatment
- prove general intelligence improvement
- replace domain experts
- maximize short-term answer accuracy
- continuously interrupt the user
- autonomously train an intervention policy with reinforcement learning
- act as a conventional conversational assistant

## Safety and privacy

Voice and reasoning logs may contain highly sensitive personal or professional information.

The initial implementation should therefore:

- default to local or private storage
- make recording status explicit
- allow deletion of all session data
- separate raw audio from derived structures
- avoid storing unsupported personality or ability judgments
- attach confidence, evidence, timestamps, and expiry conditions to inferred user-state data

## Epistemic status

The following remain unproven design hypotheses:

- that human cognition is globally more architecture-efficient than frontier models
- that large portions of model knowledge can be transferred into reusable human structures
- that minimal real-time intervention produces durable cognitive improvement
- that such improvement generalizes across domains
- that human-generated outputs can materially improve future model training

## Project status

Experimental and pre-alpha. The software path is runnable; scientific efficacy is not established.

The current objective is to run the documented human protocol, identify failure modes, and improve the policy without weakening suppression rules. The project does not currently claim scientifically validated cognitive enhancement.

## License

This repository is currently private and is not distributed under an open-source license.

All rights are reserved unless explicitly stated otherwise.
