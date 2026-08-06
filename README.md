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

Most AI assistants optimize for producing high-quality answers immediately. That is useful, but it can move the reasoning process from the person to the model.

CUELOOP is based on a different premise:

> AI should sometimes improve the environment in which a human thinks, rather than perform the thinking for them.

The project exists to address four problems:

1. **Answer substitution**  
   Completed model outputs can replace the user's own generation process.

2. **Weak internalization**  
   A user may perform better while assisted without retaining the reasoning structure after assistance is removed.

3. **Poor updateability**  
   Memorized answers are difficult to revise consistently when a premise or counterexample changes. Structured reasoning makes downstream revision possible.

4. **Over-intervention**  
   Conventional assistants are optimized to respond. In a thinking session, silence or a two-word cue may be more useful than a full answer.

CUELOOP therefore optimizes for minimal, context-dependent intervention, later verification, and eventual cue reduction—not maximum response volume.

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

## Roadmap

### M0 — Formal specification

- define states
- define actions
- define intervention prohibitions
- define deterministic state-to-action policy
- define structured model outputs

### M1 — Text prototype

- classify the current state
- generate one minimal intervention candidate
- apply deterministic intervention rules
- log decisions and outcomes

### M2 — Voice MVP

- streaming speech recognition
- pause and phrase-boundary detection
- background language-model processing
- fixed low-latency audio cues
- continuous 10-minute sessions

### M3 — Post-session structure

Extract:

- claims
- premises
- evidence
- dependencies
- counterexamples
- unresolved questions
- claims requiring verification

### M4 — Knowledge input and grounding

- ingest articles, papers, and notes
- connect spoken claims to source passages
- distinguish supported, unsupported, and inferred claims

### M5 — Long-term personalization

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

## Project status

Experimental and pre-alpha.

The current objective is to build a theory-driven personal prototype and identify clear failure modes. The project does not currently claim scientifically validated cognitive enhancement.

## License

This repository is currently private and is not distributed under an open-source license.

All rights are reserved unless explicitly stated otherwise.
