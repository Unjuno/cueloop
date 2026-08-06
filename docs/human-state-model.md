# Human State Model

The state model is an operational control abstraction. It does not diagnose the user or claim direct access to internal cognitive states.

## States

### `PRODUCTIVE`

The user is extending, revising, or connecting ideas without obvious repetition or breakdown.

**Default action:** `SILENCE`

**Observable indicators:**

- coherent continuation
- new relations or examples
- low repetition
- no unresolved interruption request

### `RETRIEVAL_FAILURE`

The user appears to know what they want to express but cannot retrieve a word, example, or previously used concept.

**Default action:** `BACKCHANNEL`, then a minimal retrieval cue if the pause persists.

**Observable indicators:**

- fillers and self-repair
- unfinished references
- explicit statements such as “what was it called?”
- repeated attempts to retrieve the same item

### `STRATEGY_FAILURE`

The user has content but does not know which reasoning operation to apply next.

**Default action:** `QUESTION`

**Observable indicators:**

- explicit uncertainty about how to proceed
- several disconnected options without a selection criterion
- repeated restart of the same reasoning segment

### `FIXATION`

The user remains on one local reasoning path despite diminishing returns.

**Default action:** `PERTURB` or `MIRROR`

**Observable indicators:**

- repeated wording or premise
- multiple conclusions derived from the same unexamined assumption
- inability to produce an alternative framing

### `STRUCTURE_GAP`

The explanation contains a conclusion but lacks an explicit premise, relation, boundary, comparison axis, or update rule.

**Default action:** `QUESTION` or `MIRROR`

**Observable indicators:**

- undefined comparison terms
- causal conclusion without mechanism
- general claim without scope
- evidence list without a relation to the claim

### `OVERLOAD`

Additional input is likely to degrade performance.

**Default action:** `SILENCE`

**Observable indicators:**

- fragmented speech
- increasing interruption sensitivity
- rapid accumulation of unresolved branches
- explicit fatigue or confusion

## State-estimation rules

1. Prefer `PRODUCTIVE` when evidence is ambiguous.
2. Never infer a stable user trait from a single state estimate.
3. Every estimate must include confidence and observable evidence.
4. Low-confidence estimates must not trigger high-impact interventions.
5. State estimates expire as the conversation changes.
6. The controller may choose `SILENCE` regardless of the proposed state.

## Future states

The following are deliberately deferred until observable definitions are stronger:

- `KNOWLEDGE_GAP`
- `OVERGENERALIZATION`
- `EVIDENCE_GAP`
- `UPDATE_FAILURE`
- `MISREPRESENTATION`
