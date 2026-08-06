# Intervention Atlas

Interventions are control actions, not content categories. Each intervention must have an intended cognitive effect, a delivery condition, and a prohibition condition.

## `SILENCE`

**Purpose:** preserve self-generation and avoid unnecessary interruption.

**Use when:**

- the user is productive
- confidence is low
- the user is overloaded
- the intervention deadline has passed

**Do not use as:** punishment, disapproval, or hidden evaluation.

## `BACKCHANNEL`

**Purpose:** signal that the user may continue without transferring control to the system.

**Examples:** short neutral acknowledgements.

**Do not use when:** it could be interpreted as factual agreement or reward.

## `QUESTION`

**Purpose:** trigger one reasoning operation.

**Examples:**

- “What does it depend on?”
- “Where does it fail?”
- “What is the comparison axis?”
- “What changes if that premise is false?”

**Constraints:** one operation per intervention; no embedded answer.

## `MIRROR`

**Purpose:** expose the current structure without evaluating it.

**Examples:**

- “One conclusion, three undefined criteria.”
- “The same premise is supporting both branches.”

**Constraints:** descriptive, concise, and reversible.

## `PERTURB`

**Purpose:** break fixation by introducing a counterexample, inversion, alternative perspective, or changed condition.

**Constraints:** use only after the current path is sufficiently externalized. Do not perturb productive generation prematurely.

## `CONSTRAIN`

**Purpose:** reduce the search space or force a representation change.

**Examples:**

- state the conclusion in one sentence
- separate observation from inference
- choose one evaluation criterion

**Risk:** excessive constraint can reduce exploration.

## `RETRIEVE`

**Purpose:** supply external knowledge only when the problem is knowledge absence rather than strategy failure.

**Constraints:** retrieval must be grounded, source-aware, and separated from the real-time cue path.

## `VERIFY`

**Purpose:** flag a claim for later grounding, logical inspection, causal inspection, or scope checking.

**Constraints:** verification is normally delayed during divergent thinking. The verifier identifies defects; it does not rewrite the answer by default.

## Global delivery rules

1. Default to silence.
2. Deliver at most one intervention at a time.
3. Cancel stale interventions.
4. Apply a cooldown after delivery.
5. Avoid repeating the same intervention form.
6. Do not introduce new knowledge during overload.
7. Do not treat backchannels as approval.
8. Log the evidence and confidence behind every delivered intervention.
