- Module compilation runs in three whole-module phases: phase 1 decodes every
  function to a semantic program and stores them all in one vector; phase 2
  inlines small leaf callees across that full set, iterating to a fixed point
  that resolves transitive chains; phase 3 prepares (SSA-lowers) every function
  from the retained semantic set.

- The entire module's decoded semantic IR is held live simultaneously for the
  duration of the inline and prepare phases.

## Facts

- 2026-04-14 (fc7c2f74) rationale: the streaming migration was completed
  end-to-end — the batch path materialized every function's MachineIR for the
  whole module before any native emission, while streaming runs
  decode->prepare->lower->optimize->emit for one function at a time directly into
  the buffer, so no whole-module IR vector is ever held (diff).

## Moves

- 2026-04-11 (89d889fb) replaced by [[compiler]]: the batched pipeline decoded
  every function's SemanticProgram up front and held the whole module's semantic
  IR live in one Vec<Option<SemanticProgram>> across a separate fixed-point
  inlining phase and a separate prepare phase, so peak compile-time memory
  scaled with the total decoded size of the module; streaming each caller
  (decode, inline retained leaf seeds, lower immediately) keeps only the tiny
  retained inline-candidate set plus one in-flight caller live at a time,
  holding the whole module's semantic IR never in memory at once (diff)
