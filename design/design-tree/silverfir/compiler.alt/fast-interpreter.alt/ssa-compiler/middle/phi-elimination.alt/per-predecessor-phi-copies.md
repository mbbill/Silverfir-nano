- Phi nodes are eliminated by inserting parallel Copy instructions, one group
  per predecessor block, before that predecessor's terminator; copies are keyed
  by predecessor block alone and critical edges are not split.

## Facts

- 2025-10-19 (e7e31a28) rationale: this first-generation phi elimination inserts
  a plain sequential copy per phi source at each predecessor's exit (identity
  copies elided, copies batched and sorted by predecessor id for determinism);
  it does no critical-edge splitting and no parallel-copy / swap sequencing, so
  two phis whose sources and destinations overlap on one edge can be miscompiled
  by the straight-line copy order — the simple form chosen because VIR already
  carried explicit register allocation and the MVP only needed the common
  non-overlapping case (diff).

- 2025-10-25 (b2109ad2) pitfall: emitting phi-elimination copies in naive
  sequential order corrupts values when a VReg is both a destination and a later
  source (the second copy reads the already-overwritten value); the copies carry
  parallel semantics, so dst/src conflicts must be broken with temporaries rather
  than emitted as a plain list (diff).

## Moves

- 2025-10-23 (fbb7e707) replaced by [[phi-elimination]]: the per-predecessor
  representation keyed copies by predecessor block alone and so could not express
  which successor a copy belonged to, so a copy inserted before a multi-successor
  predecessor's terminator executed on every successor edge until critical edges
  were split with a landing-pad block that runs the copy only on its intended
  edge (diff).
