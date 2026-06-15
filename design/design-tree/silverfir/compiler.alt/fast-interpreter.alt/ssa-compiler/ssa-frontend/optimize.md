- After construction the SSA function runs a sequence of SSA-to-SSA cleanup
  passes before the middle stage, in order: instruction fusion ([[ssa-fusion]]),
  unreachable-block elimination, phi-predecessor cleanup, trivial-phi
  elimination, dead-phi elimination, and dead-code elimination.

- Trivial-phi elimination collapses a phi whose inputs reduce to a single
  unique value once self-references are dropped, substituting all uses with
  that value and iterating to collapse transitive chains.

- Phi-predecessor cleanup removes phi-node incoming entries whose source block
  is no longer an actual CFG predecessor before the function is handed to the
  middle stage.

- Dead-code elimination runs last, iterating to a fixpoint and removing every
  instruction whose results are all unused, except instructions kept for side
  effects.

## Facts

- 2025-11-25 (9de0feaa) rationale: the frontend can leave phi nodes with stale
  incoming entries (a terminator rewired away from a block, an unreachable block
  not yet removed), so phi predecessors are not automatically consistent with
  the CFG; the cleanup pass normalizes them, which lets the previously-disabled
  phi-predecessor consistency validation be re-enabled, scoped to reachable
  blocks only so stale entries in unreachable blocks do not trip it (diff).

- 2025-11-28 (d9fab688) rationale: SSA fusion rewrites only the root
  instruction and leaves the now-unused operand instructions in place; cleaning
  them up is delegated to a separate DCE pass ordered after all other
  optimizations, rather than each fusion pattern removing its own dead inputs —
  the DCE pass was introduced together with fusion to absorb exactly this
  cleanup (diff).

- 2025-11-28 (c1548240) pitfall: DCE's side-effect predicate must treat
  trapping operations as having side effects even though they produce a value —
  an initial version listed only stores/calls/global-set and would delete a
  load, integer div/rem, float-to-int trunc, ref-as-non-null, or ref-cast whose
  result was unused, but the spec suite verifies those traps still fire, so the
  predicate was widened to keep every potentially-trapping instruction (diff).
