- A block boundary is a generic SSA contract: a successor block declares the
  live SSA params it needs, and each edge binds predecessor live-out to those
  params explicitly; there is no positional stack-order contract, and taken-branch
  payload travels through canonical operand slots rather than a positional live
  window (`SsaBlock`, `params`).

## Facts

- 2026-07-22 (b933a0ec) statement: typed loop-carried stack values remain
  explicit SSA/block parameters through cleanup instead of being published and
  reloaded through a frame slot at each backedge. Counter-param's loop changed
  from load/sub/store to `sub` + `cbnz`, reducing 0.749 to 0.254 ms; predecessor
  discovery is built once so the cleanup remains linear rather than rescanning
  the CFG per block (sourced).

- 2026-07-23 (fc6c058f) statement: empty-goto threading maintains one incoming
  edge index while composing bindings, requeues affected sources/targets, and
  tombstones every successfully threaded block before one final compaction.
  It does not rescan the CFG or renumber the parallel block vectors per removed
  block; cache-only repair blocks and guarded entry-block handling retain their
  existing semantics (sourced).

## Moves

- 2026-03-12 (455661a0) replaced [[positional-tos-window-boundary]]: a positional
  stack-order block contract forced edges to remap branch payload as live
  boundary SSA, reintroducing hidden stack policy that disagrees with the
  canonical slot-based branch layout; replacing it with successor-declared live
  params plus explicit edge bindings lets taken branch payload travel through
  canonical operand slots (published during frontend preparation, reloaded by the
  target's prepared prefix) and lets backend lowering reconcile bindings into
  real registers or moves without a positional contract (code).
