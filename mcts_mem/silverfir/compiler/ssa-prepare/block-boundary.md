- A block boundary is a generic SSA contract: a successor block declares the
  live SSA params it needs, and each edge binds predecessor live-out to those
  params explicitly; there is no positional stack-order contract, and taken-branch
  payload travels through canonical operand slots rather than a positional live
  window (`SsaBlock`, `params`).

## Moves

- 2026-03-12 (455661a0) replaced [[positional-tos-window-boundary]]: a positional
  stack-order block contract forced edges to remap branch payload as live
  boundary SSA, reintroducing hidden stack policy that disagrees with the
  canonical slot-based branch layout; replacing it with successor-declared live
  params plus explicit edge bindings lets taken branch payload travel through
  canonical operand slots (published during frontend preparation, reloaded by the
  target's prepared prefix) and lets backend lowering reconcile bindings into
  real registers or moves without a positional contract (code).
