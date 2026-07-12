- One shared engine owns the typed operand-window state evolution: stack
  height, spill depth, the per-slot type stack, and an alias tag per live
  entry, with a single implementation of fill, spill, capacity clamping,
  eviction, and the call-boundary transitions (`Window`); the rewriter's emit
  pass and the exact-plan walker drive the same engine, while the planner's
  lightweight measure pass shares only the structural table and the pure
  window arithmetic.

- A driver interface separates emission from state evolution: drivers own only
  op emission and SSA value identity, notified through hooks; the emit driver
  keeps a parallel value window in lockstep, a measure driver evolves state
  without emitting, and value identities never live in the engine.

- The per-op structural discipline is a single table shared by every consumer;
  where the planner and the rewriter deliberately interpret one op differently
  (calls, single-result returns, else), the divergence is a named table
  variant with its rationale recorded, never two parallel matches.

- The planner's measure pass stays cache-free with the capacity clamp off; the
  region solver's residency capacity is the budget minus the measured peak.

## Facts

- 2026-07-12 rationale: the measure pass's cache-free, clamp-off reading is a
  correctness invariant, not slack — its peak is a true upper bound of the
  rewriter's live pressure and residency capacity is budget minus that peak, so
  tightening the measure pass over-admits residents the machine cannot bind
  (the 30779e5d pitfall class) (code).

- 2026-07-12 rationale: the single-result-return table variant is deliberately
  asymmetric (planner spill-all, rewriter fill-one-spill-rest) and must not be
  normalized: the planner marks unreachable after every return, overwriting
  spill depth, so its spill-all can never reach a captured block-entry state
  (code).

## Moves

- 2026-07-12 (35a439c7) replaced [[twin-simulations]]: the planner's
  lightweight simulation and the rewriter's live-window discipline were two
  implementations of one stack policy that had to be manually mirrored (the
  planner a conservative upper bound of the rewriter); extracting one engine
  with measure and emit drivers makes divergence structurally impossible and
  let the later exact-plan walker reuse the same transitions (code)
