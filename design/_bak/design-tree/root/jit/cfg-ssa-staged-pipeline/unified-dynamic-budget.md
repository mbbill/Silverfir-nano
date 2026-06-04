# Unified dynamic register budget

A single unified dynamic bank per register class, shared between cached locals
and transient stack values, with a per-block joint planner deciding the split at
every block boundary. ARM64 GP dynamic = 22 registers, FP dynamic = 29. The
planner's invariant: at every program point and every block boundary, live
transient SSA values + resident cached locals ≤ total dynamic budget. Cached
locals are not SSA values but cross block boundaries as explicit boundary state.

The register file is partitioned `[fixed | gp_dynamic | fp_dynamic]`; fixed holds
the runtime context pointer, frame pointer, `mem0_base`, `mem0_size`. The dynamic
banks are ordered pools (volatile lanes first, then preserved) supplied by
`BackendConfig`; cached-local residency and transient ownership are tracked in
`BlockLowerContext` state by metadata, not by register number.

This choice is held at LOW confidence. The implementation, ALGORITHM4, is a
replaceable middle-end pass, not an invariant.

This node opens one sub-problem: which locals stay resident, in which regions —
`local-cache-residency-planning/`.

## In practice

Must:
- Share one dynamic bank per register class between cached locals and
  transients; let the per-block joint planner re-split it at every block
  boundary.
- Hold the budget invariant: at every program point, live transient units +
  resident cached-local units ≤ dynamic budget for that class.
- Carry cached locals across block boundaries as explicit SSA boundary state
  (`LocalEnsureCache` / `LocalReserveCache` / `LocalDropCache`), not as implicit
  backend register state.
- Keep `BlockLowerContext` ownership metadata authoritative over register
  number; the bank ordering (volatile-then-preserved) is an ABI preference only.
- Reserve the fixed bank (`ctx`, `fp`, `mem0_base`, `mem0_size`) outside the
  dynamic budget; these are never transient or cache lanes.

Must not:
- Let the backend run a full register allocator; a backend allocation failure is
  by definition a middle-layer planning bug.
- Statically fix a per-function cached/transient partition (that is the
  abandoned `static-two-bank-split`).
- Treat ALGORITHM4 as a fixed invariant; it is the fourth iteration of the
  budget-splitting algorithm and is replaceable.

## Ground rules — local-cache-residency-planning
Must:
- Residency per region must respect `capacity = dynamic budget − headroom`,
  where headroom is the peak live **transient** pressure across the region's
  blocks — computed from transients only, never from cache occupancy (no
  double-counting).
- All blocks in a region share one public cache set; cache transitions
  (`Ensure`/`Reserve`/`Drop`) appear only at region boundaries, with edge-repair
  blocks reconciling mismatched edges.
- An `i64` local on a 32-bit GP target must be charged as two units everywhere —
  capacity, transition cost, and lane width — consistently.

Must not:
- Must not select cache sets per block (boundary churn) or mutate the public
  set inside a straight-line region.
- Must not promise capacity that does not exist: pathological transient pressure
  shrinks a region's cache capacity; it is never absorbed by overcommitting.
