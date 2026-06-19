- The joint planner pre-computes, for every semantic op, a before/after transient
  EntryState (`OpPlan`) and a pre-op boundary contract — a Spill/Fill/DropCache
  PrepAction script plus the exact transient stack-height, spill-depth and
  live-type shape that must hold — and the rewriter consults the planner before
  each op, validates the supplied contract against its current block state, and
  materializes exactly the prescribed spills and fills.

## Moves

- 2026-04-13 (205c43cf) replaced by [[transient-prep]]: the per-op before/after
  plan and pre-op contract were materialized for the whole function only to be
  replayed by the rewriter; deciding fill/spill inline from local block state
  removes that per-op array and keeps only a compact stack-height/spill-depth
  entry per op for branch-target queries (code)
