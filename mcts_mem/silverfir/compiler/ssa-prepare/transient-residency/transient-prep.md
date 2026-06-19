- The rewriter decides each op's transient fill/spill inline from local block
  state, keeping only a compact per-op stack-height/spill-depth entry for
  branch-target queries rather than a materialized whole-function plan
  (`apply_inline_prefix`).

## Moves

- 2026-04-13 (205c43cf) replaced [[planner-owned-prep]]: the per-op before/after
  plan and pre-op contract were materialized for the whole function only to be
  replayed by the rewriter; deciding fill/spill inline from local block state
  removes that per-op array and keeps only a compact stack-height/spill-depth
  entry per op for branch-target queries (code)
