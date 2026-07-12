- The planner publishes a tentative per-block entry cache set (the region
  solver's residency projected per block); the rewriter lowers with it, then
  finalizes the published entry by trimming carried-in locals the block never
  qualifies (`filter_block_entry_cached_slots`).

- The published exit set is re-derived after lowering by replaying the block's
  emitted cache ops under the finalized entry
  (`simulate_materialized_cache_exit`), and boundary-repair blocks are derived
  post-hoc by diffing each predecessor's exit against its successor's entry.

- The plan is advisory: rewrite observes lowered reality and reconciles,
  rather than the plan predicting it.

## Moves

- 2026-07-12 (2c2e010e) replaced by [[exact-plan]]: the planner's entry sets
  were intentionally tentative with rewrite observing the actual exit and
  finalizing post-hoc (filter, exit re-simulation, and post-hoc edge-repair
  derivation) — three reconciliation layers against an advisory plan; the
  exact-simulation walker computes the rows and repair actions from the same
  shared engine, deleting the reconciliation and making the plan the contract
  (code)
