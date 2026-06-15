- Cache residency is decided per block: each block gets its own resident set
  seeded from block-local value ranking, chosen as a tentative entry boundary and
  finalized after one lowering pass (`choose_tentative_block_entry`).

- The per-block objective minimizes only that block's frame-access cost;
  transition cost at edges between disagreeing blocks is not part of the
  objective.

- Disagreements between a block's chosen entry set and its predecessors are
  reconciled afterward by edge-repair blocks emitting ensure/drop ops at the
  boundary; live transient SSA values and resident cached locals share one
  per-bank dynamic budget.

## Facts

- 2026-04-06 (a50a44d4) measurement: across 9 WASI benchmarks (ARM64) the
  per-block planner left the cache set completely unstable, produced 20-31%
  boundary-only blocks existing solely to run ensure/drop ops, and grew code size
  1.3-1.7x, making it strictly worse than the old fixed-bank system on every
  benchmark — full tables in [[per-block-residency.fact/per-block-churn]] (diff).

## Moves

- 2026-04-03 (8aab7e14) replaced [[multi-pass-middle-cache]]: the old middle
  owned cache and spill decisions across several passes over a whole-function
  local-cache preference table and then reconstructed block live-ins from
  already-lowered cache ops; whole-function hotness is misleading because cache
  usefulness is region/CFG-local, so the middle is rebuilt to choose cache
  residency and transient spills jointly in one pass against a single shared
  dynamic-bank budget, with each block's entry boundary known up front from the
  plan rather than reconstructed afterward (diff).

- 2026-04-06 (0b5d2ea0) replaced by [[cache-residency]]: choosing a resident set
  independently per block minimized per-block access cost but ignored transition
  cost at edges, so the cache set churned at almost every block boundary; solving
  residency per region on the Wasm loop tree by minimizing one weighted cost
  (benefit minus call tax minus per-boundary transition cost subject to
  per-region capacity) makes whole-function stability and loop-specific overrides
  emerge instead of being fought (diff).
