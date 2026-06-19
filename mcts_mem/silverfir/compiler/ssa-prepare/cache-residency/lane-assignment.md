- After the region solver has chosen each block's resident set, a separate
  lane-assignment phase maps those residents to physical register lanes per bank
  with sticky inheritance: the root region gets a deterministic seed layout, each
  child copies kept locals into their parent's lanes, new locals fill the freed
  holes first, and a resident is remapped to a different lane only when
  fragmentation forces it (`compute_block_entry_cache_params`).

- Drops leave holes rather than compacting survivors down; a local resident on
  both sides of an edge keeps its lane and crosses without a register move.

- The middle IR stays set-based; lane choice lives at machine lowering, where the
  exact per-bank sizes are known, and a block improves its entry layout from its
  predecessors' exit layouts and looks through successor live-in cache state,
  leaving the boundary edge-repair pass fewer reconciliation moves to insert.

## Facts

- 2026-04-06 (366923b2) statement: per the deleted middle/LANE_MAPPING.md (added
  a50a44d4, deleted 38809e62), the lane-assignment cost (one register move per
  remapped shared-local unit) is kept out of ALGORITHM4's resident-set edge_cost
  on purpose in v1 — the two cost models are intentionally separate, so ALGORITHM4
  may approve a transition slightly cheaper than reality when a shared local is
  remapped, but sticky inheritance and no-compaction make such remaps rare; if
  profiling later shows material remap cost, the intended fix is to add an
  estimated remap penalty back into ALGORITHM4's edge_cost — a tuning change, not
  an architectural split (sourced).

- 2026-04-06 (366923b2) statement: per the same deleted middle/LANE_MAPPING.md,
  the v1 lane-remap design recommended subtree stickiness (keep locals resident
  through more of the subtree, weighted by descendant edge frequency) as the
  mandatory secondary objective in the fragmentation fallback, to prevent
  cascading remap damage down a region's descendants; the shipped exact GP search
  did not take it — it minimizes only unit move-cost (one register-move per
  remapped shared-local lane unit, no edge-frequency weighting) and breaks ties by
  lowest lane index. A future revisit of remap-cost minimization can reconsider
  stickiness; it was a deliberate non-inclusion, not an oversight (sourced).

- 2026-04-29 (ae6fcc9c) pitfall: the cache-layout planner walked the CFG and
  dominator tree with native recursion in three places (reverse-postorder DFS,
  idom-reachability marking, dominator-tree layout assignment); all three were
  converted to explicit-stack iterative loops with a regression pin compiling a
  12,000-block linear CFG and a 12,000-deep linear idom tree. Lesson: any
  compiler-internal graph walk whose depth scales with user-controlled CFG/idom
  depth must be iterative, never recursive — the worst case is input-controlled,
  not bounded by source size (code).

- 2026-04-29 (ae6fcc9c) pitfall: the original recursion overflowed the native
  thread stack on a synthetic 200 KB function (single_fn_200k.wasm) that lowered
  to 8,701 SSA blocks, crashing on the small Windows native thread stack —
  concrete evidence that a real on-device module, not just an adversarial input,
  can drive CFG/idom depth past the recursion limit (sourced).

- 2026-05-16 (8b6dd066) rationale: fixed-local-only `call_indirect` is treated as
  a local JIT call for cache-preserve planning, so its caches can be carried
  across the call the same way as direct local calls (code).

## Moves

- 2026-04-06 (366923b2) replaced [[compact-lane-assignment]]: compacting each
  block's entry cache lanes into a sequential prefix by global slot order
  renumbered a still-resident shared local to a different lane whenever an
  earlier-ordered local was added or dropped, causing avoidable cross-edge moves;
  assigning lanes top-down with sticky inheritance and leaving holes after drops
  keeps a shared local on the same lane across edges (code).
