- After the region solver has chosen each block's resident set, a separate
  lane-assignment phase maps those residents to physical register lanes per bank
  with sticky inheritance: the root region gets a deterministic seed layout, each
  child copies kept locals into their parent's lanes, new locals fill the freed
  holes first, and a resident is remapped to a different lane only when
  fragmentation or a preserved-versus-volatile preference mismatch forces it
  (`compute_block_entry_cache_params`).

- Drops leave holes rather than compacting survivors down; a local resident on
  both sides of an edge keeps its lane and crosses without a register move.

- The middle IR stays set-based; lane choice lives at machine lowering, where the
  exact per-bank sizes are known, and a block improves its entry layout from its
  predecessors' exit layouts, leaving the boundary edge-repair pass fewer
  reconciliation moves to insert.

- The placement pass consumes two structure-derived signals computed by the
  middle-end instead of deriving them itself: the preserved-lane preference (a
  whole-function promotion of locals whose local-JIT-call cross count meets the
  backend threshold) and each entry row's Ensure-versus-Reserve requirement.

- Both signals are published after cleanup, constant folding, and sink
  planning, by scanning the final blocks and final control flow
  (`final_signals`); nothing derives them from pre-cleanup state.

- The entry-requirement outer vector remains intentionally empty from rewrite
  through structural cleanup. Rewrite retains its emitted first-touch
  classifications only in a flat temporary arena long enough for edge repair,
  then drops them; cleanup accepts the absent program side table, and
  `final_signals` remains the first publisher of machine-facing rows.

- The keep-across-call decision re-checks the physically assigned register's
  preservation class in addition to the preference bit; call classification
  reads the module facts threaded into lowering, and reference-typed locals
  are never kept across calls.

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

- 2026-07-12 (30aac662) measurement: middle-side lane assignment was built and
  rejected — with identical MachineIR op inventories and improved edge
  parallel-move counts (117 to 89 mismatched positions on the worst stream
  function), native code still inflated (+322 mov / +172 branch on that
  function; 8 of 9 corpus modules regressed, lua +7362, speedtest1 +30117)
  because register choice couples to transient allocation, scratch borrowing,
  and block-trace selection that only machine lowering sees (sourced).

- 2026-07-12 (30aac662) measurement: lifting the entry argument-lane footprint
  into the root region's lane floor pinned every cached resident above the
  footprint function-wide (a six-parameter call-free function's eleven caches
  shifted from lanes 0-10 to 6-16, +145 instructions from per-exit
  preserved-register restores); the argument footprint is an entry-block
  capacity concern, not a lane-placement floor (sourced).

- 2026-07-12 (30aac662) measurement: a HARD volatile-first rule in the
  rejected middle-side assigner (an unpreferred resident may never take a
  preserved-suffix lane while a volatile lane is free) measured net-negative
  (+58 on bzip2) — in a shuffle-dominated regime the region-entry move it
  forces costs more than the preserved-lane save/restore it avoids; the
  machine's shipped soft cost-based steering toward volatile lanes for
  unpreferred residents is a different, retained mechanism (sourced).

- 2026-07-12 (30aac662) pitfall: the cleanup pass that merges a single-
  predecessor goto successor appends the successor's ops but keeps the
  predecessor's entry-requirement row, so a slot the merged body write-firsts
  stays classified Ensure instead of Reserve (code).

- 2026-07-12 rationale: the merge staleness direction is provably Ensure-only
  (the merge preserves the predecessor's classification and never manufactures
  a Reserve), wasteful but never a dropped live value; deriving the published
  signals after cleanup removes the staleness class entirely (code).

- 2026-07-12 measurement: the binding-time preference matching at machine
  lowering is load-bearing, not subsumed by the entry-layout pass —
  neutralizing it (forcing the preference check true) regressed coremark by
  +289 and lua by +1844 native instructions, because the entry layout places
  only entry-resident caches while binding-time matching is what puts a
  call-crossing MID-BLOCK cache into a preserved lane (sourced).

- 2026-07-23 (7a8dc84b) measurement: rewrite allocated a provisional
  Ensure-versus-Reserve row for every ordinary, bridge, and repair block, then
  cleanup reindexed those rows even though edge repair never read them and
  final-signals unconditionally replaced them. Deferring first publication
  until final SSA removed those allocations and the stale side-table lifetime.
  FFmpeg's deterministic 14,290-function native index remained byte-identical
  and all 356 release tests passed. Two exact-parent ABBA startup pairs favored
  the candidate by 1.1% on mean point estimates, but their intervals overlapped
  and adjacent CPU profiles left `prepare_function` at 35.83% inclusive, so the
  structural simplification is retained without a confident wall-time claim
  ([[compiler.fact/startup-campaign-2026-07-22]]) (sourced).

- 2026-07-23 (b903d80b) measurement: emit-side repair counting and
  materialization repeatedly rescanned each successor's ops for the same
  first-touch answer, even though entry filtering had already classified every
  retained slot. Rewrite now appends those exact classifications to one flat
  span arena, consumes them for edge repair, and drops the arena before
  cleanup. `rewrite_function` fell from 999 to 938 serial FFmpeg samples
  (6.1% absolute); exact-parent ABBA startup point estimates averaged 0.65%
  faster. FFmpeg's deterministic native index remained byte-identical, all
  357 release tests passed, and fat-LTO text grew 500 bytes
  ([[compiler.fact/startup-campaign-2026-07-22]]) (sourced).

## Moves

- 2026-04-06 (366923b2) replaced [[compact-lane-assignment]]: compacting each
  block's entry cache lanes into a sequential prefix by global slot order
  renumbered a still-resident shared local to a different lane whenever an
  earlier-ordered local was added or dropped, causing avoidable cross-edge moves;
  assigning lanes top-down with sticky inheritance and leaving holes after drops
  keeps a shared local on the same lane across edges (code).

- 2026-07-12 (30aac662) replaced [[middle-region-lanes]]: with identical
  MachineIR op inventories and improved edge parallel-move counts the
  pure-middle lane assignment still inflated native code because register
  choice couples to transient allocation, scratch borrowing and trace
  selection that only the machine sees; lane placement returned below LIR
  while the structure-derived preference and requirement signals stayed in
  the middle (code)

- 2026-07-12 (30aac662) replaced [[machine-preference-dataflow]]: the
  preserved-lane preference and the entry Ensure-versus-Reserve requirement
  are pure functions of the final SSA and module facts, so the middle computes
  them once over the final program and byte-identical output proves
  equivalence; the machine's liveness dataflow and per-block requirement
  re-scan are deleted, and only physical placement still needs machine context
  (code)
