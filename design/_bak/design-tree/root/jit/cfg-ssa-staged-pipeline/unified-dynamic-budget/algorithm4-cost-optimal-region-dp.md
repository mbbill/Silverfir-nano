# ALGORITHM4 — cost-optimal residency via region-tree DP

Residency is a cost to minimize, not a constraint. A region-tree dynamic program
maximizes `benefit(R,L)` (weighted frame-access savings) minus `call_tax(R,L)`
minus boundary `mismatch_cost(R)`, subject to a per-region capacity constraint,
solved by Lagrangian relaxation: a per-local tree DP over the Callahan–Koblenz
region tree at fixed capacity prices, alternating with subgradient updates on the
per-region dual prices. A second lane-mapping phase assigns the chosen residents
to physical lanes with sticky inheritance from the parent, leaves holes after
drops (no compaction), and micro-repacks with register moves only when
fragmentation forces it. i64 on 32-bit consumes 2 lanes.

The region tree is read straight off the Wasm decode — one region per `loop`,
parent = enclosing loop or root — with no SCC discovery and no irreducibility
handling. ALGORITHM4 is the sole per-function residency planner, not the heavy
tier of an amortizing allocator. It is the fourth iteration of this planner and
is treated as replaceable.

## In practice

Must:
- Build the region tree directly from Wasm loop nesting (`{root} ∪ {one region
  per loop}`, `parent(R)` = enclosing loop or root); no SCC discovery.
- Optimize `Σ benefit(R,L)·x[R,L] − Σ call_tax(R,L)·x[R,L] − Σ mismatch_cost(R)`
  subject to `Σ_L units(L)·x[R,L] ≤ cap(R)` per region, in one unit system
  (weighted frame-op equivalents).
- Charge loop-boundary residency changes symmetrically on entry and exit
  (`(entry_freq + exit_freq)·trans_cost(L)`); charge the root only one-time entry
  materialization.
- Compute `cap(R) = dynamic_budget − headroom(R)` where `headroom(R)` is the peak
  live transient units over the region's owned blocks, from `OpPlan.before` /
  `OpPlan.after`, per bank.
- Solve with the per-local tree DP at fixed prices alternating with subgradient
  price updates (`PRICE_ITERS = 12`), then project to feasibility per
  region/bank.
- Emit `LocalEnsureCache` / `LocalDropCache` (and `LocalReserveCache` for
  write-first entries) only at edges where `Owner(pred) ≠ Owner(succ)`.
- Run lane mapping per bank: child inherits parent lanes for shared locals, fill
  holes with additions (width-2 / i64 pairs first), micro-repack only on
  fragmentation; resolve lane remaps with parallel register moves, not frame
  churn.
- Use explicit (non-recursive) stacks for the region/idom/CFG walks in
  `lower_cache_layout.rs` so deep CFGs do not overflow the native thread stack.

Must not:
- Treat whole-function stability or loop-specific overrides as hard-coded special
  cases; they must emerge from the cost model.
- Compact cache lanes down after a local is dropped, forcing shared locals to
  slide.
- Double-count add/drop membership cost in both the residency objective and the
  lane-mapping cost.
- Promise a region capacity beyond its exact transient headroom; over-pressure is
  handled by the rewriter's weakest-public-local eviction fallback, not by
  softening headroom.
