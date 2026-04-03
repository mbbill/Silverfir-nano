# Joint Transient + Local Cache Plan — Refined

## Executive Summary

The current implementation is **much closer to the target architecture than the old fixed-cache design**, but it still misses the planned performance because the global cache-state problem is not actually solved yet.

What is already in good shape:

- SSA-IR already uses explicit local/cache operations instead of logical local versioning.
- Straight-line preparation already reasons about **joint transient + cached-local pressure** against the **combined dynamic bank**.
- The machine layer already behaves like a mostly unified dynamic bank for ordinary allocation: transient values and cached locals both allocate from the same dynamic GP/FP pools, with only preference ordering and a tiny reserved helper escape hatch remaining.

What is still fundamentally wrong:

1. **CFG boundary planning is still conservative and incomplete.**
   The late pass computes carried cache state by predecessor intersection and inserts only **drop-only** repair blocks. That means it cannot establish a profitable canonical entry state for a successor block; it can only preserve a subset of what every predecessor already has.
2. **Loop headers and hot joins are therefore pessimized.**
   A loop header with a cold preheader and a hot backedge loses cache carry precisely where carry matters most.
3. **The planner still caps the cacheable-local universe by the old fixed cache budget.**
   This prevents disjoint regions from caching different locals even when the joint dynamic bank has room.
4. **The IR lacks a state-only op for “make this local resident in cache”.**
   Real edge repair needs to establish cache residency without manufacturing an SSA value.
5. **Dirty/clean cache state is lost at edges.**
   Carried cache entries are currently treated as dirty on block entry, which forces unnecessary writebacks later.
6. **Stale call-side metadata is still present.**
   `skip_reload` plumbing survives in the IR and machine interfaces even though the current pipeline clears or ignores it.
7. **Late optimization ordering is not ideal.**
   Sink planning currently targets only slot stores and runs before the late cache-state rewrite, so it misses profitable `LocalSetCache` sinks and sink opportunities created by demotion.

The main conclusion is:

> The remaining work is not “improve machine register choice”. The main missing piece is a real **global cache-state planner in `middle/`** with explicit repair and an SSA contract that can express state repair cleanly.

---

## Design Goals

- Share one dynamic GP bank and one dynamic FP bank between:
  - transient SSA values, and
  - cached locals.
- Keep transient values semantically distinct from locals.
- Keep cache eviction policy in `middle/`.
- Keep physical register choice in `machine/`.
- Allow the transient/cache ratio to vary at arbitrary program points.
- Preserve exact Wasm stack shape while the planner that needs it still has it.
- Make boundary behavior explicit and testable.

## Non-Goals

- Do not make locals into SSA variables.
- Do not expose physical register ids or cache-position ids in SSA-IR.
- Do not let `machine/` infer edge carry or eviction policy.
- Do not keep stale metadata-only policy channels once the explicit plan exists.

---

## What The Current Code Actually Does

### Good news: the machine layer is already mostly on the right track

The machine layer is **not** the main bottleneck.

- `src/vm/machine/lower_regalloc.rs`
  - ordinary transient allocation uses `first_free_transient()` over the full dynamic bank, with preference order rather than a hard semantic split;
  - cached-local binding also scans the full dynamic bank.
- This means the machine already behaves like a **unified dynamic bank with preference ordering**, which is close to the target design.

That is important: the performance gap is not primarily coming from physical register assignment.

### The real bottlenecks are in the middle-end

#### 1. `spill_plan.rs` still limits which locals may ever become cached

`src/vm/middle/spill_plan.rs`

- `build_cache_plan()` truncates the candidate set to the old `gp_local_cache_budget` / `fp_local_cache_budget`.
- This is still a fixed-cache-era cap on the **cacheable universe**, even though the actual pressure accounting now uses the combined dynamic budget.

Why this is bad:

- it prevents different disjoint regions from caching different locals;
- it blocks profitable caching when transient pressure is low;
- it reintroduces a hidden fixed split at the policy level.

#### 2. `resource_plan.rs` does not choose a true canonical entry state

`src/vm/middle/resource_plan.rs`

- `compute_block_entry_cached_slots()` keeps only slots that are wanted by the block **and** already present in **all** predecessor exits;
- `insert_boundary_repair_blocks()` only inserts `LocalDropCache` operations.

That is not a true canonical-entry design.

It means:

- the successor does **not** own its entry state;
- the entry state is merely the common subset of predecessor exits;
- repair blocks cannot add missing cached locals;
- joins and loops are systematically underpowered.

#### 3. Repair blocks are not real repair blocks

Current repair blocks can only drop caches.

They cannot:

- load a local into cache for the successor,
- clean a dirty carried cache while preserving residency,
- re-establish a profitable header state on cold edges.

This is the single clearest reason the current implementation does not reach the planned performance.

#### 4. The current IR cannot express state-only cache materialization cleanly

Current explicit local ops are value-oriented:

- `LocalGetSlot`
- `LocalSetSlot`
- `LocalGetCache`
- `LocalSetCache`
- `LocalDropCache`

But edge repair needs a state-oriented operation:

- “make `slot` resident in cache now, without producing an SSA value”.

Without that, late repair either becomes awkward or falls back to hidden machine policy.

#### 5. Edge carry loses dirty/clean precision

`src/vm/machine/lower_context.rs`

- carried entry caches are currently bound live and marked dirty by default;
- `_initial_cache_dirty` exists as a hook but is unused.

This is safe, but it causes unnecessary stores later when a carried cache was actually clean.

#### 6. Stale call metadata is still hanging around

- `src/vm/middle/ssa_ir/ir.rs` still exposes `skip_reload` in `SsaCallOp`;
- `src/vm/middle/resource_plan.rs` clears it;
- `src/vm/machine/lower_inst.rs` ignores it in `begin_continuation_block_selective()`.

This is dead policy plumbing. It should be removed or replaced by explicit repair.

#### 7. Sink planning is leaving performance on the table

`src/vm/middle/sink_plan.rs`

- sink planning currently matches only `LocalSetSlot` stores;
- it runs before `resource_plan.rs`, which may later demote or preserve cache stores differently.

This misses two profitable cases:

- direct sink into `LocalSetCache` / cache registers;
- sink opportunities that appear only after late demotion.

---

## Refined Architectural Split

The right architecture is:

1. **semantic stack preparation + straight-line joint planning**
2. **SSA lowering to explicit local/cache vocabulary**
3. **global cache-state planning over the SSA CFG**
4. **late non-destructive cleanup / fusion**
5. **machine lowering as plan realization only**

### `middle/` owns

- which locals are worth caching;
- when a local access is slot-based vs cache-based;
- which locals are resident at each block entry/exit;
- which cached locals are dirty vs clean when that matters to later behavior;
- when an edge carries cache state directly;
- when an edge uses a repair block;
- when a local must be dropped or re-materialized.

### `machine/` owns

- physical register binding;
- parallel copies for block params and carried cache params;
- concrete loads/stores/moves;
- alias breakage when cache registers are overwritten;
- narrow explicit helper scratch after call/transient publication boundaries.

### `machine/` must not own

- cache victim choice;
- edge carry policy;
- call continuation cache policy;
- hidden “best effort” cache inference.

---

## Refined SSA-IR Contract

### Required final vocabulary

Keep:

- `Value { ... }`
- `Fill { slot, dst }`
- `Spill { slot, src }`
- `LocalGetSlot { slot, dst }`
- `LocalSetSlot { slot, src }`
- `LocalGetCache { slot, dst }`
- `LocalSetCache { slot, src }`
- `LocalDropCache { slot }`
- `Call(...)`

### Required new operation

Add:

- `LocalEnsureCache { slot }`

Semantics:

- after the op, `slot` is resident in the cache;
- if already resident, it is a no-op;
- if not resident, lower from the canonical slot into a newly bound cache register(s);
- it produces **no SSA value**;
- a newly materialized cache entry is **clean**.

This op is required for:

- edge repair,
- loop-header re-materialization on cold entries,
- explicit continuation repair after boundaries if desired,
- keeping boundary policy out of `machine/`.

### Recommended follow-up operation

If exact dirty-state canonicalization is implemented, also add:

- `LocalWritebackCache { slot }`

Semantics:

- `slot` must already be resident;
- if dirty, store back to the canonical slot;
- keep it resident;
- mark it clean.

This is not strictly required for the first functional rewrite, but it is the cleanest way to avoid forced drop+reload sequences when a clean canonical entry is preferred.

### Remove from final SSA-IR

Remove:

- `skip_reload` from `SsaCallOp`
- machine-facing cache-ranking metadata (`SsaLocalCachePrefs`) once the middle-end no longer needs to expose it
- `CachedLocalInfo` from the backend handoff unless a proven backend consumer still exists

### Replace stale machine-facing ranking with explicit type data

If `machine/` still needs slot types for explicit cached locals, expose a compact final side table such as:

- `local_slot_types` or equivalent slot->type mapping

Do **not** keep ranking / hotness metadata in final SSA-IR just to recover types.

---

## Straight-Line Planning: What Should Stay And What Should Change

### Keep

`src/vm/middle/spill_plan.rs` should remain the place that still has exact Wasm stack shape and therefore decides:

- transient publication (`Fill` / `Spill`),
- straight-line local access flavor (`Slot` vs `Cache`),
- local pressure trade-offs within a block/region.

### Change

#### Remove the fixed cache-candidate cap

Do **not** bound the cacheable-local universe by `gp_local_cache_budget` / `fp_local_cache_budget`.

Instead:

- rank all locals by usefulness;
- optionally soft-cap the candidate universe by a compile-time heuristic window, e.g. top `k * dynamic_budget` per bank;
- keep simultaneous residency constrained only by the **combined dynamic bank**.

This preserves performance without exploding compile-time.

#### Keep access planning local, but treat it as provisional

The straight-line planner should choose cache-vs-slot access flavor provisionally, subject to later CFG-state repair.

It should not pretend to have solved the global edge-state problem.

---

## Global Cache-State Planning: The Missing Piece

Replace the current `resource_plan.rs` with a real CFG-aware planner.

### State model

Per bank, at minimum track:

- resident cached slots
- optional dirty/clean state per resident slot
- live transient pressure at each program point

A practical state shape is:

- `CacheState = { resident_slots ordered canonically, maybe_dirty_bits }`

### What the planner must compute

For each block:

- canonical entry cache state
- canonical exit cache state(s) or at least simulated exit summary
- explicit repair required on each incoming edge

### Key rule

A block owns **one canonical entry state**.

Predecessors do **not** choose arbitrary successor cache state.

If an incoming edge does not match the successor entry state, `middle/` inserts a repair block.

### Crucial correction to the current implementation

The canonical entry state must **not** be computed as:

> desired slots intersected with all predecessor exits

That rule destroys profitable carry at loop headers and hot joins.

Instead:

- choose the entry state for the block itself based on block-local profitability and CFG context;
- repair incompatible incoming edges explicitly.

### Consequences

This immediately fixes the biggest structural performance failures:

- hot loop backedges can carry cached locals even when the cold preheader cannot;
- hot predecessors no longer lose cache carry because a cold predecessor disagrees;
- compatible edges become truly cheap;
- incompatible edges become explicitly visible and optimizable.

---

## How Canonical Entry States Should Be Chosen

The block entry state should be selected by profitability, not by predecessor intersection.

At minimum, the scoring model should consider:

- local access frequency inside the block;
- next-use distance of each cached local;
- bank-unit cost of the local (`i64` cost on 32-bit matters);
- whether the local is first used near block entry;
- whether the local can be dropped before later transient peaks;
- loop/header hotness;
- edge hotness when available;
- dirty drop cost vs clean drop cost.

### Important correction to current block-level heuristics

Do **not** reject an entry-cached local merely because it would not fit against the block’s global peak live count.

That is too conservative.

A local that is heavily used near block entry may still be profitable if the planner can:

- carry it into the block,
- use it immediately,
- drop it before the later transient peak.

So the planner must simulate cache lifetime inside the block, not only compare entry residency against whole-block peak pressure.

---

## Edge Repair Requirements

Repair blocks must support both directions of repair:

1. **drop extras**
   - predecessor has a cached local that successor entry does not want
   - use `LocalDropCache`

2. **materialize missing residency**
   - successor entry wants a cached local that predecessor does not carry
   - use `LocalEnsureCache`

3. **optional dirty canonicalization**
   - if exact clean/dirty entry state is chosen
   - use `LocalWritebackCache` when needed

### Repair block shape

Repair blocks should:

- keep the same SSA value params as the original successor edge;
- contain only explicit cache-state repair plus any trivial copy cleanup;
- end in a single `Goto` to the real successor.

### This is what “repair in middle” really means

It is not enough to insert drop-only bridge blocks.

True repair means the edge can both:

- shed state, and
- re-establish state.

---

## Call Boundaries

### Current behavior

The current implementation effectively flushes cache state at calls and rebuilds it later on demand.

That is acceptable as a conservative baseline.

### Refined rule

Treat a call as a boundary with an explicit post-call cache state.

For the first correct implementation:

- internal and external calls may conservatively reset cache residency to empty;
- any profitable re-materialization after the call is expressed explicitly via normal cache ops or `LocalEnsureCache`.

### Remove stale call metadata

Do **not** keep `skip_reload` metadata.

It is inferior to explicit repair because it:

- hides policy from SSA-IR,
- complicates call lowering interfaces,
- is already partly dead in the current code.

### Later optimization (optional)

After the full explicit boundary planner is working, call boundaries may be refined further for:

- preserved callee-saved cache lanes across helper/runtime calls,
- selective clean carry,
- cost-based eager continuation repair.

But that must still be represented explicitly, not via stale hidden metadata.

---

## Dirty-State Handling

Dirty-state precision matters for performance.

### Current problem

A carried cache entry becomes dirty by default at block entry.

That causes unnecessary stores later.

### Required change

Thread canonical entry dirty-state into machine lowering.

At minimum:

- add final entry-state dirty metadata parallel to block-entry cached slots;
- use the existing `_initial_cache_dirty` hook in `BlockLowerContext::new()` instead of defaulting carried entries to dirty.

### Preferred long-term model

Make dirty-state part of the canonical cache state selected by the middle-end.

Then the planner can choose between:

- carrying a slot dirty,
- repairing it to clean,
- dropping it entirely.

---

## Optimization Pass Ordering

The pass ordering should become:

1. semantic preparation / straight-line joint planning
2. SSA lowering
3. non-stack-shape-destroying CFG cleanup (`thread_jumps` etc.)
4. global cache-state planning (`resource_plan` rewritten)
5. late sink / fusion pass
6. constant folding and other safe local cleanups
7. validation

### Late sink / fusion corrections

`sink_plan.rs` should target:

- `LocalSetSlot`
- `LocalSetCache`

And a late sink/fusion pass should run **after** the cache-state rewrite so it can see final local-store flavor.

That pass does not need to disturb stack-shape planning because it runs after all stack-shape-sensitive decisions are done.

---

## File-Level Change List

### `src/vm/middle/ssa_ir/ir.rs`

Required:

- add `LocalEnsureCache { slot }`
- optionally add `LocalWritebackCache { slot }`
- remove `skip_reload` from `SsaCallOp`
- remove final machine-facing ranking metadata once replaced
- add final slot-type side table if needed
- add canonical entry dirty-state side table if implemented

### `src/vm/middle/ssa_ir/validate.rs`

Required:

- validate the new cache-state ops
- validate slot-type coverage for them
- validate any new entry dirty-state sidecar

### `src/vm/middle/spill_plan.rs`

Required:

- stop truncating cache candidates to the old fixed cache budget
- keep full ranking as planner-private input
- continue straight-line capacity planning against the combined dynamic budget

### `src/vm/middle/resource_plan.rs`

Required full rewrite:

- replace predecessor-intersection entry selection with true canonical block-owned entry states
- compute to convergence with a worklist/fixed-point, not a one-or-two-pass heuristic
- model block-local cache lifetime, not only whole-block peak pressure
- insert real repair blocks using `LocalEnsureCache` and `LocalDropCache`
- optionally handle dirty canonicalization
- remove `clear_skip_reload()` and all related dead policy cleanup

### `src/vm/middle/lower_ops.rs`

Adjust:

- do not hardcode `local.tee` to slot-only semantics in the plan;
- lower it as store + reload using the planner-selected flavor, unless a later fused form is introduced.

### `src/vm/middle/sink_plan.rs`

Required:

- support sinks into `LocalSetCache`
- stay register-agnostic; just annotate the slot
- ensure the final sink/fusion stage runs after the cache-state rewrite

### `src/vm/middle/local_cache.rs`

Cleanup:

- remove stale comments and dead call-reload design remnants
- keep only ranking / analysis that is still genuinely consumed by the planner

### `src/vm/machine/lower_inst.rs`

Required:

- lower `LocalEnsureCache`
- optionally lower `LocalWritebackCache`
- remove unused selective continuation call plumbing if not explicit anymore

### `src/vm/machine/lower_context.rs`

Required:

- honor initial carried-cache dirty-state instead of defaulting all entry carries to dirty
- keep cache binding/allocation policy purely mechanical
- continue allocating cached locals and transient values over the unified dynamic bank with preference ordering

### `src/vm/machine/lower_module.rs`

Required:

- remove dead `skip_reload` threading
- thread final canonical entry dirty-state if added
- continue exposing hidden cache edge args from `block_entry_cached_slots` (or its replacement) only as realization of the middle-end plan

### `src/vm/machine/lower_call.rs`

Required:

- remove ignored `skip_reload` interface
- keep calls as explicit cache-state boundaries

### `src/vm/backend.rs` and backend ABI docs

Recommended:

- update comments so `gp_local_cache_budget` / `fp_local_cache_budget` are no longer described as semantic cache-planner caps
- describe them as physical preference / layout inputs, while the middle-end budgets against total dynamic units
- consider renaming in a follow-up if code churn is acceptable

---

## Recommended Implementation Sequence

### Phase 1 — Clean the contract

- add `LocalEnsureCache`
- remove `skip_reload`
- wire slot-type data needed by final SSA-IR consumers
- wire initial carried-cache dirty-state if you keep dirty precision in phase 1

### Phase 2 — Rewrite `resource_plan.rs`

- choose real canonical entry states
- insert real repair blocks
- stop using predecessor intersection as the entry-state rule
- compute to convergence

### Phase 3 — Remove the fixed candidate cap

- rank all locals
- optionally soft-cap only for compile-time reasons
- keep simultaneous residency bounded only by dynamic capacity

### Phase 4 — Restore missed optimizations

- sink into `LocalSetCache`
- rerun a late sink/fusion pass after cache-state planning
- add optional dirty canonicalization / `LocalWritebackCache`

### Phase 5 — Cleanup

- delete stale metadata and comments
- simplify machine call interfaces
- remove machine dependence on cache-ranking sidecars

---

## Test Plan

Add focused tests for the following cases.

### Boundary / CFG tests

- loop header with cold preheader + hot backedge
- hot/cold join where only the hot predecessor should carry cache and the cold edge should repair
- successor entry wants a cached local that only some predecessors currently carry
- repair block that must both drop one slot and materialize another

### Dirty-state tests

- clean cache carried across edge and later dropped without redundant store
- dirty cache carried across edge and later dropped with required store
- canonical clean entry chosen via repair (if implemented)

### Call tests

- call followed by overwrite-before-read of a cached local (no reload needed, no stale metadata path)
- call followed by multiple cached uses where eager continuation repair is profitable (if implemented)
- internal/local call vs external/runtime call behavior kept explicit and correct

### Pressure tests

- block with early local reuse and late transient peak: entry caching should still be allowed if the local can be dropped before the peak
- 32-bit GP target with `i64` cached locals and mixed transient pressure

### Optimization tests

- sink into `LocalSetCache`
- sink opportunities that appear only after late cache demotion

---

## Success Criteria

The rewrite is successful when all of the following are true:

- The final handoff no longer depends on stale hidden cache policy metadata.
- `middle/` owns both eviction and boundary-state policy.
- `machine/` only realizes that policy.
- Block entry cache states are chosen by the block, not by predecessor intersection.
- Repair blocks can both drop and materialize cache state.
- Loop headers and hot joins preserve profitable carry.
- The planner no longer caps the cacheable-local universe by the old fixed cache budget.
- Clean/dirty carry is not blindly collapsed to “dirty”.
- `skip_reload` is gone.
- Late sink/fusion works for both slot and cache stores.
- Performance-sensitive cases improve specifically at loop headers, hot joins, and cache-heavy regions separated by cold edges.

---

## Bottom Line

The current code already completed the easy half of the migration:

- explicit cache ops,
- joint straight-line pressure accounting,
- mostly unified machine dynamic banks.

The missing half is the hard one:

- a real global cache-state planner,
- a state-oriented cache-materialization op,
- true edge repair,
- removal of stale hidden metadata.

That is the work that will unlock the performance the original design intended.
