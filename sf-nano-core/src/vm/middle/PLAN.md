# Middle Rewrite Plan

## Goal

Rewrite `middle/` from first principles so it does one job cleanly:

- take Wasm semantic IR
- emit explicit SSA-IR for `machine/`
- at the same time choose transient spills/fills and local-cache usage correctly

Correctness comes first. Optimization passes can be added later.

## Non-goals

- no backward-compatibility architecture
- no whole-function cache hotness table
- no late pass that reconstructs boundary state from emitted cache ops
- no duplicated ownership of boundary decisions

## Core model

The key idea is:

- transient SSA values and cached locals share one dynamic-bank budget
- cached locals are not SSA values
- but cached locals do cross block boundaries as explicit boundary state

For each bank, the invariant is:

`live transient SSA values + resident cached locals <= total dynamic budget`

This must hold at every program point and every block boundary.

Boundary state has two parts:

- `cached_locals`: resident local slots, in deterministic order
- `stack_values`: live transient SSA values, in stack order

Transient stack shape is constrained by Wasm semantics. The main policy choices
are:

- which values deserve residency
- when residency must be materialized
- which values to evict under pressure

## High-level pipeline

1. Build an explicit CFG from Wasm semantic IR.
2. Lower that CFG to slot-only SSA.
3. Let `joint_plan/` choose:
   - canonical predecessor
   - block entry/exit residency
   - per-op lowering and pressure policy
   - edge repair requirements
4. Let `rewrite.rs` lower once from those chosen decisions.
5. Run structural cleanup on the prepared SSA-IR.
6. Run middle SSA optimization:
   - absorb single-use constant producers into `SsaOperand::Const`
   - fold fully constant pure leaf ops
   - remove dead constant producers
7. Hand `machine/` explicit SSA-IR with explicit:
   - blocks and edges
   - SSA values
   - `Spill` / `Fill`
   - `LocalGet/SetSlot`
   - `LocalGet/SetCache`
   - `LocalEnsureCache` / `LocalDropCache`

The design constraints are:

- no whole-function cache hotness model
- no late reconstruction of boundary state
- no repeated full re-lowering loop
- heuristics are swappable, ownership is not

## Planner and rewrite ownership

`joint_plan/` is a consultant. It is not the owner of mutable lowering state.

`rewrite.rs` is the owner of:

- the current transient stack state
- the current cached-local state
- local materialization state
- dynamic bank usage
- SSA value allocation and mappings
- emitted ops
- block entry and exit boundary state as they are being realized

At each decision point:

1. `rewrite.rs` knows the current real state
2. it asks `joint_plan/` for a policy decision
3. `joint_plan/` answers with a decision
4. `rewrite.rs` applies that decision, updates state, and emits IR

This is the intended ownership split.

## What the planner needs to know

The planner needs facts, not ownership of the mutable rewrite state object.

### Function-level facts

- CFG
- slot-only SSA blocks
- exact Wasm stack contract on block entry and exit
- local slot type, bank, and cost
- dynamic budgets
- block-local value ranking
- canonical predecessor for each block

### Block-entry facts

- exact incoming transient stack state
- chosen entry cached-local set

### Per-instruction facts supplied by the rewriter

At a decision point, `rewrite.rs` supplies:

- current block
- current instruction
- current transient stack state
- current resident cached locals
- current dynamic pressure per bank
- current local dirty/clean state if tracked
- current local future-use facts in the rest of the block
- current transient future-use facts if needed

This is enough for the planner to answer policy questions.

## What the planner outputs

The planner does not emit IR. It returns decisions.

### Whole-block outputs

- canonical predecessor
- tentative entry cached-local set
- finalized entry cached-local set after one lowering pass
- repair requirement derived from that finalized entry

### Per-instruction outputs

At each relevant point, the planner should be able to answer:

- should this `LocalGet` use slot or cache
- should this `LocalSet` use cache
- under pressure, should we drop a cached local or spill a transient value
- which cached locals should be admitted at block entry

### Edge outputs

For each non-canonical incoming edge:

- which locals must be ensured
- which locals must be dropped

That is enough for `rewrite.rs` to materialize legal SSA-IR.

## Planning interface

The planner must exist as an explicit interface even if some policies start
simple.

The important rule is:

- `rewrite.rs` owns the real lowering state
- `joint_plan/` answers policy questions about that state

### Mandatory planning responsibilities

These must be implemented from the beginning because they are required for
correctness.

#### 1. Transient spill/fill planning

This is mandatory.

The new system uses the full dynamic-bank budget, but transient stack legality
still needs an explicit plan.

For each instruction, the planner must be able to answer:

- how many transient stack values must remain resident for this op to be legal
- which transient values may be spilled
- when a spilled transient must be filled before use

Examples:

- `i32.add`
  requires the top two stack values to be resident before the op
- `local.set`
  requires the top one stack value to be resident before the op
- `br_if`
  requires the condition value to be resident before the op

So even with a simple policy, `rewrite.rs` must never spill below the transient
floor required by the current op.

#### 2. Pressure resolution

When dynamic pressure exceeds budget, the planner must decide:

- drop which cached local
- spill which transient value

The initial policy can be simple:

- as long as the transient spill plan remains satisfied, prefer dropping cached
  locals before spilling transients

This is enough for a correct first implementation.

#### 3. Local access mode

For every local access, the planner must choose:

- `LocalGetSlot` or `LocalGetCache`
- `LocalSetCache` by default
- `LocalSetSlot` only as a temporary fallback while the old path still exists

This depends on current residency, pressure, and block-local value ranking.

### Boundary-level planning responsibilities

These must also exist as explicit planner outputs, even if the initial policy
is simple.

#### 4. Block entry resident set

For each block, the planner must define which values are expected to be resident
on entry.

This is chosen from one ranked resident-set policy that considers:

- local cache candidates
- touched entry-stack values
- carried values from the canonical predecessor exit
- subject to the shared dynamic budget

This decision is not the same as entry materialization.

For locals, the planner must distinguish:

- resident on entry
- must be ensured at entry

A local may deserve boundary residency even when it should not be loaded on the
incoming edge.

Important example:

- if the first access is a write, the local may still deserve entry and exit
  residency on a hot path
- but the old value should not be loaded on entry
- the first `LocalSetCache` should materialize it instead

The key consistency failure is:

- a value is materialized on entry
- and then dropped before its first use

That means the planner's residency ranking and eviction policy are inconsistent.

The important single-pass rule is:

- first build a tentative entry
- lower the block once from that tentative entry
- then finalize entry by trimming only useless carried-in locals

The finalized entry is:

```text
final_entry =
    read_first_on_entry
    union
    (tentative_entry intersect actual_exit)

Where:
- `read_first_on_entry` means locals whose incoming value is actually needed by
  the block before any in-block write
- write-first locals do not become part of the public boundary just because
  they are used later in the block
- if a write-first local survives cached inside the block, that is still fine;
  it just should not force loop-entry reserve/repair churn
```

Only locals unused in the block can be removed by this step.

So:

- if a carried local is used in the block, keep it in entry
- if a carried local is unused but survives to exit, keep it in entry
- if a carried local is unused and does not survive to exit, remove it from
  entry

This avoids any full recompute loop. The block is lowered once.

At the first semantic op of a CFG block, the chosen `block_open(block)`
transient state is still the authoritative structural boundary. But the first
`before_op` in that block may legally fill values from that structural entry
before executing the op. This matters for typed loop headers: the loop params
may stay structurally spilled at the boundary while the first loop-body op
still requires them live immediately before execution.

#### 5. Observed block exit cache state

Block exit is not independently planned first.

The rewriter lowers once from the tentative entry and observes the actual exit
cached-local state.

That observed exit is then used only for:

- trimming unused carried-in locals from tentative entry
- keeping hot loop state close to the planned entry
- deriving trivial edge repair

#### 6. Edge repair

Under the current Wasm/SSA model, repair is almost entirely a cached-local
question.

The transient stack contract is already fixed by Wasm semantics, and SSA block
params already handle transient value flow.

So once the finalized block entry is known, repair is simple:

- which locals to `LocalEnsureCache`
- which locals to `LocalDropCache`

Repair exists only to match every incoming edge to that finalized entry. It must
never be used to reconstruct what the block entry should have been.

### Planning responsibilities that can stay simple initially

These interfaces should exist now, but their policy can remain basic until
later.

#### 7. Canonical predecessor choice

The planner must choose one canonical incoming edge per block.

The initial policy can be:

- single-predecessor block: that predecessor
- loop header: the backedge
- ordinary merge: the first or hottest predecessor

#### 8. Cache eviction order

The planner should expose which cached local to drop first under pressure.

The initial policy can be:

- carried-through cached locals unused in this block first
- then colder locals used in this block

#### 9. Transient spill order

The planner should expose which transient to spill first when cached locals are
not enough.

The initial policy can be:

- spill lower-priority or deeper stack values first, while preserving the
  transient floor required by the current op

### Optional later planning responsibilities

These do not block the first correct rewrite, but the interface should leave
room for them.

#### 10. Dirty-cache writeback policy

If dirty-state tracking is enabled, the planner may later choose whether a
dirty cached local should:

- stay resident and dirty
- be written back and kept resident
- be dropped with writeback

#### 11. Hot-edge weighting and profile bias

The planner may later use block/edge profile information to improve:

- canonical predecessor choice
- cache carry-through decisions
- merge behavior

## What the rewriter needs to keep

`rewrite.rs` is stateful. It must keep the real lowering facts as it walks the block.

### Dynamic block state

- transient stack values, in stack order
- resident cached locals, in deterministic order
- dirty/clean state for cached locals if tracked
- dynamic usage per bank
- current local residency/materialization state
- current emitted ops

### Structural state

- current block id
- current instruction index
- value allocator
- already-fixed transient block params and edge arguments from Wasm semantics

### Boundary state

- chosen block entry boundary
- actual block exit boundary as produced by rewriting

The rewriter is the part that makes the program real. The planner only tells it
what to do.

## Summary

The new `middle/` should be built around this ownership split:

- front half:
  - semantic CFG formation
  - slot-only SSA lowering
- decision layer:
  - `joint_plan/` decides cache and spill policy
- main transformation:
  - `rewrite.rs` performs one-pass joint block rewrite using exact stack state plus chosen entry cache state
- boundary reconciliation:
  - canonical predecessor
  - repair blocks only on non-canonical incoming edges
- post-rewrite cleanup:
  - thread empty goto blocks
  - merge trivial goto-only successors
  - remove unreachable blocks
  - canonicalize pure cache-materialization runs
- post-cleanup middle SSA optimization:
  - absorb constants into leaf operands when backend lowering already supports immediates
  - fold fully constant pure leaf ops
  - delete dead const producers

The critical principle is:

- the entry boundary is known before lowering the block
- the block is lowered from that known boundary
- non-canonical edges repair into it

That is the clean replacement for the old `middle/`.

## Current unified boundary policy

This section describes the intended simple near-optimal policy for boundary
selection.

### First-principles goal

The goal is not to cache locals separately from stack values.

The goal is:

- keep the highest-value values resident in registers
- subject to the shared dynamic-bank budget
- across the hottest part of execution
- while avoiding useless churn

Those resident values can be:

- local caches
- transient stack values

Boundary selection is not an isolated problem. It is one part of resident-set
planning.

### One policy for all blocks

Use one policy for all blocks.

Do not invent a separate loop algorithm for cache selection.

Loops still matter, but they should fall out naturally from:

- canonical predecessor choice
- value ranking
- exit-to-entry carry-through

The only explicit loop bias needed is:

- loop headers should prefer the backedge as canonical predecessor

Everything else should use the same boundary-selection algorithm as ordinary
blocks.

### Entry region

For block entry planning, only the block entry region matters.

The entry region runs from block start to the first hard barrier.

Typical hard barriers:

- call
- return
- control terminator
- any forced cache-clear boundary

Values not touched meaningfully before the first hard barrier should not be
preloaded eagerly at block entry.

Pure structural cleanup of a value is not a meaningful hot-use signal.

In particular:

- a pure `Drop` of an entry stack value should not count as evidence that the
  value deserves entry residency
- otherwise the planner can keep cold entry stack live, then immediately evict
  a genuinely useful ensured local under pressure

### Ranked values

For each block, rank boundary candidates from one unified list.

Candidate kinds:

- local slot `slotN`
- entry stack value `stack[-k]`

At rewrite time the entry stack already exists as explicit SSA values, so stack
usage can be counted just like local usage.

Local candidates and stack candidates therefore share one ranking model.

The realization is still constrained on the stack side:

- locals can be chosen as any subset
- stack values are chosen as a resident top suffix

So the stack side is equivalent to choosing a `spill_depth`.

### Ranking signals

Keep the signals simple and cheap.

For a local candidate:

- touched before the first hard barrier
- first-use distance
- use count before the first hard barrier
- carried by the canonical predecessor exit or not
- whether the first access is a read or a write

For an entry stack value:

- touched before the first hard barrier
- first-touch distance
- touch count before the first hard barrier

The stack-side "touched" signal means semantically used, not merely popped for
cleanup.

The most important signals are:

- carried on canonical predecessor exit
- used early
- reused enough to justify residency

Single-use locals are not automatically bad boundary candidates.

If a local is used early, preloading it may still be good because:

- the load happens earlier than immediate use
- a later `machine/` aliasing pass may fold away the reg-to-reg copy path

So single-use locals should not be rejected by policy alone.

The first-access kind matters more:

- read-first local: if chosen and not already carried, it may need entry
  materialization
- write-first local: if chosen, it should usually reserve residency but not
  trigger an entry load

Without profile data, the base hot-edge signal should stay simple:

- use the canonical predecessor as the carry-through signal
- prefer loop backedges as canonical predecessors
- do not require weighted incoming edge coverage in the base algorithm

Profile-weighted incoming coverage can remain a later optional improvement.

### Residency versus materialization

Boundary planning must distinguish:

- which values deserve residency
- which values must be materialized immediately at entry

Those are not the same decision.

For locals:

- read-first and chosen but not carried: materialize on entry
- write-first and chosen but not carried: reserve residency but do not load on
  entry

For stack values:

- the entry transient contract is structural
- block-open must not rewrite it just to make room for cached locals
- later mid-block pressure may still spill transients according to the normal
  pressure policy

Function entry should follow the same planner policy:

- the planner may still choose entry cached locals for the entry block
- the backend should materialize them with prologue loads from frame slots
- they must not become hidden extra entry-block params

### Entry-state rule

At `block_open(block)`:

1. start from the exact Wasm stack contract
2. keep that transient contract unchanged
3. rank entry-region locals
4. add a direct-successor carry bonus for tiny dispatcher / guard blocks, so
   hot pass-through locals can stay resident through a block that does not
   read them itself
5. prefer values already carried by the canonical predecessor exit
6. choose cached locals under the remaining budget left by the structural
   transient contract
7. decide which chosen locals must actually be ensured on entry
8. allow write-first locals to reserve boundary residency without forcing an
   entry load

This means the current boundary choice is about cached locals, not about
rewriting stack join shape.

### Pseudocode

```text
function plan_block_open(block):
    structural_transient = exact_structural_entry_transient(block)
    canonical_pred = canonical_predecessor(block)
    carried_locals = []
    if canonical_pred exists:
        carried_locals = exit_cached_locals(canonical_pred)
    budgets = dynamic_bank_budgets(block)   // {gp, fp}
    region = scan_entry_region(block)
    successor_bonus = score_direct_successor_hotness(block)
    remaining_budget = budgets - cost_by_bank(structural_transient.live_values)

    local_candidates = []
    for each local L in union(region.ranked_locals, successor_bonus.locals):
        info = region.local_info(L)
        local_candidates.push({
            local: L,
            carried: L in carried_locals,
            first_access: info.first_access_kind,   // ReadFirst or WriteFirst
            first_read: info.first_read_distance,
            first_write: info.first_write_distance,
            read_count: info.read_count,
            write_count: info.write_count,
            score: score_local(info, successor_bonus[L], L in carried_locals),
        })

    chosen_locals = choose_best_locals_under_budget(
        local_candidates,
        remaining_budget,
    )

    ensure_on_entry = []
    for each chosen local C in chosen_locals:
        if C.carried:
            continue
        if C.first_access == ReadFirst:
            ensure_on_entry.push(C.local)
        else:
            // Write-first locals reserve boundary residency but do not load the
            // old value on entry. The first LocalSetCache materializes them.
            continue

    return BlockOpenDecision {
        structural_transient: structural_transient,
        entry_cached_locals: chosen_locals,
        ensure_on_entry: ensure_on_entry,
    }


function score_local(info, successor_bonus, carried):
    if info.first_access_kind == None and successor_bonus == 0:
        return VERY_LOW

    score = 0
    score += successor_bonus
    if carried:
        score += CARRY_BONUS

    if info.first_access_kind == ReadFirst:
        score += EARLY_USE_BONUS / (1 + info.first_read_distance)
        score += REUSE_BONUS * info.read_count
    else:
        score += EARLY_USE_BONUS / (1 + info.first_write_distance)
        score += WRITE_FIRST_BOUNDARY_BONUS
        score += REUSE_BONUS * info.write_count

    return score

```

This pseudocode is intentionally simple:

- transient entry shape is structural and is not optimized at block-open
- cached locals are the real boundary-choice problem
- direct-successor hotness lets tiny dispatch headers keep pass-through loop
  locals hot instead of forcing backedge churn
- write-first locals can be boundary-resident without forcing an entry load
- canonical predecessor carry-through is the base hot-edge signal
- single-use early locals are still eligible; later `machine/` local aliasing
  may delete the reg-to-reg copy path
- the same value ordering should later guide cache eviction; otherwise the
  `dropped_before_first_use` diagnostic will expose an inconsistent planner

### Entry finalization and repair

Block-entry planning is single-pass:

1. build a tentative entry from hot block values plus carried predecessor values
2. lower once from that tentative entry
3. observe the actual exit
4. finalize entry as:

```text
final_entry =
    read_first_on_entry
    union
    ((tentative_entry with no entry-region access) intersect actual_exit)
```

5. repair all incoming edges to that finalized entry

This means repair is a consequence of finalized entry, not a separate
optimization problem.

```text
function finalize_block_entry(tentative_entry, entry_region_info, actual_exit):
    final_entry = {}
    for each local L in tentative_entry:
        if entry_region_info[L].first_access == ReadFirst:
            final_entry.add(L)
        else if L in actual_exit:
            final_entry.add(L)
        else:
            // If a tentative local dies before exit, it was not worth keeping
            // in the public boundary. This trims write-first temporaries while
            // still letting surviving write-first loop locals stay hot.
            skip
    return final_entry


function derive_edge_repair(pred_exit, final_entry):
    ensure_cached_locals = []
    reserve_cached_locals = []
    for each local L in final_entry - pred_exit:
        if first_access_kind(L) == WriteFirst:
            reserve_cached_locals.push(L)
        else:
            ensure_cached_locals.push(L)
    drop_cached_locals = pred_exit - final_entry
    return {
        ensure_cached_locals,
        reserve_cached_locals,
        drop_cached_locals,
    }
```

For function entry, use the same rule from an empty predecessor. In the current
implementation that materialization may be emitted as one synthetic entry-repair
block that jumps to the original entry block.

### Post-rewrite cleanup

Repair is still the right boundary mechanism, but it can create extra structural
blocks in the prepared SSA.

That cleanup is now part of the intended pipeline, not an optional later pass.

The cleanup rules should stay purely structural:

- do not change the finalized entry/exit cache policy
- do not invent new cache behavior
- do not weaken edge semantics
- only remove blocks or ops whose behavior is already implied by neighbors

The cleanup should do four things:

1. thread empty goto blocks by composing edge bindings through them
2. merge an unconditional goto predecessor into a single-predecessor successor
3. remove unreachable blocks
4. canonicalize runs of pure cache materialization ops (`ensure`, `reserve`,
   `drop`) while preserving required `drop-before-materialize` ordering

```text
function cleanup_prepared_ssa(program):
    repeat until fixed point:
        simplify_cache_only_runs(program)
        if thread_one_empty_goto_block(program):
            continue
        if merge_one_goto_successor(program):
            continue
        if remove_unreachable_blocks(program):
            continue


function thread_one_empty_goto_block(program):
    find block B such that:
        B.ops is empty
        B.terminator is Goto(T, bindings)
    rewrite every predecessor edge P -> B into P -> T
    compose predecessor bindings with B.bindings
    remove B


function merge_one_goto_successor(program):
    find block P such that:
        P.terminator is Goto(S, bindings)
        S has exactly one predecessor
    substitute S.params with bindings inside S.ops and S.terminator
    append substituted S.ops to P.ops
    replace P.terminator with substituted S.terminator
    remove S


function simplify_cache_only_runs(program):
    within each maximal run of:
        LocalEnsureCache
        LocalReserveCache
        LocalDropCache
    keep only the net effect per local
    but preserve any required drop that must happen before a later ensure or
    reserve for the same local
```

### Failure condition

For boundary selection, the key diagnostic failure is:

- a value is materialized into the entry state
- then it is dropped before its first use

That means the planner admitted the wrong value or used an inconsistent eviction
ranking.

If a value survives until use and is later evicted due to pressure, that is a
pressure-resolution question, not a boundary-selection question.

The post-lowering check can be written as:

```text
function validate_block_open(block, decision, lowering_trace):
    for each local L in decision.ensure_on_entry:
        if lowering_trace.dropped_before_first_use(L):
            fail("block_open admitted the wrong local or eviction ranking disagrees")
```

### Why loops work under the same policy

In a loop:

- the backedge is canonical
- values carried on the backedge exit seed the tentative entry
- values used early and repeatedly inside the loop rank highly

So a profitable loop-hot resident set should emerge naturally without needing a
separate loop-only cache-selection algorithm.

The steady-state goal is:

- hot loop backedge exit should stay as close as possible to the finalized loop
  entry

That keeps the loop hot without repeated repair churn.

## Mid-block pressure policy

Mid-block planning must choose operation shape and eviction together.

It is not enough to decide:

- `local.get` should use cache

and only then ask:

- what should be evicted

because different lowering shapes need different numbers of live registers.

Important example:

- `LocalGetSlot -> SSA` may need one new live register
- `LocalGetCache -> SSA` may need two live registers if we want both:
  - the cached local to remain resident
  - the SSA result value

So pressure resolution must plan the whole step together.

The fit check must cover both:

- the transient window before the op
- the immediate transient window after the op

Otherwise a plan can fit before the op and still overflow immediately after a
push or other result-producing op.

### Eviction classes

At a pressure point, classify resident values in this order:

1. `unused_in_block`
   Values not used anywhere in the current block.
2. `dead_after_point`
   Values used earlier in the block but not in the remaining instructions.
3. `live_after_point`
   Values still used in the remaining instructions.

This gives the first simple policy:

- evict `unused_in_block` first
- then `dead_after_point`
- only evict `live_after_point` if forced

Among values in the same class, use the same whole-block ranking policy that
drives block-entry hotness.

So if two values are both dead after the current point:

- keep the one that is hotter for the block as a whole
- evict the colder one

This is enough to preserve loop steady state without separately planning exit.

For transients, the implementation tracks whole-block usage of block-local
stack symbols. That lets the planner rank:

- cached locals
- the current bottom live transient

under the same keep-key model.

Important legality nuance:

- stack spilling is suffix-based
- so the only spillable transient is the current bottom live value

That means pressure sometimes has to spill through a cold bottom value in the
"wrong" bank in order to expose the deeper values that are really causing the
overflow. The policy should prefer victims that directly relieve the current
overflow, but if none exist it must still allow bottom-transient spill to make
progress.

### Pseudocode

```text
function plan_op(block, op_index, current_state):
    op = semantic_op(block, op_index)
    floor = required_transient_floor(op)
    block_ranking = block_hot_ranking(block)

    candidates = enumerate_op_shapes(op, current_state)
    // Examples:
    // - LocalGetSlot
    // - LocalGetCacheAlreadyResident
    // - LocalGetCacheAndKeepResident
    // - LocalSetCache

    best = none

    for each shape in candidates:
        state = clone(current_state)

        realize_transient_floor(state, floor)

        demand = extra_live_cost_by_bank(shape, state)
        evictions = evict_to_fit(block, op_index, state, demand, shape, block_ranking)
        if evictions == impossible:
            continue

        candidate = {
            shape: shape,
            evictions: evictions,
            score: shape_score(shape) - eviction_cost(evictions),
        }

        if best is none or candidate.score > best.score:
            best = candidate

    return best


function evict_to_fit(block, op_index, state, demand, shape, block_ranking):
    evictions = []

    while not fits_after(state, demand):
        victim = choose_eviction_victim(
            block,
            op_index,
            state,
            shape,
            block_ranking,
        )
        if victim == none:
            return impossible

        apply_evict(state, victim)
        evictions.push(victim)

    return evictions


function choose_eviction_victim(block, op_index, state, shape, block_ranking):
    cache_victim = weakest_cached_local_in_overflowing_bank(
        block,
        op_index,
        state,
        shape,
        block_ranking,
    )
    bottom_transient = bottom_live_transient(state)

    if bottom_transient != none:
        transient_keep = keep_key(block, op_index, bottom_transient, shape, block_ranking)
        transient_relief = relief_score(bottom_transient, state)
    else:
        transient_keep = none
        transient_relief = -1

    if cache_victim != none:
        cache_keep = keep_key(block, op_index, cache_victim, shape, block_ranking)
        cache_relief = relief_score(cache_victim, state)
    else:
        cache_keep = none
        cache_relief = -1

    if cache_relief > transient_relief:
        return cache_victim
    if transient_relief > cache_relief:
        return bottom_transient
    if cache_keep != none and (transient_keep == none or cache_keep <= transient_keep):
        return cache_victim
    return bottom_transient


function keep_key(block, op_index, value, shape, block_ranking):
    if used_now_by_shape(value, op_index, shape):
        return (3, INF)

    if used_after_point(value, block, op_index):
        return (
            2,
            remaining_use_score(value, block, op_index)
                + block_hot_score(value, block_ranking),
        )

    if used_anywhere_in_block(value, block):
        return (
            1,
            block_hot_score(value, block_ranking),
        )

    return (0, 0)
```

This policy is intentionally conservative:

- values unused in the entire block are always the first eviction candidates
- values dead after the current point are next
- values still needed later are last
- values required by the current op shape get the highest possible keep key
- op lowering mode and eviction are chosen together, not in separate planner
  passes

## Local access policy

When there is spare capacity, local access policy is simple:

- always keep the local cached

The interesting case is when there is no spare capacity for an additional cached
local.

For `local.get`, this does not mean the op itself is impossible. The op still
needs one register for the SSA result no matter what.

The real policy question is:

- after producing the SSA result, should we spend one additional cache slot to
  keep this local resident?

So `local.get` should be treated as:

- mandatory: produce the SSA result
- optional: keep the local cached after the get

That optional part should reuse the same keep-key logic as pressure eviction.

`local.tee` asks the same cache-admission question, but without the mandatory
result-allocation pressure:

- it consumes one SSA value
- it produces one SSA value
- stack height does not grow
- the only real policy question is whether to keep the local cached after the
  tee

### Pseudocode

```text
function plan_local_get(block, op_index, local, state, block_ranking):
    if local already resident in cache:
        return LocalGetCacheAlreadyResident

    // The SSA result is mandatory. The extra cache slot is optional.
    target_keep = keep_key_for_local_after_get(block, op_index, local, block_ranking)

    if fits_extra_cache_slot_for_local(state, local):
        return LocalGetSlotPlusCache

    victim = weakest_resident_value_in_same_bank(
        block,
        op_index,
        state,
        shape = LocalGetSlotPlusCache,
        block_ranking,
    )

    if victim != none and target_keep > keep_key(block, op_index, victim, LocalGetSlotPlusCache, block_ranking):
        return LocalGetSlotPlusCacheAndEvict(victim)

    return LocalGetSlotOnly


function keep_key_for_local_after_get(block, op_index, local, block_ranking):
    info = remaining_local_info(block, op_index, local)
    if not info.used_after_point:
        return (0, 0)

    return (
        2,
        remaining_use_score(local, block, op_index)
            + block_hot_score(local, block_ranking),
    )


function plan_local_set(block, op_index, local, state, block_ranking):
    // local.set consumes one SSA value, so the local can naturally take over
    // that residency. No extra cache slot needs to be found at the set point.
    return LocalSetCache


function plan_local_tee(block, op_index, local, state, block_ranking):
    if local already resident in cache:
        return LocalTeeCacheAlreadyResident

    // local.tee keeps stack height flat. The question is only whether the local
    // also deserves cache residency after the tee.
    target_keep = keep_key_for_local_after_get(block, op_index, local, block_ranking)

    if fits_extra_cache_slot_for_local(state, local):
        return LocalTeeCache

    victim = weakest_resident_value_in_same_bank(
        block,
        op_index,
        state,
        shape = LocalTeeCache,
        block_ranking,
    )

    if victim != none and target_keep > keep_key(block, op_index, victim, LocalTeeCache, block_ranking):
        return LocalTeeCacheAndEvict(victim)

    return LocalTeeSlot
```

This policy stays simple:

- with spare capacity, always cache
- under pressure, compare the target local against the weakest current resident
  value in the same bank
- `local.get` pays one mandatory register for the SSA result and only asks
  whether the extra cache residency is worth it
- `local.set` naturally becomes `LocalSetCache` because it consumes one SSA slot
- `local.tee` asks the same cache-admission question as `local.get`, but it does
  not pay the extra stack-growth cost

Later TODO:

- once cache-first `local.set` works end to end and dirty/writeback remains
  backend-owned, delete `LocalSetSlot` from the middle-layer policy and SSA-IR

## Backend bank contract

The backend contract is now:

- one `gp_dynamic_budget`
- one `fp_dynamic_budget`
- no static cached-local/linear-value partition inside either bank

This has an important consequence for `machine/`:

- semantic `LinearValue` means "this dynamic register currently holds one
  linear SSA-like machine value"
- semantic "cached local" means "this dynamic register is currently bound to a
  local cache"
- therefore late machine passes must not infer linear-value ownership from the
  machine register number alone
- MachineIR should carry explicit ownership for block params and for ambiguous
  defs such as `Move` / `Load`

Physical register order is still allowed and useful, but it is only a backend
allocation preference. It is not a semantic class boundary.
