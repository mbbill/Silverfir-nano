# ALGORITHM4 + Lane Mapping

This file merges the original design docs:

- `sf-nano-core/src/vm/middle/ALGORITHM4.md`
- `sf-nano-core/src/vm/middle/LANE_MAPPING.md`

The text below stays intentionally close to those originals. I have only made
small edits where needed to add implementation-status notes or to fold the two
docs into one file.

## Current Status

- The public resident-set solver from `ALGORITHM4` is mostly implemented in
  `sf-nano-core/src/vm/middle/joint_plan/region_solver.rs`.
- The final extraction differs slightly from the original Step 5 description:
  the current code does a recursive per-region capacity-constrained extraction
  instead of a marginal-value projection pass.
- Rewrite-side public-state use and edge repair are implemented, including
  `LocalEnsureCache`, `LocalDropCache`, and `LocalReserveCache`.
- Machine-side lane mapping is now mostly implemented in
  `sf-nano-core/src/vm/machine/lower_cache_layout.rs`.
- Still not yet implemented:
  - private flex promotion for non-resident locals as described below
  - feeding lane-remap cost back into `ALGORITHM4`'s residency objective
  - subtree-stickiness as an explicit objective in machine-side micro-repack
  - exact fragmented FP micro-repack analogous to the GP exact search

## Algorithm 4: Cost-Optimal Public Residency via Region-Tree DP

### Problem

See `PRESSURE.md` for measured data.

The current per-block cache planner minimizes per-block frame access cost but
ignores transition cost at edges. The result is massive boundary churn.

Previous algorithms (`ALGORITHM2`: global set, `ALGORITHM3`: root + loop
override) treat stability as a structural constraint rather than a cost to
optimize. They work, but they are approximations of what the algorithm should
really do: minimize total cost.

### First-principles objective

For one register bank, let:

- `x[R,L] ∈ {0,1}`: local `L` is publicly resident in region `R`
- `units(L)`: register units consumed by `L` (`i64` on 32-bit = 2, else 1)
- `benefit(R,L)`: weighted frame ops saved inside `R` if `L` is resident
- `call_tax(R,L)`: weighted cost of keeping `L` across calls inside `R`
- `entry_freq(R)`: estimated frequency of entering `R` from its parent
- `exit_freq(R)`: estimated frequency of leaving `R` to its parent
- `trans_cost(L)`: cost of one ensure or drop transition for `L` (= 1
  frame-op equivalent per unit)
- `cap(R)`: available register-unit capacity at `R` after transient headroom

All cost terms are in the same unit: weighted frame-op equivalents.
`benefit` and `call_tax` are pre-weighted by block frequency during
construction. The objective contains no additional `freq(R)` multiplier.

Objective:

```text
maximize
    Σ(R,L) benefit(R,L) * x[R,L]
  - Σ(R,L) call_tax(R,L) * x[R,L]
  - Σ(R)   mismatch_cost(R)

subject to
    Σ_L units(L) * x[R,L] <= cap(R)     for every region R
```

Where `mismatch_cost(R)` charges for every residency change at the boundary
between region `R` and its parent:

```text
mismatch_cost(R) =
    Σ_L edge_cost(x[parent(R),L], x[R,L], R, L)
```

For loop regions, residency changes cost on both entry and exit:

```text
edge_cost(p, s, R_loop, L) =
    if p == s: 0
    else:      (entry_freq(R) + exit_freq(R)) * trans_cost(L)
```

This is symmetric: adding a local at a loop boundary costs the same as
removing one, because both require one op on entry and one on exit.

For the root region, there is no parent to restore on function return.
The only cost is one-time entry materialization:

```text
edge_cost(0, s, R_root, L) =
    if s == 0: 0
    if s == 1: entry_freq(root) * trans_cost(L)
```

The root is always evaluated with parent state = 0.

### Why this formulation is right

- Whole-function stability emerges if transition costs dominate benefit
  for most locals. No stability constraint needed.
- Loop-specific overrides emerge when a local's in-loop benefit exceeds
  the transition cost at the loop boundary. No loop special-case needed.
- Call cost is internalized as a penalty on residency, not a structural
  barrier. Call-heavy loops naturally get smaller resident sets.
- Capacity is a real constraint, not a heuristic cutoff.
- `i64` unit cost is built in via `units(L)`.
- One unit system: everything is weighted frame-op equivalents.

### Practical solver: region-tree DP with capacity prices

#### Step 1: Build the region tree and compute inputs

Use the Wasm structured loop hierarchy directly. No SCC discovery needed.

```text
Regions = { root } ∪ { one region per Wasm loop instruction }
parent(R) = enclosing loop, or root if top-level
OwnedBlocks(R) = blocks in R not inside any child loop
```

For each region `R`, compute from its owned blocks:

##### benefit(R, L)

```text
benefit(R, L) = Σ B in OwnedBlocks(R):
    block_weight(B) * access_count(B, L)
```

`access_count(B, L)` = total `local.get` + `local.set` + `local.tee` of `L`
in block `B`. Each such access costs one frame op if uncached, zero if cached.

`block_weight(B)` = estimated execution frequency. Default: `10^loop_depth(B)`.

The result is in weighted-frame-op units. No further `freq(R)` multiplier is
applied when this term enters the objective.

##### call_tax(R, L)

```text
call_tax(R, L) = Σ B in OwnedBlocks(R):
    block_weight(B) * calls_in_block(B) * keep_cost(L)
```

`keep_cost(L)` depends on the current call implementation:

| Strategy | keep_cost(L) |
| --- | --- |
| Re-ensure after every call | `units(L)` |
| Callee-saved subset | `0 if fits, else units(L)` |
| Machine-level save/restore | `0` |

For v1, use `keep_cost(L) = units(L)` (re-ensure). Conservative and simple.

Already in weighted-frame-op units.

##### entry_freq(R) and exit_freq(R)

```text
entry_freq(R) = block_weight(header(R)) / assumed_trip_count
exit_freq(R) = entry_freq(R)
```

For the root: `entry_freq = 1`.

`assumed_trip_count` is a tuning constant (default: 8). It only affects the
ratio between per-iteration benefit and per-entry transition cost. It does
not need to be accurate.

##### trans_cost(L)

```text
trans_cost(L) = units(L)
```

One frame op per register unit per transition (ensure or drop).

##### cap(R): available cache capacity

```text
cap(R) = dynamic_budget - headroom(R)
```

`headroom(R)` must account for the full live transient pressure at any point
inside the region, not just the stack-depth swing. This includes:

- carried entry-stack values that remain live inside the block
- block-internal computation results (leaf op outputs, `local.get` results for
  non-resident locals)
- values live across spill points

The existing rewriter tracks this via `fits_with_cached_locals` and
`count_live_bank_budget_units`. The region solver should use the same
computation, evaluated at the worst-case block within the region:

```text
headroom(R) = max over B in OwnedBlocks(R) of:
    peak_live_transient_units(B)
```

Where `peak_live_transient_units(B)` counts, at the highest-pressure point
in block `B`, the total register units occupied by transient values
(SSA values that are not public-resident locals). This is computed from the
semantic stack shapes and the slot-only SSA, reusing the existing
infrastructure in `joint_plan/entry_region.rs` and `joint_plan/pressure.rs`.

On x86_64 (budget=9) or ARMv7a (budget=8), this headroom eats a larger
fraction of the budget, naturally limiting how many locals can be resident.

The headroom is exact, not softened. No upper clamp is applied. If one
pathological block in a region has very high transient pressure, the region's
capacity shrinks accordingly. The cost model then decides which locals are
still worth caching at the reduced capacity: a local that barely justified
residency will be priced out, which is the correct behavior.

The only floor is `MIN_HEADROOM = 3` (worst-case single-op semantic
requirement), ensuring the solver never promises more capacity than exists.

If a rare block's actual transient pressure exceeds even the exact headroom
(due to estimation error or edge cases), the rewriter's pressure fallback
handles it: temporarily evict the weakest public local within that block and
restore before the terminator. This is a safety net, not a normal path.

Status note:

- The current code does compute exact per-bank peak live transient units from
  `OpPlan.before` and `OpPlan.after`, and subtracts those from the dynamic bank
  budgets.
- The current code does not literally implement the `MIN_HEADROOM` floor
  described above.
- Rare over-budget cases are handled by rewrite-time pressure fallback.

#### Step 2: Introduce capacity prices

For each region `R`, maintain a dual price `λ[R] >= 0` representing the
marginal cost of one register unit at `R`.

Define the price-adjusted reward for the DP:

```text
reward(R, L) = benefit(R, L) - call_tax(R, L) - λ[R] * units(L)
```

All three terms are in weighted frame-op equivalents. The subtraction is
meaningful.

When `λ[R] = 0`, every local with `benefit > call_tax` wants residency.
As `λ[R]` rises, weaker locals get priced out. The price balances supply
(capacity) and demand (locals wanting residency).

#### Step 3: Per-local tree DP (the core)

For fixed `λ`, each local `L` is an independent optimization on the region
tree.

The DP computes, for each region `R` and parent state `p ∈ {0,1}`, the
maximum net value of the subtree rooted at `R`:

```text
DP[R, p] =
    max over s ∈ {0,1} of:
        V(R, p, s)

V(R, p, s) =
    reward(R, L) * s
  - edge_cost(p, s, R, L)
  + Σ child C of R: DP[C, s]
```

Edge cost depends on whether `R` is the root or a loop:

```text
edge_cost(p, s, R, L) =
    if p == s:
        0
    elif R is root:
        // Function entry: materialize once. No restore on function return
        // (the frame is discarded). Only pay when s=1 (adding a local).
        if s == 1: entry_freq(root) * trans_cost(L)
        else:      0   // root can't have p=1 (parent_state is always 0)
    else:
        // Loop boundary: pay on both entry and exit edges.
        // Entry: ensure (if 0→1) or drop (if 1→0).
        // Exit: reverse op to restore parent state.
        (entry_freq(R) + exit_freq(R)) * trans_cost(L)
```

Note: the root is always evaluated with `p=0` (nothing cached before
function entry), so the `s=0` branch of the root case is unreachable.
The formula is stated explicitly for completeness.

Solve bottom-up (leaves first). Extract decisions top-down starting from
`DP[root, p=0]`.

Complexity: `O(regions)` per local per iteration.

#### Step 4: Update capacity prices

After solving all locals for the current `λ`:

```text
demand[R] = Σ_L units(L) * x[R,L]
overload[R] = demand[R] - cap(R)
λ[R] = max(0, λ[R] + step * overload[R])
```

A reasonable step size: `step = 1.0 / cap(R)`.

Repeat steps 3-4 for a small number of iterations (3-8).

Convergence is fast because:

- The region tree is small (typically 5-20 nodes)
- Each local's DP is `O(regions)`
- Most locals have clear benefit rankings; few are on the margin

Status note:

- The current implementation does this with a fixed `PRICE_ITERS = 12`.

#### Step 5: Final projection

After the last iteration, some regions may still slightly exceed capacity
(the Lagrangian relaxation provides an upper bound, not a feasible solution).

Project to feasibility using the DP's own subtree values:

```text
for each region R where demand[R] > cap(R):
    for each resident L at R:
        marginal_value[L] = V(R, parent_state, s=1) - V(R, parent_state, s=0)
        // This is the full subtree impact of removing L at R,
        // including propagated effects on child regions.
    sort residents by marginal_value ascending
    while demand[R] > cap(R):
        remove the lowest-marginal-value resident
        recompute affected child DP entries
        demand[R] -= units(L)
```

`marginal_value` uses the DP's own `V` function, which accounts for the
subtree impact. A local that is cheap locally but stabilizes several child
regions will have high marginal value and survive the projection.

Status note:

- This is the main place where the current code differs from the original
  sketch. The implementation does a recursive per-region/bank feasible
  extraction using a capacity-constrained knapsack over the already-computed DP
  values, rather than a marginal-value projection pass.

### What emerges naturally

#### Low-pressure function (e.g. 10 locals, budget=22)

All locals have positive reward at `λ=0`. Capacity is never tight. Every local
is resident at root. No loop overrides (already resident everywhere).

Result: one stable set. Same as `ALGORITHM2`.

#### Hot loop with different locals

The DP finds high benefit inside the loop, low in the root. Transition cost is
amortized over many iterations. The solver adds a few locals at the loop
boundary and drops a few parent locals.

Result: root + loop-specific override. Same as `ALGORITHM3`, but derived.

#### Call-heavy loop

`call_tax` reduces reward for every resident. The solver naturally caches
fewer locals in call-heavy loops.

Result: smaller resident set near calls. No special logic.

#### Tiny cold loop

Low benefit (low freq). Transition cost dominates. Solver keeps parent state.

Result: inherits parent. No overhead.

#### x86_64 (budget=9) and ARMv7a (budget=8)

Smaller `cap(R)` -> higher `λ`. Fewer locals resident anywhere. But the
solver still allocates optimally given the budget.

On ARMv7a with `i64` locals: `units(L) = 2`, so each `i64` costs twice as much
capacity and twice as much transition cost. The solver naturally prefers `i32`
locals when capacity is tight.

### Integration with the pipeline

```text
1. cfg::build_semantic_cfg()             // unchanged
2. slot_ssa::lower_slot_only_ssa()       // unchanged
3. region_solver::solve()                // NEW: replaces joint_plan internals
   a. build region tree from Wasm loop structure
   b. compute benefit, call_tax, headroom, cap per (region, local)
   c. run DP iterations with dual price updates (3-8 rounds)
   d. extract final x[R,L] assignments
4. rewrite::rewrite_function()           // SIMPLIFIED
   - block_open: return region's public set
   - no tentative/finalize loop
   - no per-block cache selection
   - emit ensure/drop only at region transitions
5. cleanup::cleanup_program()            // mostly no-op
6. optimize::optimize_program()          // unchanged
7. sink_plan::plan_sinks()               // unchanged
```

The existing analysis infrastructure (`entry_region.rs` for access counts,
`cfg.rs` for loop structure, `pressure.rs` for live-transient computation)
provides the input data. The region solver replaces the per-block tentative
entry logic in `joint_plan/build.rs`.

Status note:

- The overall split above is mostly implemented.
- One detail is slightly different in the current code: rewrite still filters
  the emitted `block_entry_cached_slots` down to the subset the block actually
  needs to materialize or carry through, and then inserts edge repair blocks.
- Rewrite also still has mid-block pressure fallback drops.

### Lowering policy

#### Public state

At each block, the public state is `{L : x[Owner(B), L] = 1}`.

All blocks in the same region share the same public state. No per-block
variation.

#### Private flex promotion

Non-resident locals can be temporarily cached inside a block using spare
register capacity. Private promotions die before the terminator.

Status note:

- This private flex promotion policy is not yet implemented as described here.
- The current `local_access` policy is intentionally simpler: a local op uses
  cache form if the slot is already resident or if the block's solved public
  set includes it.

#### Pressure fallback

If a block's actual transient pressure exceeds headroom:

1. Spill cold deep stack values
2. Evict private flex promotions
3. Temporarily evict weakest public local (rare, restore before terminator)

Status note:

- Rewrite does implement pressure fallback and weakest-public-local eviction.
- Since private flex promotion is not yet implemented, the fallback currently
  operates on public cached locals plus transient values.

#### Region transitions

At edges where `Owner(pred) ≠ Owner(succ)`:

- `LocalDropCache` for locals in pred's state but not succ's state
- `LocalEnsureCache` for locals in succ's state but not pred's state

Emitted as:

- Inline at end of predecessor (if single successor)
- In a synthetic edge block (if needed for multi-predecessor targets)

Status note:

- This is implemented.
- The current rewrite also uses `LocalReserveCache` for write-first block
  entries that need the cache lane but not the incoming value.

### Complexity budget

| Phase | Cost |
| --- | --- |
| Region tree + inputs | `O(blocks × locals_accessed)` |
| Per DP iteration | `O(locals × regions)` |
| Price update | `O(regions)` |
| Total solver (I iterations) | `O(I × locals × regions)` |
| Final projection | `O(regions × locals × regions)` |
| Total | `O(blocks × L + I × L × R)` |

With `L=50`, `R=10`, `I=5`: solver = 2,500 ops. Negligible.
With `L=200`, `R=50`, `I=8`: solver = 80,000 ops. Still fast.

### Summary

1. Formulate public residency as cost minimization in one unit system
   (weighted frame-op equivalents): benefit minus call tax minus transition
   cost, subject to per-region capacity.
2. Solve on the Wasm region tree using per-local tree DP with capacity
   dual prices. Mismatch cost is symmetric and charges both entry and exit.
3. Lower each block with its region's solved public state. Private flex
   promotion handles non-resident locals within blocks.

Stability, loop overrides, call awareness, unit-cost sensitivity, and
small-budget behavior all emerge from the cost model.

## Lane Mapping: Order-Aware Public Cache Placement

### Problem

`ALGORITHM4.md` chooses the public resident set for each region:

- which locals are public-resident
- which locals are not

It does not choose the physical cache lane for each resident local.

That is a separate problem, and it matters.

Two regions can have almost the same resident set and still churn if shared
locals slide to different lanes.

Example:

```text
global deterministic order = [a, b, c]

parent resident set = {a, c}
child  resident set = {c}
```

If the implementation always compacts by order, then:

```text
parent lanes: a@0, c@1
child  lanes: c@0
```

`c` is resident on both sides, but still moves every time the edge executes.
That is churn.

So the real public state is not just:

```text
resident set
```

It is:

```text
resident set + lane map
```

### Current backend behavior

Today, the backend effectively imposes a deterministic compact order:

- middle IR carries `block_entry_cached_slots`
- machine lowering builds a global `cached_locals` order from first appearance
- each block's entry slots are sorted by that global order
- entry cache params are then assigned sequential dynamic registers

This is deterministic, but not edge-optimal.

It avoids arbitrary order noise, but it still causes avoidable moves when:

- a shared local stays resident across an edge
- an earlier-ordered local is dropped or added
- the shared local gets renumbered to a different lane by compaction

Status note:

- This section is now historical. The current machine backend has a dedicated
  cache-layout pass in `sf-nano-core/src/vm/machine/lower_cache_layout.rs`.
- That pass now does sticky inheritance, hole filling, and GP exact repack.
- What is still not yet implemented is feeding remap cost back into
  `ALGORITHM4`, plus the subtree-stickiness objective described below.

### Goal

Given:

- a region tree from `ALGORITHM4`
- a solved resident set `S[R]` for each region and bank

Choose, for each region `R`, a lane map:

```text
M[R, L] = concrete lane segment for local L
```

such that:

1. shared locals keep the same lane across region edges whenever possible
2. dropped locals free lanes without forcing remaining locals to slide
3. new locals fill free holes before moving shared locals
4. remap, when unavoidable, is done with register moves, not frame churn
5. GP/FP banks and 32-bit `i64` pair constraints are respected

### Key rule

Do not compact after drops.

If a local is shared across an edge and its old lane is still legal, keep it
there and leave holes behind dropped locals.

This is the main difference from the current compact-by-order behavior.

### Data model

Solve each bank independently.

For one bank:

- `K`: total physical dynamic lanes in that bank
- `S[R]`: locals resident in region `R`
- `w(L)`: width of local `L` in lane units
  - `1` for normal GP/FP values
  - `2` for GP `i64` on 32-bit
- `seg(i, w)`: contiguous lane interval `[i, i+w)`
- `M[R, L]`: assigned segment for `L` in region `R`

Feasibility:

- `M[R, L]` exists iff `L in S[R]`
- segments do not overlap
- `len(M[R, L]) = w(L)`
- GP `i64` on 32-bit must occupy a contiguous pair

### Cost model

`ALGORITHM4` already prices:

- add/drop membership changes
- call tax
- public-capacity pressure

Lane mapping adds a secondary cost: the extra register-move cost from
changing the lane of a local that is resident on both sides of an edge.

For one tree edge `P -> R`:

```text
extra_lane_cost(P -> R) =
    sum over L in (S[P] ∩ S[R]):
        edge_freq(P -> R) * move_cost(L) * [M[P,L] != M[R,L]]
```

Where:

```text
move_cost(L) = w(L)
```

One register move per unit is a good v1 approximation.

Important:

- `L in S[P] \ S[R]`: add no lane cost here; that drop is already priced by
  `ALGORITHM4`
- `L in S[R] \ S[P]`: add no lane cost here; that ensure/reserve is already
  priced by `ALGORITHM4`
- only shared locals that change lanes pay extra lane cost

#### Relationship to `ALGORITHM4`'s cost model

`ALGORITHM4`'s `edge_cost` does not include `extra_lane_cost`. The two cost
models are intentionally separate in v1:

- `ALGORITHM4` chooses resident sets assuming transitions cost
  `(entry_freq + exit_freq) * trans_cost(L)` per membership change.
- `LANE_MAPPING` minimizes the secondary lane-remap cost given those solved
  sets.

This means `ALGORITHM4` may approve a transition that is slightly more
expensive than it believes, because a shared local gets remapped. In practice,
sticky inheritance and no-compaction make such remaps rare.

If profiling later shows material remap cost, the fix is to add an estimated
remap penalty back into `ALGORITHM4`'s `edge_cost` function. That is a
tuning change, not an architectural one.

Status note:

- This remap cost is still not fed back into the middle-end solver.

### Output contract

Lane mapping is a second phase after `ALGORITHM4`:

1. `ALGORITHM4` chooses `S[R]`
2. lane mapping chooses `M[R, L]`

The mapping should live at machine-lowering level, not in the region solver.

That keeps responsibilities clean:

- `ALGORITHM4`: set selection
- `LANE_MAPPING`: physical placement

### Core algorithm

Use a top-down region-tree assignment with sticky inheritance.

#### Step 1: Root layout

Choose one concrete mapping for the root.

Root has no parent, so this is just a deterministic seed layout.

Recommended policy:

1. sort root residents by:
   - larger width first
   - larger `ALGORITHM4` root marginal value first
   - tie by slot id
2. place them into the lowest legal free segments

This does not need to be globally optimal. Once chosen, child regions inherit
from it, which is what removes churn.

Status note:

- The current machine implementation seeds layouts from the entry/root side and
  uses deterministic ordering, but it does not explicitly compute or use
  `ALGORITHM4` root marginal values.

#### Step 2: Child inherits parent lanes

For child region `R` with parent `P`, partition:

```text
Keep = S[P] ∩ S[R]
Drop = S[P] \ S[R]
Add  = S[R] \ S[P]
```

Start `M[R]` by copying every kept local into the same segment:

```text
for L in Keep:
    M[R, L] = M[P, L]
```

This is the default. Shared locals do not move unless there is a concrete
reason.

Dropped locals simply disappear, leaving holes.

#### Step 3: Fill holes with new locals

Assign new locals from `Add` into the currently free segments.

Policy:

1. place width-2 locals first
2. prefer exact-fit holes
3. then prefer best-fit holes
4. then prefer lower lane index for determinism

If all additions fit into holes, stop. No shared local moved.

#### Step 4: Rare micro-repack

If additions do not fit because of fragmentation, run a small exact search on
the affected bank.

This only happens when contiguity matters, mainly:

- GP `i64` on 32-bit requiring a contiguous pair (no even-parity alignment
  requirement in the current backend: `lower_context.rs` and
  `lower_regalloc.rs` only require two adjacent dynamic regs, not an
  even-aligned pair)
- multiple additions competing for fragmented holes

Search frontier:

- all `Add` locals
- only those `Keep` locals whose current segment overlaps a needed placement or
  blocks formation of a required contiguous hole

Do not repack the entire bank by default.

Objective for the frontier:

```text
minimize
    sum over L in Keep_frontier:
        edge_freq(P -> R) * move_cost(L) * [new_seg(L) != M[P,L]]
```

Secondary objective (recommended, not optional for v1):

- prefer keeping high-stickiness locals in place (see below)
- among equal-cost solutions, move less subtree-sticky locals first

Final tie-break:

- lower lane indices

Exhaustive search is acceptable. The hard-target GP bank sizes are tiny:
x86_64 = 9 lanes, ARMv7a = 8 lanes. Even an all-scalar worst case is at most
`8! = 40,320` candidate layouts. With pair constraints the feasible set is
smaller. In practice the frontier contains 2-4 locals, making the search
trivially cheap.

Status note:

- The current GP machine implementation does have an exact repack search.
- It minimizes moved shared-local cost and prefers parent-preserved placement,
  but it does not explicitly model subtree stickiness.
- FP currently does not have an analogous exact fragmented repack path.

### Subtree stickiness

When micro-repack must choose which shared local to move, it should prefer to
keep locals that remain resident through more of the subtree.

Define:

```text
stickiness(R, L) =
    total descendant edge frequency below R where L stays resident
```

High-stickiness locals are expensive to move because all descendants inherit
the moved lane. A move at region `R` does not directly force sibling regions
to move (siblings inherit from the parent, not from each other), but it does
propagate down into `R`'s descendants: every child, grandchild, etc. inherits
the new lane, potentially creating new fragmentation and remap cascades in the
subtree below.

Low-stickiness locals (resident only in `R` and few descendants) are better
move candidates.

This should be the default secondary objective in micro-repack, not an
optional tie-break. The common case (inherit + fill holes) does not need it,
but the rare fragmented case absolutely should consider subtree cost to avoid
cascading remap damage.

Status note:

- This subtree-stickiness objective is not explicitly implemented yet.

### Why holes are correct

Leaving a hole is not waste.

Example:

```text
parent: a@0, c@1
child:  c@1
```

Lane `0` is free in the child. That lane can still be used by transient values.
The child does not need to compact `c` down to lane `0`.

So:

- occupancy is sparse
- shared locals preserve identity
- transients use whatever dynamic lanes are free at the point of use

This is exactly what we want.

### Backend prerequisites

Sparse lane assignment is not an existing backend property. The current
entry-lane code is compact and sequential:

- `target_entry_cache_params()` in `lower_context.rs` sorts block entry
  slots by global `cached_locals` order and assigns sequential dynamic
  registers.
- `bind_cached_local_to_regs` and related functions assume a contiguous prefix of
  dynamic registers holds cached locals.

However, the generic edge protocol is already explicit-reg based, not
prefix-layout based. Edge stubs in `lower_module.rs`, `pipeline.rs`, and
`lower_inst.rs` already thread specific reserved regs and emit parallel
moves. So the real required change set is:

1. Replace compact entry assignment: `target_entry_cache_params()` must
   accept a sparse lane map instead of compacting by order.
2. Replace cache-binding assumptions: `bind_cached_local_to_regs` must bind to the
   lane specified by the map, not the next sequential register.
3. Transient allocation must skip occupied lanes: the allocator must treat
   cache-occupied lanes as unavailable, not assume they form a prefix.

The edge ABI itself does not need to change.

Status note:

- This prerequisites section is also now mostly historical.
- The current machine backend does implement sparse per-block cache layouts and
  threads explicit cache-entry params accordingly.
- What remains missing is mostly in the objective refinement and special-case
  search polish described elsewhere in this merged doc.

#### Required edge cases

When a shared local must change lanes, that should not become frame churn.
The machine layer needs cache remap support via register moves:

1. shared local, same lane: thread in-place with reserved-reg params.
   Zero extra work.
2. shared local, different lane: emit parallel register moves in the
   edge block. No frame load/store.
3. membership change: existing `Ensure/Reserve/Drop` behavior, already
   handled by `ALGORITHM4`.

Lane remap is not the same as membership repair.

#### Middle IR impact

No middle-IR lane annotation is required for v1.

Middle IR stays set-based:

- `block_entry_cached_slots`
- `LocalEnsureCache` / `LocalDropCache` / `LocalReserveCache`

Lane mapping is computed at the machine layer, once exact bank sizes, lane
widths, and the physical register file are known.

### Pseudocode

```text
solve_lane_mapping(region_tree, resident_sets):
    for bank in [gp, fp]:
        M[root, bank] = choose_root_layout(root, bank)
        assign_children(root, bank)

assign_children(parent, bank):
    for child in children(parent):
        M[child, bank] = inherit_kept_lanes(M[parent, bank], S[parent], S[child])

        if place_additions_into_holes(M[child, bank], S[child]):
            commit
        else:
            M[child, bank] = micro_repack(parent, child, bank)

        assign_children(child, bank)
```

`micro_repack(parent, child, bank)`:

```text
frontier = additions
         + kept locals blocking required placements

search all feasible assignments for frontier
minimize moved shared-local cost
keep non-frontier kept locals fixed
```

### Complexity

Common case:

- `O(|S[R]|)` per region
- just inherit + fill holes

Rare fragmented case:

- exact search over a tiny frontier
- frontier is usually very small because most locals remain fixed

This is cheap enough for a JIT because:

- lane count is small
- region count is small
- micro-repack is uncommon

### Relationship to `ALGORITHM4`

`ALGORITHM4` is still the right algorithm for choosing which locals should
be public-resident.

This document adds the missing second half:

- `ALGORITHM4`: resident-set optimization
- `LANE_MAPPING`: physical lane preservation

Without this phase, membership churn is reduced but not eliminated.
With this phase, stability extends to lane identity as well.

### Summary

The correct public-cache pipeline is:

1. solve resident sets by cost (`ALGORITHM4`)
2. assign concrete lanes per region with sticky inheritance
3. leave holes after drops instead of compacting
4. fill holes with additions
5. only remap shared locals when fragmentation makes it necessary
6. lower remaps as register moves, not frame reload/store churn

That is the missing piece needed to make public residency actually stable.
