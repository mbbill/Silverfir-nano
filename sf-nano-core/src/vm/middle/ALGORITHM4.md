# Algorithm 4: Cost-Optimal Public Residency via Region-Tree DP

## Problem

See PRESSURE.md for measured data.

The current per-block cache planner minimizes per-block frame access cost but
ignores transition cost at edges. The result is massive boundary churn.

Previous algorithms (ALGORITHM2: global set, ALGORITHM3: root + loop override)
treat stability as a structural constraint rather than a cost to optimize.
They work, but they are approximations of what the algorithm should really do:
minimize total cost.

## First-principles objective

For one register bank, let:

- `x[R,L] ∈ {0,1}`: local `L` is publicly resident in region `R`
- `units(L)`: register units consumed by `L` (i64 on 32-bit = 2, else 1)
- `benefit(R,L)`: weighted frame ops saved inside `R` if `L` is resident
- `call_tax(R,L)`: weighted cost of keeping `L` across calls inside `R`
- `entry_freq(R)`: estimated frequency of entering `R` from its parent
- `exit_freq(R)`: estimated frequency of leaving `R` to its parent
- `trans_cost(L)`: cost of one ensure or drop transition for `L` (= 1
  frame-op equivalent per unit)
- `cap(R)`: available register-unit capacity at `R` after transient headroom

All cost terms are in the same unit: **weighted frame-op equivalents**.
`benefit` and `call_tax` are pre-weighted by block frequency during
construction. The objective contains no additional `freq(R)` multiplier.

Objective:

```
maximize
    Σ(R,L) benefit(R,L) * x[R,L]
  - Σ(R,L) call_tax(R,L) * x[R,L]
  - Σ(R)   mismatch_cost(R)

subject to
    Σ_L units(L) * x[R,L] <= cap(R)     for every region R
```

Where `mismatch_cost(R)` charges for every residency change at the boundary
between region `R` and its parent:

```
mismatch_cost(R) =
    Σ_L edge_cost(x[parent(R),L], x[R,L], R, L)
```

For **loop regions**, residency changes cost on both entry and exit:

```
edge_cost(p, s, R_loop, L) =
    if p == s: 0
    else:      (entry_freq(R) + exit_freq(R)) * trans_cost(L)
```

This is symmetric: adding a local at a loop boundary costs the same as
removing one, because both require one op on entry and one on exit.

For the **root region**, there is no parent to restore on function return.
The only cost is one-time entry materialization:

```
edge_cost(0, s, R_root, L) =
    if s == 0: 0
    if s == 1: entry_freq(root) * trans_cost(L)
```

The root is always evaluated with parent state = 0.

## Why this formulation is right

- **Whole-function stability emerges** if transition costs dominate benefit
  for most locals. No stability constraint needed.
- **Loop-specific overrides emerge** when a local's in-loop benefit exceeds
  the transition cost at the loop boundary. No loop special-case needed.
- **Call cost is internalized** as a penalty on residency, not a structural
  barrier. Call-heavy loops naturally get smaller resident sets.
- **Capacity is a real constraint**, not a heuristic cutoff.
- **i64 unit cost** is built in via `units(L)`.
- **One unit system**: everything is weighted frame-op equivalents.

## Practical solver: region-tree DP with capacity prices

### Step 1: Build the region tree and compute inputs

Use the Wasm structured loop hierarchy directly. No SCC discovery needed.

```
Regions = { root } ∪ { one region per Wasm loop instruction }
parent(R) = enclosing loop, or root if top-level
OwnedBlocks(R) = blocks in R not inside any child loop
```

For each region `R`, compute from its owned blocks:

#### benefit(R, L)

```
benefit(R, L) = Σ B in OwnedBlocks(R):
    block_weight(B) * access_count(B, L)
```

`access_count(B, L)` = total `local.get` + `local.set` + `local.tee` of `L`
in block `B`. Each such access costs one frame op if uncached, zero if cached.

`block_weight(B)` = estimated execution frequency. Default: `10^loop_depth(B)`.

The result is in weighted-frame-op units. No further `freq(R)` multiplier is
applied when this term enters the objective.

#### call_tax(R, L)

```
call_tax(R, L) = Σ B in OwnedBlocks(R):
    block_weight(B) * calls_in_block(B) * keep_cost(L)
```

`keep_cost(L)` depends on the current call implementation:

| Strategy                    | keep_cost(L)      |
|-----------------------------|-------------------|
| Re-ensure after every call  | units(L)          |
| Callee-saved subset         | 0 if fits, else units(L) |
| Machine-level save/restore  | 0                 |

For v1, use `keep_cost(L) = units(L)` (re-ensure). Conservative and simple.

Already in weighted-frame-op units.

#### entry_freq(R) and exit_freq(R)

```
entry_freq(R) = block_weight(header(R)) / assumed_trip_count
exit_freq(R) = entry_freq(R)
```

For the root: `entry_freq = 1`.

`assumed_trip_count` is a tuning constant (default: 8). It only affects the
ratio between per-iteration benefit and per-entry transition cost. It does
not need to be accurate.

#### trans_cost(L)

```
trans_cost(L) = units(L)
```

One frame op per register unit per transition (ensure or drop).

#### cap(R): available cache capacity

```
cap(R) = dynamic_budget - headroom(R)
```

`headroom(R)` must account for the full live transient pressure at any point
inside the region, not just the stack-depth swing. This includes:

- carried entry-stack values that remain live inside the block
- block-internal computation results (leaf op outputs, local.get results for
  non-resident locals)
- values live across spill points

The existing rewriter tracks this via `fits_with_cached_locals` and
`count_live_bank_budget_units`. The region solver should use the same
computation, evaluated at the worst-case block within the region:

```
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

The headroom is **exact**, not softened. No upper clamp is applied. If one
pathological block in a region has very high transient pressure, the region's
capacity shrinks accordingly. The cost model then decides which locals are
still worth caching at the reduced capacity — a local that barely justified
residency will be priced out, which is the correct behavior.

The only floor is `MIN_HEADROOM` = 3 (worst-case single-op semantic
requirement), ensuring the solver never promises more capacity than exists.

If a rare block's actual transient pressure exceeds even the exact headroom
(due to estimation error or edge cases), the rewriter's pressure fallback
handles it: temporarily evict the weakest public local within that block and
restore before the terminator. This is a safety net, not a normal path.

### Step 2: Introduce capacity prices

For each region `R`, maintain a dual price `λ[R] >= 0` representing the
marginal cost of one register unit at `R`.

Define the price-adjusted reward for the DP:

```
reward(R, L) = benefit(R, L) - call_tax(R, L) - λ[R] * units(L)
```

All three terms are in weighted frame-op equivalents. The subtraction is
meaningful.

When `λ[R] = 0`, every local with `benefit > call_tax` wants residency.
As `λ[R]` rises, weaker locals get priced out. The price balances supply
(capacity) and demand (locals wanting residency).

### Step 3: Per-local tree DP (the core)

For fixed `λ`, each local `L` is an independent optimization on the region
tree.

The DP computes, for each region `R` and parent state `p ∈ {0,1}`, the
maximum net value of the subtree rooted at `R`:

```
DP[R, p] =
    max over s ∈ {0,1} of:
        V(R, p, s)

V(R, p, s) =
    reward(R, L) * s
  - edge_cost(p, s, R, L)
  + Σ child C of R: DP[C, s]
```

Edge cost depends on whether `R` is the root or a loop:

```
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

**Complexity**: O(regions) per local per iteration.

### Step 4: Update capacity prices

After solving all locals for the current `λ`:

```
demand[R] = Σ_L units(L) * x[R,L]
overload[R] = demand[R] - cap(R)
λ[R] = max(0, λ[R] + step * overload[R])
```

A reasonable step size: `step = 1.0 / cap(R)`.

Repeat steps 3-4 for a small number of iterations (3-8).

Convergence is fast because:
- The region tree is small (typically 5-20 nodes)
- Each local's DP is O(regions)
- Most locals have clear benefit rankings; few are on the margin

### Step 5: Final projection

After the last iteration, some regions may still slightly exceed capacity
(the Lagrangian relaxation provides an upper bound, not a feasible solution).

Project to feasibility using the DP's own subtree values:

```
for each region R where demand[R] > cap(R):
    for each resident L at R:
        marginal_value[L] = V(R, parent_state, s=1) - V(R, parent_state, s=0)
        // This is the full subtree impact of removing L at R,
        // including propagated effects on child regions.
    sort residents by marginal_value ascending
    while demand[R] > cap(R]:
        remove the lowest-marginal-value resident
        recompute affected child DP entries
        demand[R] -= units(L)
```

`marginal_value` uses the DP's own V function, which accounts for the
subtree impact. A local that is cheap locally but stabilizes several child
regions will have high marginal value and survive the projection.

## What emerges naturally

### Low-pressure function (e.g., 10 locals, budget=22)

All locals have positive reward at `λ=0`. Capacity is never tight. Every local
is resident at root. No loop overrides (already resident everywhere).

Result: one stable set. Same as ALGORITHM2.

### Hot loop with different locals

The DP finds high benefit inside the loop, low in the root. Transition cost is
amortized over many iterations. The solver adds a few locals at the loop
boundary and drops a few parent locals.

Result: root + loop-specific override. Same as ALGORITHM3, but derived.

### Call-heavy loop

`call_tax` reduces reward for every resident. The solver naturally caches
fewer locals in call-heavy loops.

Result: smaller resident set near calls. No special logic.

### Tiny cold loop

Low benefit (low freq). Transition cost dominates. Solver keeps parent state.

Result: inherits parent. No overhead.

### x86_64 (budget=9) and ARMv7a (budget=8)

Smaller `cap(R)` → higher `λ`. Fewer locals resident anywhere. But the
solver still allocates optimally given the budget.

On ARMv7a with i64 locals: `units(L) = 2`, so each i64 costs twice as much
capacity and twice as much transition cost. The solver naturally prefers i32
locals when capacity is tight.

## Integration with the pipeline

```
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

## Lowering policy

### Public state

At each block, the public state is `{L : x[Owner(B), L] = 1}`.

All blocks in the same region share the same public state. No per-block
variation.

### Private flex promotion

Non-resident locals can be temporarily cached inside a block using spare
register capacity. Private promotions die before the terminator.

### Pressure fallback

If a block's actual transient pressure exceeds headroom:

1. Spill cold deep stack values
2. Evict private flex promotions
3. Temporarily evict weakest public local (rare, restore before terminator)

### Region transitions

At edges where `Owner(pred) ≠ Owner(succ)`:

- `LocalDropCache` for locals in pred's state but not succ's state
- `LocalEnsureCache` for locals in succ's state but not pred's state

Emitted as:
- Inline at end of predecessor (if single successor)
- In a synthetic edge block (if needed for multi-predecessor targets)

## Complexity budget

| Phase | Cost |
|-------|------|
| Region tree + inputs | O(blocks × locals_accessed) |
| Per DP iteration | O(locals × regions) |
| Price update | O(regions) |
| Total solver (I iterations) | O(I × locals × regions) |
| Final projection | O(regions × locals × regions) |
| **Total** | **O(blocks × L + I × L × R)** |

With L=50, R=10, I=5: solver = 2,500 ops. Negligible.
With L=200, R=50, I=8: solver = 80,000 ops. Still fast.

## Summary

1. **Formulate** public residency as cost minimization in one unit system
   (weighted frame-op equivalents): benefit minus call tax minus transition
   cost, subject to per-region capacity.

2. **Solve** on the Wasm region tree using per-local tree DP with capacity
   dual prices. Mismatch cost is symmetric and charges both entry and exit.

3. **Lower** each block with its region's solved public state. Private flex
   promotion handles non-resident locals within blocks.

Stability, loop overrides, call awareness, unit-cost sensitivity, and
small-budget behavior all emerge from the cost model.
