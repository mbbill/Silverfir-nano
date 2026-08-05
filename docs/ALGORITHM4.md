# ALGORITHM4: Cost-Optimal Public Cache Residency for a WebAssembly JIT

## Abstract

> Vocabulary note (2026-07-13): the middle-end now calls the residency unit a
> **cell** (`CellId`) — the multi-use value slot the solver prices; wasm locals
> are one origin of cells, and a cell's frame home is a published property
> (`cell_homes`), not its identity. "Lane" below is unchanged: a physical
> register position in the bank. Older op names `Local*Cache`/`Local*Slot` are
> today's `Cell*Cache`/`Cell*Slot`.

This document specifies `ALGORITHM4` and its companion lane-mapping phase, the
middle-end public-cache allocator used by Silverfir-nano, a WebAssembly 2.0
JIT-only runtime. `ALGORITHM4` selects the set of locals that are publicly
resident in register lanes across each region of a function, minimizing a
single weighted-frame-op cost function that combines access benefit, call tax,
and boundary transition cost subject to per-region capacity. A second phase
assigns the chosen residents to physical lanes with sticky inheritance,
leaving holes after drops instead of compacting, and remapping only when
unavoidable.

The algorithmic components — hierarchical region-based allocation
[[CK91](#ck91)], region-based register promotion [[CL97](#cl97)], Lagrangian
relaxation on tree-structured problems [[Fis81](#fis81)], biased coloring
[[CACCHM81](#cacchm81), [BCT94](#bct94)], and lane placement with lifetime
holes [[PS99](#ps99)] — each have established prior art. The contribution of
this work is not a new algorithm but a *good fit*: Wasm's structured control
flow delivers the region tree for free, and JIT-scale problem sizes make
Lagrangian iteration cheap enough to run per-function without amortization.

> **Status (2026-07-12, commit 748c8416).** The Lagrangian price iterations
> (Steps 3–5 below) were removed after a two-target measurement: they were
> statically neutral on arm64 (net −9 native instructions across the 9-module
> corpus) and actively harmful in the register-scarce regime they were designed
> for (armv7 qemu coremark −3.99%). The shipped solver is the capacity-blind
> per-local tree DP (Steps 1–2, 4 with `λ ≡ 0`) followed directly by the
> feasibility projection (Step 6), with the projection's knapsack ordering
> items by descending resident-subtree potential so value ties commit ancestor
> continuity to the slot with the most downstream value — an ordering the price
> noise had masked the need for. The `algorithm4:iters=` policy parameter is
> gone. Sections describing prices are kept as the design record; see the
> design tree node `cache-residency/capacity-search` for the measurements.

## 1. Introduction

Silverfir-nano compiles WebAssembly through a three-stage pipeline: structured
IR (`wasm/`) → SSA-IR (`middle/`) → MachineIR (`machine/`) → native code
(`arch/`). Locals in WebAssembly are named frame slots; reading or writing a
local in the straight lowering costs one memory operation. A per-function
cache maps a subset of locals into dynamic register lanes so that subsequent
`local.get`, `local.set`, and `local.tee` on those locals become register
operations.

The question this document answers is: *which locals should occupy cache
lanes, in which regions of the function, and in which physical lanes?*

### 1.1 Why the naive approaches fail

A per-block cache planner that minimizes per-block frame access cost ignores
transition cost at edges. The result is boundary churn: locals are repeatedly
ensured and dropped as control flow crosses block boundaries that do not
agree on the resident set. Structural alternatives — fixing one global set,
or fixing a root set with loop-local overrides — treat stability as a
constraint rather than a cost to optimize. They are approximations of what
the allocator should do directly: minimize total cost.

### 1.2 Design rationale: why the formulation fits the setting

Two properties of the environment make this problem tractable in ways that
general register allocation is not.

**Structured control flow gives the region tree for free.** WebAssembly's
control-flow constructs (`block`, `loop`, `if`) are well-nested by
construction [[HRS+17](#hrs17)]. There is no SCC discovery, no irreducibility
handling, and no heuristic region formation. Every `loop` instruction defines
a region whose header, back edge, and exit set are immediate from the
syntactic structure. This is exactly the setting envisioned by
Callahan-Koblenz hierarchical allocation [[CK91](#ck91)], but delivered
without the analysis cost that general CFGs require.

**JIT budget makes Lagrangian iteration cheap.** The problem instance size is
small — a typical Wasm function has tens of locals and single-digit to low
tens of regions. A per-local tree DP with `O(regions)` work per iteration,
run for a fixed 8–12 subgradient rounds, costs a few thousand operations per
function. This is well below the per-function cost of instruction selection
and register allocation downstream. Lagrangian relaxation has been used in
compilers before [[AG01](#ag01)], but typically as the heavyweight branch of
a two-tier allocator that amortizes over long-running code; at JIT scale it
can be the only path.

Together, these mean the algorithm can be stated in the form a textbook would
expect — per-region capacity constraints, per-local tree DP, dual prices
updated by subgradient steps — without resorting to heuristic surrogates.

## 2. Problem Statement

### 2.1 Notation

For a single register bank, let:

| Symbol | Meaning |
| --- | --- |
| `x[R,L] ∈ {0,1}` | local `L` is publicly resident in region `R` |
| `units(L)` | register units consumed by `L` (`i64` on 32-bit = 2, else 1) |
| `benefit(R,L)` | weighted frame ops saved inside `R` if `L` is resident |
| `call_tax(R,L)` | weighted cost of keeping `L` across calls inside `R` |
| `entry_freq(R)` | estimated frequency of entering `R` from its parent |
| `exit_freq(R)` | estimated frequency of leaving `R` to its parent |
| `trans_cost(L)` | cost of one ensure or drop transition for `L` (one frame-op equivalent per unit) |
| `cap(R)` | available register-unit capacity at `R` after transient headroom |

All cost terms are in one unit: weighted frame-op equivalents. `benefit` and
`call_tax` are pre-weighted by block frequency during construction, so the
objective carries no additional `freq(R)` multiplier.

### 2.2 Objective

```text
maximize
    Σ(R,L) benefit(R,L) · x[R,L]
  - Σ(R,L) call_tax(R,L) · x[R,L]
  - Σ(R)   mismatch_cost(R)

subject to
    Σ_L units(L) · x[R,L] ≤ cap(R)     for every region R
```

where `mismatch_cost(R)` charges for every residency change at the boundary
between `R` and its parent:

```text
mismatch_cost(R) =
    Σ_L edge_cost(x[parent(R),L], x[R,L], R, L)
```

### 2.3 Edge costs

For loop regions, residency changes cost on both entry and exit:

```text
edge_cost(p, s, R_loop, L) =
    0                                            if p == s
    (entry_freq(R) + exit_freq(R)) · trans_cost(L)  otherwise
```

The symmetry is deliberate: adding a local at a loop boundary costs one op on
entry to ensure and one on exit to drop back to the parent state; removing
one is the reverse. Both cost the same.

For the root region there is no parent state to restore on function return
(the frame is discarded), so only entry materialization is charged:

```text
edge_cost(0, s, R_root, L) =
    0                                  if s == 0
    entry_freq(root) · trans_cost(L)   if s == 1
```

The root is always evaluated with parent state `p = 0`.

### 2.4 Why this formulation is right

The formulation has a property that the earlier approximations lacked: the
qualitative behaviors we want are not *designed in*, they *emerge* from
optimizing the cost.

- Whole-function stability emerges when transition cost dominates benefit
  for most locals. No stability constraint is needed.
- Loop-specific overrides emerge when in-loop benefit exceeds the loop
  boundary cost. No loop special case is needed.
- Call cost is internalized as a per-residency penalty, not a structural
  barrier. Call-heavy loops naturally get smaller resident sets.
- Capacity is a real constraint, not a heuristic cutoff.
- `i64` pair cost on 32-bit is internalized via `units(L)`.
- One unit system means `benefit − call_tax − λ · units` is dimensionally
  meaningful. Most RA cost models paper over unit mismatches.

## 3. Region Solver: Tree DP with Capacity Prices

The core solver is an alternation between a *primal* step (per-local tree DP
at fixed prices) and a *dual* step (subgradient update on capacity prices).
This is standard Lagrangian relaxation [[Fis81](#fis81)] specialized to the
region tree.

### 3.1 Step 1: Build the region tree

```text
Regions = { root } ∪ { one region per Wasm loop instruction }
parent(R) = enclosing loop, or root if top-level
OwnedBlocks(R) = blocks in R not inside any child loop
```

No SCC discovery is required; the structure is read off the Wasm decode. This
is the `CK91` tile tree, obtained without analysis.

### 3.2 Step 2: Compute inputs per region

**Benefit.**

```text
benefit(R, L) = Σ B ∈ OwnedBlocks(R):
    block_weight(B) · access_count(B, L)
```

`access_count(B, L)` is `local.get + local.set + local.tee` count for `L` in
block `B`. Each such access costs one frame op if uncached, zero if cached.
`block_weight(B) = 8^loop_depth(B)` by default.

**Call tax.**

```text
call_tax(R, L) = Σ B ∈ OwnedBlocks(R):
    block_weight(B) · calls_in_block(B) · keep_cost(L)
```

`keep_cost(L)` depends on the call protocol. The current v1 uses the
re-ensure model: `keep_cost(L) = units(L)`. Conservative and simple.

**Frequencies.**

```text
entry_freq(R) = block_weight(header(R)) / assumed_trip_count
exit_freq(R)  = entry_freq(R)
entry_freq(root) = 1
```

`assumed_trip_count = 8` is a tuning constant. It affects the ratio between
per-iteration benefit and per-entry transition cost, and nothing else; it
does not need to be accurate.

**Transition cost.** `trans_cost(L) = edge_scale * units(L)`. The backend
default is `edge_scale = 1.0`, except on x86_64 where benchmark tuning uses
`1.5` to account for the extra frame traffic and cache-repair instructions
caused by region-boundary residency changes. An explicit
`SF_CACHE_POLICY=algorithm4:edge=...` value replaces the backend default.

**Capacity.**

```text
cap(R) = dynamic_budget − headroom(R)
headroom(R) = max B ∈ OwnedBlocks(R): peak_live_transient_units(B)
```

`peak_live_transient_units(B)` counts, at the highest-pressure point in
block `B`, register units occupied by transient SSA values that are not
public-resident locals. It is computed from the semantic stack shapes and
slot-only SSA, reusing the infrastructure in `joint_plan/entry_region.rs` and
`joint_plan/pressure.rs`.

On x86_64 (budget = 9) or ARMv7a (budget = 8) this headroom eats a
significant fraction of the budget and naturally limits residency. The
headroom is exact, not softened. If one pathological block has high
transient pressure the region's capacity shrinks accordingly, and the cost
model decides which locals are still worth caching at the reduced capacity —
a local that barely justified residency will be priced out, which is the
correct behavior.

The only floor is `MIN_HEADROOM = 3`, the worst-case single-op semantic
requirement, so the solver never promises more capacity than exists.

If a rare block's actual transient pressure exceeds even the exact headroom
(edge cases, estimation error), the rewriter's pressure fallback handles it:
temporarily evict the weakest public local within that block and restore
before the terminator. This is a safety net, not a normal path.

*Status.* The current code computes exact per-bank peak live transient units
from `OpPlan.before` and `OpPlan.after` and subtracts those from the dynamic
bank budgets. `MIN_HEADROOM` is not literally implemented; the rewriter
covers the residual cases.

### 3.3 Step 3: Capacity prices

For each region `R`, maintain a dual price `λ[R] ≥ 0`, the marginal cost of
one register unit at `R`. Define the price-adjusted reward:

```text
reward(R, L) = benefit(R, L) − call_tax(R, L) − λ[R] · units(L)
```

When `λ[R] = 0`, every local with positive `benefit − call_tax` wants
residency. As `λ[R]` rises, weaker locals are priced out. The price balances
demand (locals wanting residency) against supply (capacity).

### 3.4 Step 4: Per-local tree DP

For fixed `λ`, each local `L` becomes an independent optimization on the
region tree. This is the key decomposition property of the Lagrangian
relaxation: dualizing the per-region capacity constraint separates the
locals, which otherwise compete for the same capacity.

```text
DP[R, p] = max over s ∈ {0,1} of V(R, p, s)

V(R, p, s) = reward(R, L) · s
           − edge_cost(p, s, R, L)
           + Σ child C of R: DP[C, s]
```

Solve bottom-up (leaves first). Extract decisions top-down starting from
`DP[root, p = 0]`.

Complexity: `O(regions)` per local per iteration.

### 3.5 Step 5: Subgradient price update

After solving all locals at the current `λ`:

```text
demand[R]   = Σ_L units(L) · x[R,L]
overload[R] = demand[R] − cap(R)
λ[R]        = max(0, λ[R] + step(iter, R) · overload[R])
step(iter,R)= (1.0 / (iter + 2)) / max(1, cap(R))
```

Repeat Steps 4–5 for a small number of iterations. Convergence is fast
because the region tree is small (typically 5–20 nodes), each local's DP is
`O(regions)`, and most locals have clear benefit rankings with few on the
margin.

The step is intentionally damped. With an undamped `1 / cap(R)` step, tiny
regions can oscillate between overfull and empty and the final iteration can
land on a zero-price state that ignores capacity competition.

*Status.* Removed 2026-07-12 (748c8416); see the note at the top. The
implementation had used a fixed `PRICE_ITERS = 12`.

### 3.6 Step 6: Projection to feasibility

After the last iteration some regions may still exceed capacity. Lagrangian
relaxation provides an upper bound on the primal, not a feasible solution,
so a final projection is needed:

```text
for each region R where demand[R] > cap(R):
    for each resident L at R:
        marginal_value[L] = V(R, parent_state, s=1) − V(R, parent_state, s=0)
    sort residents by marginal_value ascending
    while demand[R] > cap(R):
        remove the lowest-marginal-value resident
        recompute affected child DP entries
        demand[R] −= units(L)
```

`marginal_value` uses the DP's own `V`, so it accounts for the subtree
impact. A local that is cheap locally but stabilizes several child regions
has high marginal value and survives the projection.

*Status.* This is where the current code diverges most from the original
sketch. The implementation performs a recursive per-region/bank feasible
extraction using a capacity-constrained knapsack over the already-computed
DP values, instead of a marginal-value projection pass. The substitution is
an engineering convenience, not a material change to the algorithm.

### 3.7 Emergent behavior

Low-pressure function (10 locals, budget = 22). Every local has positive
reward at `λ = 0`. Capacity is never tight. All locals are resident at root,
no loop overrides. One stable global set, without asserting one.

Hot loop with different locals. High benefit inside the loop, low at the
root. Transition cost amortizes over many iterations. A few locals appear at
the loop boundary, a few parent locals drop. Root-plus-loop-override
behavior, derived rather than hard-coded.

Call-heavy loop. `call_tax` reduces reward for every resident. The solver
caches fewer locals near calls.

Tiny cold loop. Low frequency, transition cost dominates. Parent state
inherits; no override.

Tight budget (x86_64 = 9, ARMv7a = 8). Smaller `cap(R)` drives higher `λ`.
Fewer locals resident anywhere, but still optimally chosen. On ARMv7a with
`i64` locals, `units(L) = 2`, so each `i64` costs twice as much capacity and
twice as much transition cost; the solver naturally prefers `i32` locals
when capacity is tight.

## 4. Lane Mapping: Order-Aware Public Cache Placement

### 4.1 Problem

`ALGORITHM4` chooses the public resident set for each region; it does not
choose the physical cache lane for each resident local. That is a separate
problem, and it matters.

Two regions can have almost the same resident set and still churn if shared
locals slide to different lanes. Example:

```text
global deterministic order = [a, b, c]
parent resident set = {a, c}
child  resident set = {c}
```

If the backend compacts by order:

```text
parent lanes: a@0, c@1
child  lanes: c@0
```

`c` is resident on both sides but still moves every time the edge executes.
The real public state is not just *resident set* but *resident set + lane
map*.

### 4.2 Backend history

Earlier versions of this backend imposed a deterministic compact order: middle
IR carried `block_entry_cached_cells`, machine lowering built a global
`cached_locals` order from first appearance, and each block's entry slots
were sorted by that global order and assigned sequential dynamic registers.
Deterministic, but not edge-optimal.

*Status.* This history is now historical. The current machine backend has a
dedicated cache-layout pass in
`sf-nano-core/src/vm/jit/machine/lower_cache_layout.rs` implementing sticky
inheritance, hole filling, and GP exact repack. What remains unimplemented
is lane-remap cost feedback into `ALGORITHM4` and the subtree-stickiness
objective described in §4.6.

### 4.3 Goal

Given a region tree from `ALGORITHM4` and a solved resident set `S[R]` per
region and bank, choose for each region `R` a lane map `M[R, L]` = concrete
lane segment for local `L`, such that:

1. shared locals keep the same lane across region edges whenever possible;
2. dropped locals free lanes without forcing remaining locals to slide;
3. new locals fill free holes before moving shared locals;
4. unavoidable remap is performed with register moves, not frame churn;
5. GP/FP banks and 32-bit `i64` pair constraints are respected.

### 4.4 Key rule: do not compact after drops

If a local is shared across an edge and its old lane is still legal, keep it
there. Leave holes behind dropped locals. This is the main difference from
compact-by-order behavior, and it is the same policy used by linear-scan
allocators that track lifetime holes [[PS99](#ps99); [TWH98](#twh98)].

### 4.5 Data model

Solve each bank independently. For one bank:

- `K`: total physical dynamic lanes
- `S[R]`: locals resident in `R`
- `w(L)`: width of `L` in lane units (1 for normal GP/FP; 2 for GP `i64` on
  32-bit)
- `seg(i, w)`: contiguous lane interval `[i, i+w)`
- `M[R, L]`: assigned segment for `L` in `R`

Feasibility: `M[R, L]` exists iff `L ∈ S[R]`; segments do not overlap;
`len(M[R, L]) = w(L)`; GP `i64` on 32-bit occupies a contiguous pair.

### 4.6 Cost model

`ALGORITHM4` already prices add/drop membership changes, call tax, and
capacity pressure. Lane mapping adds a secondary cost: the extra
register-move cost from changing the lane of a local that is resident on
both sides of an edge.

For tree edge `P → R`:

```text
extra_lane_cost(P → R) =
    Σ L ∈ (S[P] ∩ S[R]):
        edge_freq(P → R) · move_cost(L) · [M[P,L] ≠ M[R,L]]
```

with `move_cost(L) = w(L)`. One register move per unit is a good v1
approximation.

Only shared locals that change lanes pay extra lane cost. Add and drop costs
are priced by `ALGORITHM4` and are not double-counted here.

#### 4.6.1 Relationship to `ALGORITHM4`'s cost model

`ALGORITHM4`'s `edge_cost` does not include `extra_lane_cost`. The two cost
models are intentionally separate in v1:

- `ALGORITHM4` chooses resident sets assuming transitions cost
  `(entry_freq + exit_freq) · trans_cost(L)` per membership change.
- Lane mapping minimizes the secondary lane-remap cost given those sets.

`ALGORITHM4` may therefore approve a transition that is slightly more
expensive than it believes, because a shared local gets remapped. In
practice, sticky inheritance and no-compaction make such remaps rare. If
profiling shows material remap cost, the fix is to add an estimated remap
penalty back into `ALGORITHM4`'s `edge_cost` — a tuning change, not an
architectural one.

*Status.* This feedback loop is not yet implemented.

### 4.7 Algorithm

Top-down region-tree assignment with sticky inheritance, a pattern used in
biased-preferencing graph allocators [[CACCHM81](#cacchm81);
[BCT94](#bct94)].

**Step 1: Root layout.** Choose a deterministic seed layout. Recommended:

1. sort root residents by (larger width first, larger `ALGORITHM4` root
   marginal value first, then slot id);
2. place them into the lowest legal free segments.

Global optimality at the root is not required — once chosen, child regions
inherit from it, which is what removes churn.

*Status.* The current implementation seeds layouts from the entry/root side
deterministically but does not explicitly compute or use `ALGORITHM4` root
marginal values.

**Step 2: Child inherits parent lanes.** For child `R` with parent `P`:

```text
Keep = S[P] ∩ S[R]
Drop = S[P] \ S[R]
Add  = S[R] \ S[P]
```

Start `M[R]` by copying every kept local into its parent segment:

```text
for L ∈ Keep: M[R, L] = M[P, L]
```

Dropped locals disappear, leaving holes.

**Step 3: Fill holes with new locals.** Assign `Add` into free segments:

1. place width-2 locals first;
2. prefer exact-fit holes;
3. then best-fit holes;
4. then lower lane index for determinism.

If all additions fit into holes, done.

**Step 4: Rare micro-repack.** If fragmentation blocks additions, run a
small exact search on the affected bank. This happens mainly for GP `i64`
pairs on 32-bit (the current backend requires two adjacent dynamic regs;
there is no even-parity alignment requirement — see `lower_context.rs` and
`lower_regalloc.rs`) and for multiple additions competing for fragmented
holes.

Frontier: all `Add` locals plus any `Keep` locals whose current segment
overlaps a needed placement or blocks formation of a required contiguous
hole. Do not repack the entire bank.

Objective:

```text
minimize
    Σ L ∈ Keep_frontier:
        edge_freq(P → R) · move_cost(L) · [new_seg(L) ≠ M[P,L]]
```

Secondary objective: prefer keeping high-stickiness locals in place (§4.8).
Tie-break on lower lane indices.

Exhaustive search is acceptable: x86_64 has 9 lanes, ARMv7a has 8. Even the
all-scalar worst case is at most `8! = 40320` candidate layouts, and pair
constraints shrink the feasible set further. In practice the frontier
contains 2–4 locals, making the search trivially cheap.

*Status.* The current GP machine implementation has an exact repack that
minimizes moved shared-local cost and prefers parent-preserved placement,
but does not explicitly model subtree stickiness. FP does not have an
analogous fragmented repack path.

### 4.8 Subtree stickiness

When micro-repack must choose which shared local to move, it should prefer
to keep locals that remain resident through more of the subtree. Define:

```text
stickiness(R, L) =
    total descendant edge frequency below R where L stays resident
```

High-stickiness locals are expensive to move because all descendants inherit
the moved lane. A move at `R` does not directly force siblings to move
(siblings inherit from the parent), but it propagates into `R`'s
descendants, potentially creating cascading fragmentation and remap damage.
Low-stickiness locals (resident only in `R` and few descendants) are better
move candidates. This should be the default secondary objective in
micro-repack, not an optional tie-break.

*Status.* Not explicitly implemented yet.

### 4.9 Why holes are correct

Leaving a hole is not waste. Example:

```text
parent: a@0, c@1
child:  c@1
```

Lane 0 is free in the child and can still be used by transient values. The
child does not need to compact `c` down to lane 0. Occupancy is sparse;
shared locals preserve identity; transients use whatever dynamic lanes are
free at the point of use. This is the behavior we want.

### 4.10 Backend support

Sparse lane assignment was not originally a backend property. The edge
protocol is already explicit-reg based (edge stubs in `lower_module.rs`,
`pipeline.rs`, `lower_inst.rs` thread specific reserved regs and emit
parallel moves), so the required changes were:

1. `target_entry_cache_params()` accepts a sparse lane map instead of
   compacting by order.
2. `bind_cached_local_to_regs` binds to the map-specified lane, not the
   next sequential register.
3. Transient allocation skips occupied lanes instead of assuming a prefix
   layout.

The edge ABI itself did not change.

*Status.* The current machine backend implements sparse per-block cache
layouts and threads explicit cache-entry params accordingly.

#### 4.10.1 Required edge cases

1. Shared local, same lane: thread in-place with reserved-reg params. Zero
   extra work.
2. Shared local, different lane: emit parallel register moves in the edge
   block. No frame load/store.
3. Membership change: `Ensure/Reserve/Drop`, priced by `ALGORITHM4`.

Lane remap is not the same as membership repair. Boissinot et al.'s
out-of-SSA translation work [[BDR+09](#bdr09)] covers the parallel-copy
lowering on which step 2 rests.

#### 4.10.2 Middle IR impact

No middle-IR lane annotation is required for v1. Middle IR stays set-based
(`block_entry_cached_cells`; `CellEnsureCache` / `CellDropCache` /
`CellReserveCache`). Lane mapping is computed at the machine layer, once
exact bank sizes, lane widths, and the physical register file are known.

### 4.11 Pseudocode

```text
solve_lane_mapping(region_tree, resident_sets):
    for bank in [gp, fp]:
        M[root, bank] = choose_root_layout(root, bank)
        assign_children(root, bank)

assign_children(parent, bank):
    for child in children(parent):
        M[child, bank] = inherit_kept_lanes(M[parent, bank],
                                            S[parent], S[child])
        if place_additions_into_holes(M[child, bank], S[child]):
            commit
        else:
            M[child, bank] = micro_repack(parent, child, bank)
        assign_children(child, bank)
```

## 5. Lowering Policy

### 5.1 Public state

At each block the public state is `{ L : x[Owner(B), L] = 1 }`. All blocks
in the same region share the same public state. No per-block variation.

### 5.2 Private flex promotion

Non-resident locals can be temporarily cached inside a block using spare
register capacity. Private promotions die before the terminator.

*Status.* Not yet implemented as described. The current `local_access`
policy is intentionally simpler: a local op uses cache form if the slot is
already resident or if the block's solved public set includes it.

### 5.3 Pressure fallback

If a block's actual transient pressure exceeds headroom:

1. spill cold deep stack values;
2. evict private flex promotions;
3. temporarily evict the weakest public local (rare; restore before
   terminator).

*Status.* Rewrite implements pressure fallback and weakest-public-local
eviction. Since private flex promotion is not yet implemented, the fallback
currently operates on public cached locals plus transient values.

### 5.4 Region transitions

At edges where `Owner(pred) ≠ Owner(succ)`:

- `CellDropCache` for locals in pred's state but not succ's state;
- `CellEnsureCache` for locals in succ's state but not pred's state.

Emitted either inline at the end of the predecessor (single successor) or in
a synthetic edge block (multi-predecessor targets).

*Status.* Implemented. The current rewrite also uses `CellReserveCache`
for write-first block entries that need the cache lane but not the incoming
value.

## 6. Pipeline Integration

```text
1. cfg::build_semantic_cfg()
2. slot_ssa::lower_slot_only_ssa()
3. region_solver::solve()                // REPLACES per-block tentative logic
   a. build region tree from Wasm loop structure
   b. compute benefit, call_tax, headroom, cap per (region, local)
   c. run DP iterations with dual price updates
   d. extract final x[R,L]
4. rewrite::rewrite_function()           // SIMPLIFIED
   - block_open returns the region's public set
   - no tentative/finalize loop
   - no per-block cache selection
   - ensure/drop only at region transitions
5. cleanup::cleanup_program()
6. optimize::optimize_program()
7. sink_plan::plan_sinks()
```

The existing analysis (`entry_region.rs` for access counts, `cfg.rs` for
loop structure, `pressure.rs` for live-transient counts) provides the input
data. The region solver replaces the per-block tentative entry logic in
`joint_plan/build.rs`.

*Status.* The split above is largely implemented. Two residual details:

1. Rewrite still filters `block_entry_cached_cells` down to the subset the
   block actually needs and inserts edge repair blocks;
2. Rewrite retains mid-block pressure fallback drops.

## 7. Complexity

| Phase | Cost |
| --- | --- |
| Region tree + inputs | `O(blocks × locals_accessed)` |
| Per DP iteration | `O(locals × regions)` |
| Price update | `O(regions)` |
| Total solver (I iterations) | `O(I × locals × regions)` |
| Final projection | `O(regions × locals × regions)` |
| Total | `O(blocks · L + I · L · R)` |

With `L = 50, R = 10, I = 5`: solver ≈ 2,500 ops. Negligible.
With `L = 200, R = 50, I = 8`: solver ≈ 80,000 ops. Still fast.

This fits well within a JIT budget and does not require amortization.

## 8. Implementation Status Summary

Implemented:

- Region solver `region_solver.rs` (resident-set selection with Lagrangian
  DP).
- Edge repair at region transitions (`CellEnsureCache`, `CellDropCache`,
  `CellReserveCache`).
- Sparse per-block cache layouts and explicit cache-entry params.
- Machine-side lane-mapping pass `lower_cache_layout.rs` with sticky
  inheritance, hole filling, and GP exact repack.
- Rewrite-time pressure fallback with weakest-public-local eviction.

Not yet implemented:

- Private flex promotion for non-resident locals.
- Lane-remap cost feedback into `ALGORITHM4`'s residency objective.
- Subtree-stickiness as an explicit objective in machine-side micro-repack.
- Exact fragmented FP micro-repack analogous to the GP exact search.
- Literal `MIN_HEADROOM = 3` floor (rewriter covers the residual cases).

## 9. Related Work

### 9.1 Hierarchical and region-based register allocation

The closest overall architectural precedent is Callahan and Koblenz's
hierarchical graph coloring [[CK91](#ck91)], which builds a tile tree from
loop nesting and allocates hierarchically, using preferencing to bias colors
at tile boundaries. A later study [[CK05](#ck05)] compares it with
Chaitin-Briggs. Lueh, Gross, and Adl-Tabatabai's graph-fusion allocator
[[LGA00](#lga00)] is another region-based approach that uses program
structure to guide splitting and spilling decisions.

`ALGORITHM4` adopts the tile-tree idea but replaces per-tile graph coloring
with per-local tree DP coupled through Lagrangian capacity prices. The
dimensions swap: Callahan-Koblenz is per-tile (solve all locals at a tile,
propagate across tiles), whereas this work is per-local (solve one local on
the whole tree at fixed prices, iterate to balance capacity).

### 9.2 Register promotion

Cooper and Lu [[CL97](#cl97)] address a problem very close to this one —
deciding when to keep memory values in registers across program regions —
using SSA and dominance analysis rather than optimization. Their region
shape and the transition-cost intuition are similar; the mechanism differs.

### 9.3 SSA-based register allocation

Hack, Grund, and Goos [[HGG06](#hgg06)] observe that SSA interference graphs
are chordal and decouple coloring, spilling, and coalescing. This work
inherits that spirit in separating set selection (§3) from lane placement
(§4), though on a different substrate.

### 9.4 Lagrangian relaxation and ILP approaches

Lagrangian relaxation of integer programs is classical [[Fis81](#fis81)].
Its application to compiler problems is less common, but Appel and George's
optimal spilling work [[AG01](#ag01)] shows that ILP-based allocation can
be practical. This work differs in two ways: (i) it uses Lagrangian
relaxation rather than full ILP, and (ii) the relaxation decomposes the
problem along the *local* axis so each subproblem is a tree DP, making
primal recovery trivially fast. The fit is exactly what the JIT setting
needs.

### 9.5 Tree DP for register decisions

The Sethi-Ullman algorithm [[SU70](#su70)] is the foundational tree DP for
register allocation on expression trees, generalized by Appel and Supowit
[[AS87](#as87)]. `ALGORITHM4`'s per-local tree DP operates on a different
tree (regions, not expressions) and a different per-node decision (residency,
not evaluation order), but shares the pattern.

### 9.6 Linear scan and lifetime holes

Poletto and Sarkar's linear scan [[PS99](#ps99)] is the canonical fast
allocator; Traub, Holloway, and Smith [[TWH98](#twh98)] extend it with
lifetime holes. The "don't compact after drops, fill holes later" policy
in §4 is the same idea.

### 9.7 Biased preferencing and SSA destruction

Chaitin et al.'s original graph coloring allocator [[CACCHM81](#cacchm81)]
and Briggs, Cooper, and Torczon's refinements [[BCT94](#bct94)] pioneered
coalescing and biasing for move minimization across moves and copies.
Boissinot and collaborators' work on liveness and out-of-SSA translation
[[BHM+08](#bhm08); [BDR+09](#bdr09)] covers the parallel-copy lowering
required at the machine layer for lane remaps.

### 9.8 WebAssembly baseline compilation

Haas et al. describe the Wasm design rationale [[HRS+17](#hrs17)]. V8's
Liftoff [[V8L18](#v8l18)] and Titzer's survey of baseline compilers
[[Tit23](#tit23)] establish the context in which this work sits: structured
CFG, single-pass compilation, and the register-allocation-in-baseline
tradition that motivates treating the public-cache problem as a
self-contained middle-end phase.

### 9.9 What is new

The specific Lagrangian decomposition that makes each local an independent
tree DP problem on a Wasm region tree, with asymmetric root-vs-loop boundary
costs and an explicit subtree-stickiness tie-breaker in lane mapping, is —
to our knowledge — novel in combination. Each ingredient has prior art; the
assembled recipe does not. The practical contribution is more accurately
described as a fit argument: Wasm's structured CFG and the JIT cost budget
make an otherwise textbook formulation directly deployable.

## 10. Summary

1. Public residency is formulated as cost minimization in one unit system
   (weighted frame-op equivalents): benefit minus call tax minus transition
   cost, subject to per-region capacity.
2. The problem is solved on the Wasm region tree using per-local tree DP
   with capacity dual prices. Mismatch cost is symmetric and charges both
   entry and exit at loop boundaries; root pays only one-time entry
   materialization.
3. Lane mapping is a second phase: sticky inheritance from the parent, holes
   after drops, fill holes with additions, micro-repack with register moves
   only when fragmentation forces it.
4. Stability, loop overrides, call awareness, unit-cost sensitivity, and
   small-budget behavior all emerge from the cost model rather than being
   hard-coded.

The algorithm is standard in its components and uncommon only in its fit:
the region tree is free, the subproblems are small, and the Lagrangian
iteration is cheap enough for a JIT. That fit is the contribution.

## References

<a id="ag01"></a>
**[AG01]** Andrew W. Appel and Lal George. *Optimal spilling for CISC
machines with few registers.* PLDI 2001, pp. 243–253.

<a id="as87"></a>
**[AS87]** Andrew W. Appel and Kenneth J. Supowit. *Generalizations of the
Sethi-Ullman algorithm for register allocation.* Software: Practice and
Experience, 17(6):417–421, 1987.

<a id="bct94"></a>
**[BCT94]** Preston Briggs, Keith D. Cooper, and Linda Torczon.
*Improvements to graph coloring register allocation.* ACM TOPLAS
16(3):428–455, 1994.

<a id="bdr09"></a>
**[BDR+09]** Benoit Boissinot, Alain Darte, Fabrice Rastello, Benoît Dupont
de Dinechin, and Christophe Guillon. *Revisiting out-of-SSA translation for
correctness, code quality, and efficiency.* CGO 2009.

<a id="bhm08"></a>
**[BHM+08]** Benoit Boissinot, Sebastian Hack, Daniel Grund, Benoît Dupont
de Dinechin, and Fabrice Rastello. *Fast liveness checking for SSA-form
programs.* CGO 2008.

<a id="cacchm81"></a>
**[CACCHM81]** Gregory J. Chaitin, Marc A. Auslander, Ashok K. Chandra, John
Cocke, Martin E. Hopkins, and Peter W. Markstein. *Register allocation via
coloring.* Computer Languages 6(1):47–57, 1981.

<a id="ck91"></a>
**[CK91]** David Callahan and Brian Koblenz. *Register allocation via
hierarchical graph coloring.* PLDI 1991, pp. 192–203.

<a id="ck05"></a>
**[CK05]** Keith D. Cooper, Anshuman Dasgupta, and Jason Eckhardt.
*Revisiting graph coloring register allocation: a study of the
Chaitin-Briggs and Callahan-Koblenz algorithms.* LCPC 2005.

<a id="cl97"></a>
**[CL97]** Keith D. Cooper and John Lu. *Register promotion in C programs.*
PLDI 1997.

<a id="fis81"></a>
**[Fis81]** Marshall L. Fisher. *The Lagrangian relaxation method for
solving integer programming problems.* Management Science 27(1):1–18, 1981.

<a id="hgg06"></a>
**[HGG06]** Sebastian Hack, Daniel Grund, and Gerhard Goos. *Register
allocation for programs in SSA-form.* Compiler Construction 2006 (LNCS
3923), pp. 247–262.

<a id="hrs17"></a>
**[HRS+17]** Andreas Haas, Andreas Rossberg, Derek L. Schuff, Ben L.
Titzer, Michael Holman, Dan Gohman, Luke Wagner, Alon Zakai, and JF
Bastien. *Bringing the web up to speed with WebAssembly.* PLDI 2017.

<a id="lga00"></a>
**[LGA00]** Guei-Yuan Lueh, Thomas Gross, and Ali-Reza Adl-Tabatabai.
*Fusion-based register allocation.* ACM TOPLAS 22(3):431–470, 2000.

<a id="ps99"></a>
**[PS99]** Massimiliano Poletto and Vivek Sarkar. *Linear scan register
allocation.* ACM TOPLAS 21(5):895–913, 1999.

<a id="su70"></a>
**[SU70]** Ravi Sethi and Jeffrey D. Ullman. *The generation of optimal
code for arithmetic expressions.* Journal of the ACM 17(4):715–728, 1970.

<a id="tit23"></a>
**[Tit23]** Ben L. Titzer. *Whose baseline compiler is it anyway?* arXiv
preprint arXiv:2305.13241, 2023.

<a id="twh98"></a>
**[TWH98]** Omri Traub, Glenn Holloway, and Michael D. Smith. *Quality and
speed in linear-scan register allocation.* PLDI 1998.

<a id="v8l18"></a>
**[V8L18]** Clemens Backes et al. *Liftoff: a new baseline compiler for
WebAssembly in V8.* V8 blog, 2018.
