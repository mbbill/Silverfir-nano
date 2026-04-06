# Lane Mapping: Order-Aware Public Cache Placement

## Problem

`ALGORITHM4.md` chooses the **public resident set** for each region:

- which locals are public-resident
- which locals are not

It does **not** choose the **physical cache lane** for each resident local.

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

## Current backend behavior

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

## Goal

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

## Key rule

**Do not compact after drops.**

If a local is shared across an edge and its old lane is still legal, keep it
there and leave holes behind dropped locals.

This is the main difference from the current compact-by-order behavior.

## Data model

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

## Cost model

`ALGORITHM4` already prices:

- add/drop membership changes
- call tax
- public-capacity pressure

Lane mapping adds a secondary cost: the **extra register-move cost** from
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
- only **shared locals that change lanes** pay extra lane cost

### Relationship to `ALGORITHM4`'s cost model

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

## Output contract

Lane mapping is a second phase after `ALGORITHM4`:

1. `ALGORITHM4` chooses `S[R]`
2. lane mapping chooses `M[R, L]`

The mapping should live at machine-lowering level, not in the region solver.

That keeps responsibilities clean:

- `ALGORITHM4`: set selection
- `LANE_MAPPING`: physical placement

## Core algorithm

Use a top-down region-tree assignment with sticky inheritance.

### Step 1: Root layout

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

### Step 2: Child inherits parent lanes

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

### Step 3: Fill holes with new locals

Assign new locals from `Add` into the currently free segments.

Policy:

1. place width-2 locals first
2. prefer exact-fit holes
3. then prefer best-fit holes
4. then prefer lower lane index for determinism

If all additions fit into holes, stop. No shared local moved.

### Step 4: Rare micro-repack

If additions do not fit because of fragmentation, run a small exact search on
the affected bank.

This only happens when contiguity matters, mainly:

- GP `i64` on 32-bit requiring a contiguous pair (no even-parity alignment
  requirement in the current backend — `lower_context.rs` and
  `lower_regalloc.rs` only require two adjacent dynamic regs, not an
  even-aligned pair)
- multiple additions competing for fragmented holes

Search frontier:

- all `Add` locals
- only those `Keep` locals whose current segment overlaps a needed placement or
  blocks formation of a required contiguous hole

Do **not** repack the entire bank by default.

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

**Exhaustive search is acceptable.** The hard-target GP bank sizes are tiny:
x86_64 = 9 lanes, ARMv7a = 8 lanes. Even an all-scalar worst case is at most
`8! = 40,320` candidate layouts. With pair constraints the feasible set is
smaller. In practice the frontier contains 2-4 locals, making the search
trivially cheap.

## Subtree stickiness

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

This should be the **default secondary objective** in micro-repack, not an
optional tie-break. The common case (inherit + fill holes) does not need it,
but the rare fragmented case absolutely should consider subtree cost to avoid
cascading remap damage.

## Why holes are correct

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

## Backend prerequisites

Sparse lane assignment is **not** an existing backend property. The current
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

1. **Replace compact entry assignment**: `target_entry_cache_params()` must
   accept a sparse lane map instead of compacting by order.
2. **Replace cache-binding assumptions**: `bind_cached_local_to_regs` must bind to the
   lane specified by the map, not the next sequential register.
3. **Transient allocation must skip occupied lanes**: the allocator must treat
   cache-occupied lanes as unavailable, not assume they form a prefix.

The edge ABI itself does not need to change.

### Required edge cases

When a shared local must change lanes, that should not become frame churn.
The machine layer needs cache remap support via register moves:

1. **shared local, same lane** — thread in-place with reserved-reg params.
   Zero extra work.
2. **shared local, different lane** — emit parallel register moves in the
   edge block. No frame load/store.
3. **membership change** — existing `Ensure/Reserve/Drop` behavior, already
   handled by `ALGORITHM4`.

Lane remap is not the same as membership repair.

### Middle IR impact

No middle-IR lane annotation is required for v1.

Middle IR stays set-based:

- `block_entry_cached_slots`
- `LocalEnsureCache` / `LocalDropCache` / `LocalReserveCache`

Lane mapping is computed at the machine layer, once exact bank sizes, lane
widths, and the physical register file are known.

## Pseudocode

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

## Complexity

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

## Relationship to `ALGORITHM4`

`ALGORITHM4` is still the right algorithm for choosing **which** locals should
be public-resident.

This document adds the missing second half:

- `ALGORITHM4`: resident-set optimization
- `LANE_MAPPING`: physical lane preservation

Without this phase, membership churn is reduced but not eliminated.
With this phase, stability extends to lane identity as well.

## Summary

The correct public-cache pipeline is:

1. solve resident sets by cost (`ALGORITHM4`)
2. assign concrete lanes per region with sticky inheritance
3. leave holes after drops instead of compacting
4. fill holes with additions
5. only remap shared locals when fragmentation makes it necessary
6. lower remaps as register moves, not frame reload/store churn

That is the missing piece needed to make public residency actually stable.
