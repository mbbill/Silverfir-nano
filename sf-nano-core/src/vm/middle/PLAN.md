# Middle Rewrite Plan

## Goal

Rewrite `middle/` from first principles so it does one job cleanly:

- take Wasm semantic IR
- emit explicit SSA-IR for `machine/`
- at the same time choose transient spills/fills and local-cache usage correctly

Correctness comes first. Optimization passes can be added later.

## Non-goals

The new `middle/` should not keep any of the old architecture around.

- no backward compatibility layers
- no function-level local-cache preference table
- no late pass that reconstructs block live-ins from already-lowered cache ops
- no duplicated ownership of boundary decisions across multiple passes

## Core model

The key idea is:

- transient SSA values and cached locals share one dynamic-bank budget
- cached locals are not SSA values
- but cached locals do cross block boundaries as explicit boundary state

For each bank, the invariant is:

`live transient SSA values + resident cached locals <= total dynamic budget`

This must hold at every program point and every block boundary.

## Boundary state

Each block boundary has two parts:

- `cached_locals`: resident local slots, in deterministic order
- `stack_values`: live transient SSA values, in stack order

Conceptually:

```text
BoundaryState =
  cached_locals: [slotA, slotB, slotN, ...]
  stack_values:  [v1, v2, v3, ...]
```

If dirty tracking is needed, it extends naturally:

```text
cached_locals: [(slotA, clean), (slotB, dirty), ...]
stack_values:  [v1, v2, v3, ...]
```

Transient stack values are determined by Wasm semantics. The part that requires policy is cached locals.

## High-level pipeline

### 1. Semantic IR -> explicit block CFG

Input:

- Wasm semantic IR with structured control

Output:

- explicit CFG with basic blocks and explicit edges

Properties:

- all merge points and loop headers are explicit
- semantic stack shape on each edge is known
- no cache decisions yet
- no spills/fills yet

### 2. Slot-only SSA lowering

Input:

- explicit semantic CFG

Output:

- slot-only SSA-form blocks

Properties:

- transient values become SSA values
- semantic `local.get/set/tee` become slot-form SSA-local ops
- no cache ops yet
- no spill/fill yet

At this stage locals are still only:

- `LocalGetSlot`
- `LocalSetSlot`

The starting point is deliberately conservative and policy-free.

### 3. In-block local-use scan

Input:

- slot-only SSA blocks

Output:

- per-block ranked local-use preference

This is not whole-function hotness. It is only block-local guidance.

For each block, compute an ordered local preference such as:

```text
preferred_locals(block) = [slot4, slot2, slot3, ...]
```

The order should favor locals that are used repeatedly or soon inside the block.

This scan is cheap and only helps choose the block entry cache state.

Only used locals in the block are in the list, so if a local is not in the list, it means the block doesn't touch it.

### 4. Choose canonical predecessor per block

Input:

- CFG
- basic block profile bias if available

Output:

- one canonical incoming edge for each non-entry block

Rules:

- single-predecessor block: that predecessor is canonical
- loop header: prefer the backedge as canonical, not the cold preheader
- ordinary merge: prefer the hotter predecessor

This is the key to keeping hot boundaries free.

### 5. One-pass joint block rewrite

This is the main transformation.

For each block, define its entry boundary using:

- exact incoming transient stack state from Wasm
- cached locals chosen from:
  - this block's preferred locals first
  - then carried cached locals from the canonical predecessor
  - clipped to the available dynamic budget after stack pressure

Then lower the block once from that entry state.

While processing each instruction:

- maintain current transient stack state
- maintain current cached-local state
- maintain current total dynamic usage
- rewrite local ops:
  - `LocalGetSlot` may become `LocalGetCache`
  - `LocalSetSlot` may become `LocalSetCache`
- insert:
  - `Spill`
  - `Fill`
  - `LocalEnsureCache`
  - `LocalDropCache`

Pressure handling policy:

- if there is room, keep hot locals cached
- if pressure is tight, first evict carried-through cached locals unused in this block
- then evict lower-priority cached locals
- if needed, spill lower-priority transient stack values

At block end, record the actual exit boundary:

- remaining cached locals
- remaining live transient stack values in stack order

For linear flow, that exit naturally becomes the next block's inherited state on the canonical path.

### 6. Repair non-canonical incoming edges

After each block has a chosen entry boundary and each predecessor has an exit boundary:

- if `pred.exit == succ.entry`, the edge is free
- otherwise insert a repair block on that edge

Repair blocks only do boundary matching:

- `LocalEnsureCache`
- `LocalDropCache`
- transient boundary repair if ever needed beyond the Wasm-fixed stack contract

The important rule is:

- block live-ins are not reconstructed from lowered cache ops
- they are already known from the chosen entry boundary
- repair only exists to match non-canonical edges to that known boundary

### 7. Optional cleanup

Not required for the first correct version.

Later cleanup can include:

- CFG simplification
- block merging
- jump threading
- unreachable block removal
- redundant `LocalEnsureCache` / `LocalDropCache` removal
- dead `Spill` / `Fill` cleanup
- local sink / direct-write cleanup

These passes must not change the ownership model above.

## Important design decisions

### No whole-function cache analysis

Whole-function local hotness is not part of the design.

It is misleading for local cache decisions because cache usefulness is region-local and CFG-local.

Only block-local local-use guidance is used.

### No late reconstruction of boundary state

This is a hard rule.

The system must never:

- lower a block first
- notice `LocalGetCache` / `LocalSetCache`
- reconstruct that the block must have wanted those locals cached at entry

That is backward.

Instead:

- choose the block entry boundary first
- then lower from that known state

### No global SSA solving

Wasm already fixes the transient stack contract across control flow.

The only real policy choice at boundaries is cached-local residency.

### No repeated full re-lowering loop

The design is intentionally one-pass over blocks after choosing:

- block-local preference order
- canonical predecessor
- block entry cache set

This is fast and already close to the end-state algorithm.

If a later version wants to improve predecessor choice or entry-cache selection, it should do so by replacing the selection heuristic, not by changing the entire architecture.

## Example shape

Starting from slot-only SSA:

```text
b0:
  v0 = ConstI32 0
  LocalSetSlot slot0, v0

  v1 = ConstI32 1
  LocalSetSlot slot1, v1

  Jump b1

b1:
  v2 = LocalGetSlot slot1
  ...
```

After the joint rewrite, if `slot0` and `slot1` are worth keeping cached through the loop:

```text
b0:
  v0 = ConstI32 0
  LocalSetSlot slot0, v0

  v1 = ConstI32 1
  LocalSetSlot slot1, v1

  Jump b0_b1_repair

b0_b1_repair:
  LocalEnsureCache slot0
  LocalEnsureCache slot1
  Jump b1

b1:
  v2 = LocalGetCache slot1
  ...
```

The hot loop backedge should match the chosen loop-header entry state directly, so it stays free.

## What `machine/` should receive

The output SSA-IR given to `machine/` should already be explicit and legal:

- explicit blocks and edges
- explicit SSA values
- explicit `Spill` / `Fill`
- explicit `LocalGetSlot`
- explicit `LocalSetSlot`
- explicit `LocalGetCache`
- explicit `LocalSetCache`
- explicit `LocalEnsureCache`
- explicit `LocalDropCache`
- explicit repair blocks where required

`machine/` should not need to rediscover local-cache policy.

## What remains swappable later

The following are algorithmic choices and can be improved later without changing the architecture:

- how block-local local-use preference is ranked
- how canonical predecessor is chosen at ordinary merges
- how many carried-through cached locals survive after preferred locals are packed
- which cached local to evict first under pressure
- which transient value to spill first under pressure
- profile weighting and loop bias

These are implementation heuristics, not architectural pieces.

## File structure proposal

The new `middle/` should be organized around clear ownership:

```text
middle/
  PLAN.md
  mod.rs
  frame.rs
  cfg.rs
  slot_ssa.rs
  rewrite.rs
  cleanup.rs
  joint_plan/
    mod.rs
    types.rs
    scan.rs
    canonical.rs
    naive.rs
    validate.rs
  ssa_ir/
    mod.rs
    ir.rs
    target.rs
    validate.rs
  tests.rs
```

Responsibilities:

- `mod.rs`
  orchestrates the whole pipeline
- `frame.rs`
  owns frame and slot layout
- `cfg.rs`
  turns structured semantic IR into explicit block CFG
- `slot_ssa.rs`
  lowers that CFG into slot-only SSA
- `rewrite.rs`
  owns the real lowering state and emits final SSA-IR
- `cleanup.rs`
  later post-lowering cleanup only
- `joint_plan/`
  owns policy and decisions
- `ssa_ir/`
  owns the final IR contract and validation

This structure keeps the core split explicit:

- `joint_plan/` decides
- `rewrite.rs` performs the rewrite and owns the facts

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
- block-local local-use ranking
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
- chosen entry cached-local set
- chosen exit cached-local set
- repair requirement for each non-canonical incoming edge

### Per-instruction outputs

At each relevant point, the planner should be able to answer:

- should this `LocalGet` use slot or cache
- should this `LocalSet` write slot or cache
- under pressure, should we drop a cached local or spill a transient value
- which cached locals should be admitted at block entry
- which cached locals should survive to block exit

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

The old system had a fixed transient window. The new system uses the full
dynamic-bank budget, but transient stack legality still needs an explicit plan.

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

So even with naive policy, `rewrite.rs` must never spill below the transient
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
- `LocalSetSlot` or `LocalSetCache`

This depends on current residency, pressure, and block-local local preference.

### Boundary-level planning responsibilities

These must also exist as explicit planner outputs, even if the initial policy
is simple.

#### 4. Block entry cache set

For each block, the planner must define which locals are expected to be cached
on entry.

This is chosen from:

- this block's own preferred locals
- carried cached locals from the canonical predecessor
- subject to the remaining dynamic budget after transient stack pressure

#### 5. Block exit cache set

For each block, the planner must define which cached locals survive to the
block boundary.

Those locals become candidates for carry-through to successor blocks.

#### 6. Edge repair

For each non-canonical incoming edge, the planner must define:

- which locals to `LocalEnsureCache`
- which locals to `LocalDropCache`

Repair exists only to match a known chosen entry boundary. It must never be
used to reconstruct what the block entry should have been.

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

The rewriter is the part that makes the program real. The planner only tells it what to do.

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
- optional cleanup later

The critical principle is:

- the entry boundary is known before lowering the block
- the block is lowered from that known boundary
- non-canonical edges repair into it

That is the clean replacement for the old `middle/`.
