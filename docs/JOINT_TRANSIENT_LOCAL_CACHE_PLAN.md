# Joint Transient And Local-Cache Planning Plan

## Goals

- Share one total dynamic register budget between transient SSA values and
  cached locals.
- Preserve the semantic split:
  - transients stay linear SSA values;
  - cached locals stay mutable slot-backed state.
- Make the spill/cache decision in `middle/`, not in `machine/`.
- Allow the transient/cache ratio to change at arbitrary program points,
  including mid-block.
- Keep physical register choice in `machine/`.

## Non-Goals

- Do not make cached locals semantically identical to transient SSA values.
- Do not introduce cache-position or physical-register ids into SSA-IR.
- Do not rely on a fixed per-function, per-region, or per-block split between
  transient lanes and cached locals.
- Do not turn `machine/` into the policy owner for cache eviction.
- Do not keep a second "logical locals" SSA-IR vocabulary alongside the final
  explicit one.

## Best Design

The best design is:

- one SSA-IR interface;
- explicit local operations from the start;
- no `LocalGet` / `LocalSet` in SSA-IR;
- no local versioning in SSA-IR;
- one joint planner in `middle/` that decides both transient and local
  cache behavior while semantic control-flow preparation still has exact stack
  shape;
- boundary repair policy in `middle/`, not in `machine/`;
- `machine/` only realizes the explicit plan.

This means the architectural split is not "two middle IRs". It is:

1. semantic/control preparation with joint transient+cache planning;
2. SSA lowering using that already-final explicit vocabulary;
3. light post-passes that must not destroy the planned stack shape.

The important distinction is:

- there may be multiple passes in `middle/`;
- there should not be two different local-operation interfaces exposed through
  `ssa_ir`.

## Current Pipeline Constraints

The current pipeline cannot support joint planning because transient budgeting
is committed too early, and local-cache policy still leaks through hidden
interfaces.

Current constraints:

- `sf-nano-core/src/vm/middle/spill_plan.rs`
  - `prepare_semantic_ops()` plans `Spill` / `Fill` against fixed
    `gp_transient_budget` and `fp_transient_budget`;
  - `spill_before_result_push()` assumes that if a transient lane exists, SSA
    lowering is allowed to use it.
- `sf-nano-core/src/vm/middle/state.rs`
  - `BlockState::ensure_live_fit()` validates only transient SSA pressure
    against the fixed transient budgets.
- `sf-nano-core/src/vm/middle/lower_ops.rs`
  - lowers semantic locals through `LocalGet` / `LocalSet`;
  - carries `local_versions`;
  - populates `ValueHome::LocalVersion`.
- `sf-nano-core/src/vm/middle/optimize.rs`
  - `forward_slot_values()` assumes canonical slot-local semantics and
    currently treats cache ops as barriers.
- `sf-nano-core/src/vm/middle/sink_plan.rs`
  - sink planning is still written against `LocalSet { slot, src, version }`.
- `sf-nano-core/src/vm/middle/local_cache_explicit.rs`
  - is only a migration pass;
  - it rewrites logical locals into explicit cache ops after the main middle
    passes have already run.
- `sf-nano-core/src/vm/machine/`
  - still consumes cache information through machine-facing metadata that was
    originally chosen before joint planning existed.

As long as `spill_plan.rs` uses a fixed transient budget before late planning
exists, "reuse leftover transient lanes for cache" is not a real joint plan.

## Required Architectural Change

The critical change is to make the joint decision while the middle-end still
has exact operand-stack shape, not after later SSA rewrites have removed it.

In practice that means:

1. `spill_plan.rs` becomes the joint transient+cache planner;
2. lowering emits the final explicit local/cache ops directly from that plan;
3. CFG-aware boundary planning chooses one canonical entry state per block for
   both transients and cached locals;
4. repair blocks are inserted in `middle/` wherever an incoming edge cannot
   satisfy that canonical entry state directly;
5. `machine/` only turns the planned edge state into concrete register moves.

## Ownership By Layer

### `wasm/`

- No change in responsibility.
- Continue to produce semantic locals, structured control flow, typed ops, and
  call structure.

### Structural `middle/`

- Plan canonical frame layout.
- Shape CFG blocks and synthetic bridge blocks required by Wasm structure.
- During semantic preparation, decide:
  - which transient values stay live;
  - which transient values are published to operand slots;
  - which local accesses are slot-based vs cache-based;
  - which cached locals must be dropped, carried, or reloaded at boundaries.
- Lower semantic IR directly into SSA-IR using the already-decided explicit
  vocabulary:
  - `Value`
  - `Fill`
  - `Spill`
  - `LocalGetSlot`
  - `LocalGetCache`
  - `LocalSetSlot`
  - `LocalSetCache`
  - `LocalDropCache`
  - `Call`
- Lower `local.tee` as `LocalSetSlot` followed by `LocalGetSlot`.
- Run structural optimizations such as:
  - CFG cleanup;
  - sink planning rewritten against explicit local ops;
  - constant folding.

Important rules:

- The joint budget decision must happen before block params and edge bindings
  are fixed.
- Slot forwarding must not run in the main pipeline unless stack-shape sidecar
  data is restored, because it destroys the information the joint planner
  relies on.
- Boundary planning must happen in `middle/`, because only `middle/` sees:
  - exact transient live-in/live-out sets;
  - CFG join structure;
  - local hotness / reuse information;
  - canonical frame publication points.
- `middle/` should choose one canonical block-entry state and repair incoming
  edges to it, rather than expecting `machine/` to infer profitable cache
  carry on its own.

### Boundary Planning In `middle/`

Boundary repair should follow the same ownership split that transient edge
bindings already use today.

For transients, the current code already works like this:

- `middle/` decides live edge values and block params;
- `machine/` lowers them into `MachineEdge.args`;
- arch lowering emits the actual parallel copies and elides identity edges.

Cached locals should use the same model.

The target boundary plan is:

- each SSA block gets one canonical entry cache layout;
- the planner chooses which cached locals are expected to already be resident
  at block entry;
- if an incoming edge cannot satisfy that entry state directly, `middle/`
  inserts a synthetic repair block for that edge;
- repair blocks contain the explicit `Fill` / `Spill` / `LocalGetCache` /
  `LocalSetCache` / `LocalDropCache` sequence needed to reach the successor's
  canonical entry state.

This means the late boundary pass, tentatively `resource_plan`, should own:

- clears stale call `skip_reload` plumbing;
- chooses canonical block-entry cache state;
- chooses when an edge carries cache state versus spills and reloads;
- inserts repair blocks when the predecessor/successor states are incompatible;
- performs edge-local copy/spill/fill elimination while it still has SSA and
  CFG information.

This pass must not leave cache-boundary policy to `machine/`.

### `machine/`

`machine/` should become a consumer of the plan, not the owner of the policy.

`machine/` remains responsible for:

- choosing concrete physical registers for SSA values;
- choosing concrete physical registers for cached locals;
- tracking `value -> reg(s)` for transients;
- tracking `slot -> reg(s)` for cached locals;
- breaking aliases when overwriting a cached local register;
- realizing explicit call and edge mechanics;
- lowering the already-planned edge state into concrete parallel copies.

`machine/` must not choose which local to evict. Eviction must already be
explicit in SSA-IR.

`machine/` may bind cached locals anywhere in the dynamic bank, as long as the
chosen registers are currently free. Cached locals are no longer limited to a
separate fixed cache-only subrange.

`machine/` should not try to infer cache-carry policy. If a cache state is
meant to cross an edge, that must already be encoded by the block-entry plan
and its repair blocks.

## Planned SSA-IR Contract

The final SSA-IR local vocabulary should be only:

- `LocalGetSlot { slot, dst }`
- `LocalSetSlot { slot, src }`
- `LocalGetCache { slot, dst }`
- `LocalSetCache { slot, src }`
- `LocalDropCache { slot }`

Keep existing transient stack traffic:

- `Fill { slot, dst }`
- `Spill { slot, src }`

Keep the existing non-local ops:

- `Value { ... }`
- `Call(...)`

There should be no `LocalGet` / `LocalSet` in final SSA-IR.
There should be no local versioning in final SSA-IR.
There should be no cache-position ids in final SSA-IR.

The slot is the SSA-level identity. `machine/` chooses any free compatible
register(s) and remembers the binding `slot -> reg(s)`.

### Semantics

`LocalGetSlot { slot, dst }`

- Read the canonical local frame slot into SSA.
- Legal only when `slot` is not currently cached.

`LocalSetSlot { slot, src }`

- Write the canonical local frame slot from SSA.
- Legal only when `slot` is not currently cached.

`LocalGetCache { slot, dst }`

- After this op, `slot` must be resident in the cache.
- If `slot` is already cached, the read is cache-to-SSA only.
- If `slot` is not cached, `machine/` allocates free register(s), loads from
  the canonical slot, and binds `slot -> reg(s)`.

`LocalSetCache { slot, src }`

- After this op, `slot` must be resident and dirty in the cache.
- If `slot` is already cached, write the cached register(s).
- If `slot` is not cached, allocate free register(s) and bind `slot -> reg(s)`.
- No frame load is required before the write.

`LocalDropCache { slot }`

- Explicit eviction.
- If the cached local is dirty, spill it back to the canonical slot first.
- Then unbind `slot -> reg(s)` and free the register(s).

## Budget Model

Budgeting is by bank units, not raw value count.

For every program point and for each bank independently:

- `transient_units + cached_local_units <= total_dynamic_units`

Examples:

- on 64-bit GP targets, `i32`, `ref`, and `i64` each consume one GP unit;
- on 32-bit GP targets, cached `i64` and transient `i64` both consume two GP
  units;
- FP values consume FP units.

Important rule:

- From SSA planning's point of view, every transient lane is assumed usable by
  transient SSA at every non-call instruction.
- There is no notion of a permanently reserved transient sub-budget in SSA-IR.

## Call Rule

Calls are the one explicit exception.

Required contract:

- before a `Call`, all transient SSA values must already have been published
  out of the transient bank;
- only after that is it safe for `machine/` to temporarily reuse transient
  registers for call lowering and ABI mechanics.

This means:

- hidden "reserved transients" are not part of the SSA budgeting model;
- any machine lowering that depends on free transient lanes at normal
  non-call instructions is incompatible with the target architecture.

The target contract is:

- non-call instructions cannot depend on hidden spare transient lanes;
- call lowering may reuse transient registers only because the call boundary
  already forced transient publication.

## Joint Boundary Handling

Boundary handling should be fully planned in `middle/`.

The right split is:

- transient edge shape is decided during semantic preparation, because block
  params and edge bindings depend on it;
- cached-local edge shape is decided by the CFG-aware boundary planner, using
  one canonical cache-entry layout per block;
- `machine/` only realizes the planned edge state through block params, edge
  args, and emitted parallel moves.

The key rule is:

- predecessors do not get to choose arbitrary successor cache state;
- each block owns one canonical cache-entry state;
- incompatible predecessors are repaired before they enter the block.

This gives one place to decide:

- whether a cached local should be carried directly across an edge;
- whether it should be spilled before the edge;
- whether the successor should reload it;
- whether a move-only repair is enough;
- whether a synthetic edge block is required.

The current implemented boundary rule for cached locals is still conservative:

- before a `Call`, drop every resident cached local;
- before a CFG exit (`Goto`, `Branch`, `BrTable`), drop every resident cached
  local.

That is only the migration state. The target state is explicit compatible-edge
carry planned in `middle/`.

### Heuristic Inputs

The current implemented planner uses:

- transient stack shape and value types;
- local read/write hotness rankings;
- simple bank-unit costs;
- conservative cache dropping at boundaries.

Future heuristic upgrades can add:

- next-use data for transients;
- loop hotness;
- compatible boundary layouts;
- edge-specific copy elimination.

## Interaction With Existing Optimizations

The existing middle passes are still useful, but they should be rewritten to
operate on explicit slot-local ops instead of logical `LocalGet` / `LocalSet`.

### Slot Forwarding

`forward_slot_values()` is intentionally not part of the active pipeline right
now. It destroys operand-stack shape information that the joint planner needs.

If it is re-enabled later, it must carry private stack-shape metadata forward
so the planner still knows where transient values can be published and
reloaded.

### Sink Planning

`plan_sinks()` in `sink_plan.rs` should target `LocalSetSlot`, not
`LocalSet { version, ... }`.

The legality condition does not fundamentally require local version numbers.
What it needs is:

- slot identity;
- producer position;
- whether a read of the old slot value exists between producer and store;
- barrier awareness.

So local versioning should be removed rather than preserved for its own sake.

### CFG Simplification

`thread_jumps.rs` is already mostly agnostic to local flavor and should remain
before late resource planning.

The late planner may still create edge-repair bridge blocks afterward when the
chosen boundary repair needs them.

## Data That Should Go Away

The following should be removed from the exposed SSA-IR contract:

- `LocalGet`
- `LocalSet`
- local version numbers
- `ValueHome::LocalVersion`

If function-wide local hotness analysis remains useful, it should become
private planning input, not machine-facing IR policy.

## Implemented Shape

The code now uses this shape:

1. `spill_plan.rs` does joint transient+cache planning during semantic
   preparation.
2. `lower_ops.rs` / `lower_term.rs` emit the final explicit local vocabulary
   directly from that plan.
3. `resource_plan.rs` still only inserts conservative cache boundary drops.
4. `machine/` realizes the plan and binds both transients and cached locals
   from the unified dynamic bank.

The important missing piece is that cached-local boundary carry is not yet
planned with canonical per-block entry layouts and repair blocks.

## Future Work

- Reintroduce slot forwarding only with preserved stack-shape side data.
- Add canonical per-block entry cache layouts in `middle/`.
- Add repair blocks for edges whose predecessor state does not match the
  successor entry state.
- Replace conservative cache boundary drops with compatible-layout carry and
  edge-specific repair when profitable.
- Improve cache victim choice beyond the current rank-based heuristic.

## Success Criteria

- SSA-IR exposes only the explicit local vocabulary:
  - `LocalGetSlot`
  - `LocalSetSlot`
  - `LocalGetCache`
  - `LocalSetCache`
  - `LocalDropCache`
- There is no `LocalGet` / `LocalSet` left in final SSA-IR.
- At every non-call program point, combined transient and cached-local usage
  fits the total dynamic bank capacity.
- Transient edge shape is decided in `middle/`, before block params are fixed.
- Cached-local boundary policy is decided in `middle/`, not inferred in
  `machine/`.
- Calls and CFG edges use explicit planned repair, not conservative blind
  flushing.
- `machine/` no longer chooses cache victims.
- Cache register choice remains physical and local to `machine/`.
- The only place `machine/` may temporarily reuse transient registers outside
  ordinary SSA mapping is after a `Call`, because the IR contract has already
  published all live transients.
