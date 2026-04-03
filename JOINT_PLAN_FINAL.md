# Joint Transient + Local Cache Final Design

This document describes the intended end state of the compiler pipeline for
joint planning of transient SSA values and cached locals.

It is a design target, not a line-by-line description of current code. The
goal is to make the architecture, the ownership split, and the optimization
story clear enough that we know what the system is trying to become even while
the implementation is still moving.

## Executive Summary

The final system should have these properties:

1. There is one canonical slot-backed frame model for locals, operand-stack
   publication, call payloads, and return values.
2. There is one explicit SSA-IR interface for locals and transient values.
   The SSA-IR does not carry hidden local-version semantics or machine policy.
3. One dynamic GP bank and one dynamic FP bank are shared between:
   - transient SSA values, and
   - cached locals.
4. The policy decision of what stays transient, what is spilled, what is
   cached, what is carried across edges, and what is evicted belongs to
   `middle/`.
5. The choice of physical registers, edge copies, concrete loads/stores, and
   ISA instruction selection belongs to `machine/`.
6. Cache behavior at block boundaries is explicit, testable, and represented
   in SSA-IR. The backend does not invent hidden cache carry policy.
7. Late optimizations work by simplifying or realizing an explicit plan, not
   by reconstructing high-level intent from low-level code.

The overall shape is:

`Wasm bytecode -> Semantic IR -> Planned explicit SSA-IR -> Machine IR -> Native code`

The key architectural idea is simple:

- locals always have canonical slot homes;
- deep stack state always has canonical slot homes;
- only a bounded transient window is register-allocated as SSA values;
- hot locals may also have cached register homes when the joint planner
  decides they are worth it.

That split is what makes the system fast without needing a heavyweight global
register allocator.

## What Matters Most

The heart of the final design is not the number of pipeline stages. It is
these three things:

1. Joint planning at each program point.
   At every point in the program, the compiler decides the combined live state:
   which values stay live as transient SSA values, which values are published
   to slots, and which locals stay resident in cache.
2. Boundary cost-aware planning at merge points.
   When different incoming paths disagree, the compiler chooses a canonical
   boundary state that minimizes weighted copies, spills, fills, and cache
   repair, with special care to keep hot edges, especially loop backedges, as
   close to free as possible.
3. Validation of the resource-fit invariant.
   Once the explicit plan is built, it is validated so downstream lowering can
   trust that all required transient live values and cached locals fit in the
   available dynamic-bank budgets.

Everything else in the pipeline exists to support those goals.

## Core Principles

### 1. One explicit IR contract

The final middle-end should expose only one backend-facing SSA interface.

That interface should contain explicit local and stack traffic:

- `Value`
- `Fill`
- `Spill`
- `LocalGetSlot`
- `LocalSetSlot`
- `LocalGetCache`
- `LocalSetCache`
- `LocalEnsureCache`
- `LocalDropCache`
- `Call`

There should not be a second logical-local interface behind it.

In particular:

- no logical `LocalGet` / `LocalSet` in final SSA-IR;
- no local versioning in final SSA-IR;
- no physical register IDs in SSA-IR;
- no cache-position IDs in SSA-IR;
- no stale metadata channels for hidden continuation or cache policy.

### 2. Canonical slots first

Every local has a canonical frame slot. Operand-stack publication also has
canonical slots. Calls use canonical frame regions for args and results.

This is not just a storage choice. It is the foundation for the whole design:

- any value can always be published to memory without inventing a new home;
- calls do not need a special register calling convention inside SSA-IR;
- cached locals remain locals, not renamed SSA variables;
- the backend never needs to solve placement for the full Wasm operand stack.

### 3. Shared dynamic banks

There is one dynamic GP bank and one dynamic FP bank. Those banks are shared by
both transient values and cached locals.

The planner is free to use more of the bank for transients at one point and
more of it for cached locals at another point. The ratio is not fixed
per-function, per-loop, or per-block.

What must remain distinct is semantics:

- transient values are SSA values;
- cached locals are mutable local state with canonical slot homes.

### 4. Resource-fit invariant

The most important invariant of the whole design is:

- live transient SSA values, plus
- the local cache residency required by the explicit local/cache operations

must never exceed the total available dynamic-bank budget in either bank.

In other words, for GP and FP independently, the final explicit program must
never require more simultaneously-resident resources than the downstream
dynamic register budget can hold.

The compiler enforces this invariant explicitly:

- transient live-value pressure is controlled by emitting `Spill` and `Fill`;
- local cache pressure is controlled by choosing and rewriting explicit local
  operations:
  `LocalGetSlot`, `LocalSetSlot`, `LocalGetCache`, `LocalSetCache`,
  `LocalEnsureCache`, and `LocalDropCache`.

This means resource fit is not a best-effort backend heuristic. It is a
property of the explicit middle-end plan.

The final explicit SSA program should be validated against this invariant so
that downstream machine lowering can rely on it. Once validated, the backend
should be able to assume that all required live SSA values and all required
cached locals for any program point can fit inside the available dynamic-bank
budgets.

### 5. Policy in `middle/`, realization in `machine/`

`middle/` owns:

- spill vs keep-live decisions for transient values;
- slot vs cache access decisions for locals;
- cache residency decisions;
- cache eviction decisions;
- block-entry and block-exit boundary state decisions;
- weighted merge-point cost decisions;
- edge repair decisions;
- dirty/clean cache-state planning where it affects later behavior.

`machine/` owns:

- physical register assignment;
- concrete register moves;
- concrete loads and stores;
- edge parallel copies;
- helper scratch use;
- target instruction selection and encoding.

`machine/` must not decide which local is the victim when pressure rises. That
must already be explicit in the IR.

### 6. Boundary behavior is part of joint planning

If a successor block wants a particular transient or cached-local boundary
state at entry, the plan must say so. If an edge cannot satisfy that entry
state directly, the middle-end must insert an explicit repair block or
explicit repair sequence.

This is not a separate concern from joint planning. It is the merge-point half
of joint planning.

Inside a straight-line region, the planner directly chooses the live-out state.
On a simple linear edge, that live-out naturally becomes the next block's
input. The hard case is when multiple incoming paths disagree. Branches and
loops create those merge points. Choosing the right canonical boundary state
there, and making the hot paths cheap, is one of the central planning
problems of the whole system.

The backend is allowed to realize the plan efficiently. It is not allowed to
invent the plan.

## The Final Pipeline

### Pipeline map to code

The conceptual pipeline above corresponds roughly to these code areas today.
The exact function boundaries may change, but these files are the intended
homes of each stage:

- Wasm decode and semantic shaping
  Files:
  `sf-nano-core/src/vm/wasm/decode.rs`,
  `sf-nano-core/src/vm/wasm/control.rs`,
  `sf-nano-core/src/vm/wasm/context.rs`,
  `sf-nano-core/src/vm/wasm/inline.rs`,
  `sf-nano-core/src/vm/wasm/sir/semantic_ir.rs`,
  `sf-nano-core/src/vm/wasm/sir/primitive_op.rs`
  Input:
  Wasm bytecode plus module/function context.
  Output:
  Semantic IR with structured control, abstract locals, typed ops, and call
  structure.

- Canonical frame planning
  Files:
  `sf-nano-core/src/vm/middle/frame.rs`
  Input:
  Semantic IR plus backend configuration.
  Output:
  Canonical frame layout and canonical slot identities for locals, operand
  publication, calls, and returns.

- Joint planning, exact-stack phase
  Files:
  `sf-nano-core/src/vm/middle/spill_plan.rs`,
  `sf-nano-core/src/vm/middle/state.rs`
  Input:
  Semantic IR, frame layout, local types, and dynamic budgets.
  Output:
  Prepared semantic stream with explicit spill/fill prefixes, local
  slot-vs-cache choices, and entry live-state facts.

- CFG shaping and SSA lowering
  Files:
  `sf-nano-core/src/vm/middle/lower_cfg.rs`,
  `sf-nano-core/src/vm/middle/lower_block.rs`,
  `sf-nano-core/src/vm/middle/lower_ops.rs`,
  `sf-nano-core/src/vm/middle/lower_term.rs`,
  `sf-nano-core/src/vm/middle/lower_edge.rs`,
  `sf-nano-core/src/vm/middle/thread_jumps.rs`,
  `sf-nano-core/src/vm/middle/ssa_ir/ir.rs`,
  `sf-nano-core/src/vm/middle/ssa_ir/validate.rs`
  Input:
  Prepared semantic stream plus canonical frame layout.
  Output:
  Explicit SSA-IR CFG with block params, explicit slot traffic, explicit
  local/cache ops, and explicit terminators.

- Joint planning, boundary phase
  Files:
  `sf-nano-core/src/vm/middle/resource_plan.rs`
  Input:
  Explicit SSA-IR.
  Output:
  Final boundary-aware resource plan with canonical merge states, transient and
  cache edge repair, and cache demotion where the chosen boundary state cannot
  sustain residency.

- Late explicit SSA cleanup
  Files:
  `sf-nano-core/src/vm/middle/optimize.rs`,
  `sf-nano-core/src/vm/middle/sink_plan.rs`
  Input:
  Explicit SSA-IR plus the chosen joint resource plan.
  Output:
  Final explicit SSA-IR with simplified CFG, sink annotations, and other
  local simplifications that do not violate the explicit plan.

- Middle-end orchestration
  Files:
  `sf-nano-core/src/vm/middle/mod.rs`
  Input:
  Semantic IR plus backend configuration.
  Output:
  Prepared function containing frame plan plus final validated SSA-IR.

- Machine lowering
  Files:
  `sf-nano-core/src/vm/machine/lower_module.rs`,
  `sf-nano-core/src/vm/machine/lower_context.rs`,
  `sf-nano-core/src/vm/machine/lower_regalloc.rs`,
  `sf-nano-core/src/vm/machine/lower_inst.rs`,
  `sf-nano-core/src/vm/machine/lower_call.rs`,
  `sf-nano-core/src/vm/machine/lower_cached.rs`,
  `sf-nano-core/src/vm/machine/lower_leaf_arith.rs`,
  `sf-nano-core/src/vm/machine/lower_leaf_special.rs`,
  `sf-nano-core/src/vm/machine/lower_i64.rs`,
  `sf-nano-core/src/vm/machine/machine_ir/`
  Input:
  Final explicit SSA-IR plus frame plan and backend config.
  Output:
  MachineIR with concrete register assignments, loads/stores, calls, edge
  moves, and cache bindings.

- Machine validation and peephole optimization
  Files:
  `sf-nano-core/src/vm/machine/validate.rs`,
  `sf-nano-core/src/vm/machine/optimize.rs`,
  `sf-nano-core/src/vm/machine/peephole/`
  Input:
  MachineIR.
  Output:
  Validated and optimized MachineIR.

- Final native encoding
  Files:
  `sf-nano-core/src/vm/arch/`
  Input:
  MachineIR plus target ABI and register mapping.
  Output:
  Final native code bytes.

### 1. Wasm Decode -> Semantic IR

### Purpose

The decoder should preserve Wasm meaning and structure while the program is
still cheap to reason about.

### What this stage produces

Semantic IR should retain:

- structured control markers such as `Block`, `Loop`, `If`, `Else`, `End`;
- abstract local operations;
- typed primitive operations;
- semantic calls;
- local types and result types;
- maximum operand stack height.

### Why this stage exists

This is the last stage where the compiler still sees Wasm structure directly.
That structure is valuable:

- loops are still loops;
- calls are still calls;
- locals are still locals;
- the exact operand stack discipline is still obvious.

Many profitable decisions are cheaper here than later because later IRs no
longer carry this structure for free.

### Optimizations here

This is the right place for optimizations that want semantic structure but do
not want machine details.

Typical examples:

- small semantic inlining;
- simplification that relies on structured control;
- collection of typed operation result facts.

The important rule is that this stage should improve the later plan without
committing to machine policy too early.

### 2. Canonical Frame Planning

### Purpose

Before lowering into explicit SSA-IR, the compiler assigns stable canonical
homes in the frame.

### What this stage decides

It plans frame slots for:

- canonical locals;
- operand-stack publication and reload;
- call scratch / call payload regions;
- return result regions.

### Why this stage exists

The entire system depends on stable canonical homes.

Without canonical slots:

- spill/fill planning becomes ad hoc;
- call lowering must invent temporary layouts later;
- cached locals lose their stable slot identity;
- boundary repair becomes more complicated than it needs to be.

With canonical slots:

- any transient can be published immediately;
- any local can always be reloaded from a known home;
- calls become explicit slot traffic;
- repair logic has a clear source of truth.

### 3. Joint Planning, Exact-Stack Phase

### Purpose

This is the critical policy stage. It still has exact Wasm stack shape, so it
is where the compiler should make the straight-line joint decisions for:

- transient pressure;
- spill/fill publication;
- cache-vs-slot local access;
- cache eviction under local pressure;
- call and control boundary preparation.

### What this stage sees

At this point the compiler still knows:

- the exact abstract operand stack height at every semantic op;
- which operands are needed now;
- which results will be pushed;
- structured control boundaries such as loops, branches, and calls.

That information is more precise than what later SSA cleanup passes can
reconstruct.

### What this stage should decide

For each semantic operation, it should decide:

- which live stack values remain transient;
- which live stack values must be published to slots;
- whether a local access should be slot-based or cache-based;
- whether resident cached locals must be dropped to make room;
- what must be published before a call or structured boundary.

The decision must be made against the combined dynamic-bank budget, not
against a fixed transient-only budget.

This stage may use stable per-local semantic facts such as:

- type and bank class;
- dynamic-bank cost, such as `i64` pair cost on 32-bit GP targets;
- whether the entry zero value may be observed before a write.

What it must not use as a primary policy input is a whole-function hotness
ranking for locals. Cache and spill decisions must be made from the current
straight-line region and later CFG-aware planning, not from a global table
that can be badly misleading outside the region where a local is actually hot.

### Why this stage exists

This is where the compiler still has the strongest semantic information for
making good resource decisions cheaply.

If we postpone this stage too far:

- exact Wasm stack shape is lost;
- later passes have to infer publication and reload points indirectly;
- the transient/cache tradeoff becomes less precise.

### What this phase produces at block ends

When this phase reaches the end of a block, it has effectively chosen a
candidate live-out state for that path:

- which transient SSA values are live out of the block;
- which of those values have already been published to slots;
- which locals are still intended to be resident in cache;
- which cached locals are clean or dirty when that matters downstream.

On a straight-line edge, that live-out naturally becomes the next block's
input. At merge points, incoming paths may disagree. Resolving that
disagreement cheaply is the boundary phase of the same joint planner.

### Expected behavior at major boundaries

At calls:

- all transient values that must survive are published to canonical slots;
- cached locals are saved, dropped, or preserved exactly according to the
  explicit plan, not hidden backend heuristics.

At structured control boundaries:

- the plan should leave behind a candidate boundary state that can later be
  turned into explicit block parameters, slot publication, and cache repair.

For `local.tee`:

- it should lower as an explicit set followed by an explicit get from the
  chosen local access mode, preserving local semantics rather than turning the
  local into an SSA variable.

### 4. CFG Shaping and Block Parameter Construction

### Purpose

The compiler now turns structured semantic control into an explicit CFG with
SSA block parameters for live transient values.

### What this stage should do

- split the semantic stream into SSA blocks;
- keep only reachable blocks;
- build block parameters for live transient values at block entries;
- introduce structural bridge blocks when Wasm structure requires them.

### Why this stage exists

Transient values crossing edges must become explicit SSA dataflow.

This stage is the bridge between:

- exact structured Wasm preparation, and
- explicit CFG-based SSA lowering.

### Important rule

The cache side of the design is not represented as SSA block parameters. Cache
state is modeled separately as explicit block-entry residency plus explicit
repair.

That keeps the semantic split clean:

- transient values remain SSA values;
- locals remain local state.

### 5. Lowering to Explicit SSA-IR

### Purpose

This stage converts the prepared semantic program into the final explicit
SSA-IR vocabulary.

### What this stage should emit

It should emit only:

- `Value` for ordinary transient-producing operations;
- `Fill` and `Spill` for transient publication/reload through operand slots;
- `LocalGetSlot` and `LocalSetSlot` for canonical slot-based local traffic;
- `LocalGetCache` and `LocalSetCache` for explicit cached-local traffic;
- `LocalEnsureCache` for state-only cache materialization;
- `LocalDropCache` for explicit eviction;
- `Call` for slot-based calls.

### Why this stage exists

After this point, local semantics should already be explicit enough that the
backend can consume them directly.

This stage is not supposed to carry hidden policy forward. It is supposed to
make the policy visible.

### 6. Structural SSA Cleanup

### Purpose

This stage simplifies the explicit SSA program without changing the explicit
resource plan model.

### What belongs here

Cheap, non-destructive cleanup such as:

- jump threading and CFG simplification;
- trivial block cleanup;
- constant folding;
- constant operand absorption;
- elimination of obviously dead explicit operations when it does not change
  stack-shape assumptions needed by later planning.

### Why this stage exists

Once the program is explicit, many small simplifications become very cheap and
very safe. Those simplifications should reduce later backend work.

### Important rule

These cleanups must not destroy the information the boundary-phase joint
planner still needs.

For example, any rewrite that depends on pretending slot traffic is equivalent
to SSA value flow must only run if it preserves the explicit resource story.

The final design prefers visible, explicit state over aggressive early
canonicalization that later has to be inferred again.

### 7. Boundary-Phase Joint Planning

### Purpose

This is the CFG merge-point phase of joint planning. It finishes the explicit
resource plan over the SSA CFG and is responsible for making block-boundary
behavior explicit and profitable.

### What this stage should decide

For each merge-capable block, it should decide one canonical entry state that
is good for that block. That entry state includes:

- the transient live-ins the block expects to receive;
- the cached locals expected to be resident at entry;
- the dirty/clean state that matters for later behavior.

For each edge, it should decide:

- whether the predecessor exit state already matches the successor entry state;
- which transient copies, spills, or fills are required if it does not;
- which cache repair operations are required if it does not;
- whether the edge should carry state directly or pay repair cost.

For each block, it should also decide:

- which accesses must be demoted back to slot traffic because the chosen
  boundary state cannot sustain residency under the resource-fit invariant;
- which cached locals should remain resident at exit when that helps the
  surrounding CFG.

### Why this is part of joint planning

Each block owns one canonical entry boundary state.

That state should be chosen for the successor because it is profitable for the
successor, not merely because all predecessors happen to agree already.

This is especially important for:

- loop headers;
- hot joins;
- blocks with one hot predecessor and one cold predecessor.

The same idea applies to transient live-ins. The planner should not think
"transient edge bindings" and "cache boundary state" are two unrelated
problems. They are two parts of one boundary state.

### Boundary cost objective

At a merge point, the incoming states usually disagree. The planner should
choose the successor's canonical entry state to minimize weighted boundary
cost, not to maximize predecessor agreement in the abstract.

The cost model should account for things like:

- transient copies;
- transient spills and fills;
- cache ensures and drops;
- dirty-state repair or writeback when needed;
- extra bridge blocks or extra edge-local work.

The objective is not "make all edges equally simple". The objective is:

- make hot boundaries as close to free as possible;
- make cold edges pay repair cost when that buys a cheaper hot path;
- avoid churn across loop headers and backedges.

For loops, this point is critical. A loop header should usually prefer the
state that keeps the hot backedge cheap, even if that makes the cold preheader
pay repair cost.

### Repair model

If a predecessor exit does not match the successor entry state, the middle-end
should insert a repair block or explicit repair sequence containing things
such as:

- transient copies, spills, or fills;
- `LocalDropCache` for locals that must no longer be resident;
- `LocalEnsureCache` for locals that must become resident.

The key point is ownership:

- the middle-end chooses the canonical entry state;
- the middle-end chooses repair;
- the backend only realizes those explicit operations.

### Why this stage exists

Exact-stack joint planning alone is not enough. Once CFG cleanup and edge
structure exist, the compiler needs a phase that reasons about joins, loops,
edge compatibility, and weighted boundary cost.

This is the stage that turns per-path live-out choices into a coherent
whole-program boundary plan.

### 8. Dirty/Clean Cache-State Planning

### Purpose

Cache residency alone is not enough. The final system should also preserve
whether a resident cached local is:

- clean: register matches canonical slot memory;
- dirty: register is newer than canonical slot memory.

### Why it matters

Dirty-state precision enables:

- saving only what must be saved before calls;
- avoiding unnecessary writeback of clean carried cache entries;
- correct and efficient repair decisions;
- better continuation behavior after calls and edge transfers.

### Final model

The final system should treat dirty/clean state as part of the explicit plan,
or at least as a precisely derived fact from the explicit plan.

The backend should never have to assume that all carried cache entries are
dirty just because they crossed an edge.

### Optional extension

If exact dirty-state canonicalization needs it, the IR may also support an
explicit state-preserving writeback operation in the future.

That is an optimization refinement, not the core architectural requirement.
The core requirement is that dirty-state behavior is explicit and precise,
never hidden behind backend guesswork.

### 9. Sink Planning

### Purpose

Sink planning identifies values whose producer can write directly into the
final local home instead of producing a transient and then storing it.

### What it should detect

Given a producer and a later local set, the planner may mark the result as
sinkable when:

- the producer is targetable;
- there is no barrier in between;
- no intervening read observes the old local contents;
- the sink is compatible with the final explicit local mode.

### Why it belongs after boundary-phase joint planning

The legality and usefulness of sinks depends on the final explicit local form.
If a late resource pass demotes a cache access to a slot access, or preserves a
cache access, sink planning should see the final answer rather than guessing.

### Optimization value

This directly removes:

- transient register traffic;
- redundant moves;
- redundant slot stores;
- redundant cache-register moves.

It is one of the highest-value late middle-end optimizations because it turns
explicit final ownership information into simple local wins.

### 10. Machine Lowering: Explicit Plan Realization

### Purpose

Machine lowering consumes the fully explicit SSA plan and turns it into
MachineIR.

### What the backend receives conceptually

It should receive:

- explicit SSA transient values and CFG edges;
- explicit slot traffic;
- explicit cached-local operations;
- per-block canonical entry boundary state;
- precise dirty/clean entry facts;
- stable slot types.

It should not need to infer hidden local policy.

### What the backend should do

For transient values:

- allocate registers from the dynamic GP/FP banks;
- reuse dead input registers when possible;
- lower `Value` operations into concrete machine ops;
- realize block-parameter edge copies.

For cached locals:

- bind each explicit cached local to compatible physical registers in the same
  dynamic bank family;
- keep the binding `slot -> register(s)` in lowering state;
- lower `LocalGetCache`, `LocalSetCache`, `LocalEnsureCache`, and
  `LocalDropCache` exactly as requested by the plan;
- track cache live and dirty state precisely.

For calls:

- save or publish only the transient and cached state required by the explicit
  plan;
- do not run a hidden continuation reload policy;
- begin continuations from the explicit boundary state chosen earlier.

### Dynamic-bank model

The final backend should behave like a unified dynamic bank per register file,
with preference ordering but without semantic partition ownership.

In practice that means:

- transient allocation can use any free compatible dynamic register;
- cached locals can bind to any free compatible dynamic register;
- helper scratch remains a narrow explicit exception, not a second policy path.

### Why this stage exists

The backend is where abstract operations become concrete register moves,
loads, stores, calls, and branches. It should be good at that job, but it
should not be responsible for reconstructing whole-program cache policy.

### 11. Machine-Level Optimization

### Purpose

After explicit lowering, MachineIR still contains enough structure for local
late optimizations that are easier at the machine level than in SSA-IR.

### What belongs here

Typical late machine optimizations include:

- copy propagation;
- redundant move elimination;
- dead store / dead reload cleanup when directly visible;
- addressing-mode fusion;
- immediate-form selection;
- edge-copy cleanup;
- branch simplification;
- target-specific peepholes.

### Why this stage exists

The middle-end chooses policy. The machine optimizer recovers compact native
forms from that explicit policy.

This stage should improve code quality without changing ownership boundaries.

### 12. Native Encoding

### Purpose

This final stage maps MachineIR to target ISA instructions and emits code.

### What belongs here

- concrete instruction encoding;
- exact register-number mapping;
- branch encoding and fixups;
- final addressing modes;
- final immediate materialization choices.

### Why this stage exists

Instruction encoding is target-specific and should happen as late as possible,
after all target-independent and machine-level structural choices are done.

## The Final SSA Contract

The final SSA-IR should mean exactly this:

- `Value`: produce transient SSA results.
- `Fill`: reload a transient from its canonical operand slot.
- `Spill`: publish a transient to its canonical operand slot.
- `LocalGetSlot`: read a local from canonical slot memory.
- `LocalSetSlot`: write a local to canonical slot memory.
- `LocalGetCache`: read a local from cache, loading it if needed according to
  explicit cache semantics.
- `LocalSetCache`: write a local through its cache binding, marking it dirty.
- `LocalEnsureCache`: make a local resident in cache without producing an SSA
  value.
- `LocalDropCache`: evict a cached local, writing back if needed.
- `Call`: perform a slot-based call with args/results in canonical frame
  regions.

This contract keeps the local semantics explicit while still allowing a very
cheap backend.

## Boundary Model

The final system should model block boundaries this way:

1. Each block has one canonical entry boundary state, chosen by the joint
   planner.
2. The transient part of that state uses SSA block parameters and explicit
   edge bindings.
3. The local-cache part of that state uses explicit canonical entry cache
   layouts plus explicit dirty/clean meaning when that matters.
4. If an incoming edge already matches the successor entry state, the edge is
   effectively free or close to free.
5. If it does not match, the middle-end inserts the minimum repair it chose
   for that edge.

This produces a clean split:

- SSA values are true SSA dataflow;
- locals are true local state;
- both are explicit.

The key performance objective is not symmetric treatment of all edges. It is to
make the hot boundaries cheap, especially loop backedges and other hot merge
paths, even when that means colder edges must pay repair cost.

## Call Model

The final call model should be:

- call args/results live in canonical frame regions;
- live transient values are explicitly published before the call if needed;
- cached locals are saved or dropped according to explicit dirty-aware policy;
- continuation entry state is explicit;
- there is no hidden selective reload metadata channel.

This makes calls predictable, debuggable, and easy to optimize locally.

## Optimization Inventory

The final system should support the following optimization families in a clean
layered way:

### Semantic / front-end optimizations

- small leaf inlining;
- semantic simplification while structure is still intact;
- precise per-local semantic facts such as type/cost and entry-read analysis.

### Joint planning within blocks

- joint transient/cache budgeting;
- spill/fill placement against exact Wasm stack shape;
- profitable slot-vs-cache local access selection;
- early cache eviction under pressure;
- precise call-boundary preparation.

### Boundary-cost-aware joint planning

- block-parameter construction for live transients;
- successor-owned canonical boundary states;
- weighted boundary-state choice at merge points;
- hot-edge and hot-backedge friendly planning;
- edge repair insertion for both transient and cache state;
- cache carry across compatible edges;
- profitable cache demotion when the final block shape cannot sustain it.

### Late explicit SSA optimizations

- constant folding;
- constant operand absorption;
- sink planning into slot or cache homes;
- elimination of redundant explicit traffic when local reasoning proves it.

### Machine-lowering optimizations

- dead-input register reuse;
- direct binding of a cached local from a dying source value;
- direct sink pre-mapping into cache registers;
- precise dirty-aware cache save behavior;
- identity edge-copy elimination;
- minimal helper scratch use.

### Machine peephole optimizations

- move elimination;
- addressing-mode fusion;
- immediate-form selection;
- load/store simplification;
- branch cleanup;
- target-specific local patterns.

## Non-Goals

The final system should not do these things:

- turn locals into general SSA variables;
- make the backend own cache eviction policy;
- expose physical register IDs in SSA-IR;
- expose cache-position IDs in SSA-IR;
- maintain two competing local-operation interfaces in the middle-end;
- rely on a fixed split between transient lanes and cached locals;
- hide continuation or boundary policy in stale metadata side channels.

## Mental Model For The Whole System

The final compiler should be thought of this way:

- Wasm decode preserves meaning and structure.
- Early middle-end assigns stable homes.
- Straight-line preparation uses exact stack shape to choose what stays live,
  what spills, and what is worth caching.
- SSA lowering makes that plan explicit.
- Late middle-end chooses one profitable cache-state story per block and makes
  edge repair explicit.
- Sink and cleanup passes remove cheap redundancies after the final plan is
  known.
- Machine lowering assigns physical registers and realizes the plan.
- Machine optimization compresses the plan into better native patterns.
- Final encoding emits machine code.

The point is not to optimize by running ever-more-complicated passes. The
point is to choose the right explicit representation and the right ownership
split so that the profitable optimizations become simple and local.

That is the final intended design.
