# Optimization-Oriented Micro-JIT Pipeline

Entry point: `sf-nano-core/src/vm/build.rs` -> `ensure_module_compiled()`

This engine is fast not because it runs a huge optimizer, but because the
pipeline is arranged so that most profitable optimizations become cheap local
rewrites.

The central idea is:

1. Keep the Wasm frontend structured long enough to make good global-ish
   choices cheaply.
2. Turn deep stack state, calls, and locals into explicit canonical homes
   early, so the backend never needs full general-purpose register allocation.
3. Preserve just enough semantic shape in MachineIR for small late peepholes
   and backend instruction selection to recover native-quality code.

So the pipeline is not "decode, then optimize a lot". It is "shape the program
so later optimization has less work and lower risk".

## Pipeline At A Glance

`Wasm Bytecode -> Semantic IR -> Prepared SSA-IR -> Machine IR -> Native Code`

| Stage | Main job | Main optimization value |
| --- | --- | --- |
| Wasm -> Semantic IR | Preserve Wasm structure and types | Keep the frontend structured enough that cheap global-ish choices are still possible (dormant leaf-inliner slot lives here) |
| Semantic IR -> Prepared SSA-IR | Make slots, transient budget, and call barriers explicit | Joint cache/transient planning, constant folding, sink planning, and low-cost CFG cleanup |
| SSA-IR -> Machine IR | Map bounded SSA live ranges into a fixed register partition | One-pass lowering, cached-local aliasing, dead-input register reuse, cheap call handling |
| Machine IR optimize | Recover native patterns from the fixed-shape lowering | Addressing-mode fusion, copy propagation, load/store forwarding, branch fusion |
| Machine IR -> Native Code | Select ISA forms late | Immediate forms, indexed addressing, shifted operands, compact edge moves, small tails |

## Why This Design Works

The pipeline deliberately separates values into three classes:

- Canonical locals: always have stable frame-slot homes, and hot ones may also
  have pinned cache registers.
- Deep Wasm stack / call payloads: always have canonical operand slots.
- Short-lived transients: only these participate in register allocation.

That split is what removes the need for a heavyweight global allocator. The
backend only has to manage a bounded transient window; everything else already
has a home.

## 1. Wasm Bytecode -> Semantic IR

Main code:

- `sf-nano-core/src/vm/build.rs` (`decode_function_semantic`)
- `sf-nano-core/src/vm/wasm/decode.rs`
- `sf-nano-core/src/vm/wasm/sir/semantic_ir.rs`
- `sf-nano-core/src/vm/wasm/sir/primitive_op.rs`
- `sf-nano-core/src/vm/wasm/inline.rs` (dormant — retained but not wired
  into the current pipeline)

### Optimization-enabling representation

`decode_to_semantic_ir()` keeps Wasm-specific structure intact:

- structured control markers (`Block`, `Loop`, `If`, `Else`, `End`)
- abstract locals (`LocalGet`, `LocalSet`, `LocalTee`)
- semantic calls (`CallDirect`, `CallIndirect`)
- typed result information (`local_types`, `result_types`, `op_result_types`)
- `max_stack_height`

This is not an optimization pass by itself. It is an optimization-friendly
representation choice.

Why it lives here:

- The frontend still knows exact Wasm structure.
- No frame slots, cache registers, or transient budgets exist yet.
- Later passes can reason about loops, calls, and locals without reverse
  engineering them from low-level code.

### Optimization: small leaf inlining (dormant)

`inline_calls_in_function()` in `wasm/inline.rs` replaces eligible
`CallDirect` sites with the callee body. The module currently has no live
callers — it is retained under `#![allow(dead_code)]` so it can be re-wired
after the middle-layer rewrite settles, but the production pipeline in
`build.rs` does not invoke it today.

Current policy (when re-enabled):

- callee must be a straight-line leaf (no nested calls, no structured
  control flow — only primitives and local ops)
- at most `MAX_INLINE_OPS = 12` semantic ops at non-loop call sites, or
  `MAX_INLINE_OPS * LOOP_INLINE_MULTIPLIER = 120` ops at call sites inside
  a loop
- at most `MAX_INLINE_PARAMS = 16` parameters
- one pass over the caller; transitive chains are not re-expanded after
  substitution

Simple example:

```wat
(func $inc (param i32) (result i32)
  local.get 0
  i32.const 1
  i32.add)

(func $twice_inc (param i32) (result i32)
  local.get 0
  call $inc
  call $inc)
```

After inlining, `$twice_inc` becomes the straight-line equivalent of:

```wat
local.get 0
i32.const 1
i32.add
i32.const 1
i32.add
```

What this optimizes:

- removes call/return overhead
- exposes more constant folding and local forwarding opportunities
- exposes larger straight-line regions to the later single-pass lowerer

Why it belongs here instead of later:

- inlining before frame layout avoids repairing slot assignments
- inlining before cached-local analysis improves hot-local scoring
- inlining before SSA lowering avoids rebuilding CFG and value mappings

How the pipeline design enables it:

- Semantic IR still expresses locals and control flow directly
- decode stores complete callee bodies, so inlining is just structured splice +
  retargeting, not machine-code surgery

## 2. Semantic IR -> Prepared SSA-IR

Main code (pipeline entry is `prepare_function` in `middle/mod.rs`):

- `sf-nano-core/src/vm/middle/mod.rs` — drives the stage
- `sf-nano-core/src/vm/middle/frame.rs` — canonical frame-slot layout
- `sf-nano-core/src/vm/middle/cfg.rs` — explicit semantic CFG + reachability
- `sf-nano-core/src/vm/middle/slot_ssa.rs` — slot-only SSA skeleton
- `sf-nano-core/src/vm/middle/joint_plan/` — joint transient/cache planner
  (entry-region analysis, local-access decisions, init-local facts,
  region solver, and the exact-simulation walker in `exact.rs` that produces
  the plan-authoritative per-block cache entry/exit rows + per-edge repair
  actions)
- `sf-nano-core/src/vm/middle/rewrite/` — final SSA-IR materialization
  (function body + edge repair), seeded from the plan's exact rows
- `sf-nano-core/src/vm/middle/cleanup.rs` — structural CFG cleanups
  (cache-run canonicalization, jump threading, single-predecessor merge,
  unreachable-block removal)
- `sf-nano-core/src/vm/middle/optimize.rs` — constant folding and
  constant-operand absorption
- `sf-nano-core/src/vm/middle/sink_plan.rs` — cached-local sink planning
- `sf-nano-core/src/vm/middle/final_signals.rs` — derives the two
  machine-facing cache signals (per-entry `Ensure`/`Reserve` requirement rows
  + whole-function preferred-preserved flags) over the FINAL SSA, after
  cleanup's block merges, using `ModuleFacts` (callee locality + table
  dispatch modes) to classify local-JIT-call crosses
- `sf-nano-core/src/vm/middle/ssa_ir/ir.rs` — SSA-IR definitions

Pipeline order in `prepare_function`:

```text
validate semantic
plan_frame_layout
cfg::build_semantic_cfg
slot_ssa::lower_slot_only_ssa
JointPlanner::build  (joint_plan/*: region solver + exact walker -> plan-authoritative cache rows)
rewrite::rewrite_function  (seeds each block from the plan's exact entry row)
cleanup::cleanup_program
optimize::optimize_program
sink_plan::plan_sinks
final_signals: derive block_entry_cache_requirements + preferred_preserved over the FINAL SSA
validate prepared SSA
```

The two machine-facing cache signals are derived last, over the final SSA, so
they see cleanup's block merges — a pre-cleanup per-block classification would
go stale when a merge folds a successor's first-touch into its predecessor. The
machine reads these rows (see stage 3); the exact walker's own requirement
classification stays internal to the plan's edge-repair derivation.

This stage is where the engine makes most of its important optimization
decisions. It does not try to produce final code yet. It produces an IR whose
shape makes later codegen cheap.

### Optimization-enabling structure: canonical frame layout

`plan_frame_layout()` gives every local, operand spill, call payload, and
return result a stable slot.

Why that matters:

- later passes never need to invent homes for values under pressure
- calls and helpers naturally operate on frame regions
- all cross-boundary values already have a canonical memory location

Simple example:

```text
locals:        fp[0..L)
call scratch:  fp[L..L+S)
operand slots: fp[L+S..]
```

This seems mundane, but it is the foundation that lets the backend avoid full
graph-coloring RA.

### Optimization: local-cache selection

The joint planner (`joint_plan/entry_region.rs` + `joint_plan/build.rs`)
scores locals per block and decides which canonical locals are worth keeping
resident in dedicated cache registers at each block entry.

What it does:

- scores locals by access frequency within a block, with a write-first
  bonus at boundaries and a reuse bonus for repeat reads
- inherits scores across block edges so hot locals stay cached through
  loops
- respects separate GP and FP budgets
- on 32-bit GP targets, charges `i64` locals as two GP units

Simple example:

```wat
loop
  local.get 0
  local.get 0
  local.set 1
end
```

If local `0` is much hotter than other locals, it is a good cache candidate.
Later `local.get 0` can become a register alias instead of a frame load.

Why it belongs here:

- this stage still sees whole-function local traffic
- later MachineIR only sees slot numbers, not meaningful "hot local" structure

How the pipeline enables it:

- locals are still semantic locals here
- the backend config already exposes explicit cache budgets

### Optimization: entry zero-init elision for cached locals

`joint_plan::init_locals::locals_reads_before_write` computes
`reads_before_write` for each local slot and stores it in
`SsaProgram.local_slot_info`.

What it does:

- if a non-parameter local is definitely written before any read at entry
  scope, the backend skips zero-initializing its cache register
- if the local may be read before a write, the backend materializes the Wasm
  mandated zero

Simple example:

```wat
local.set 0, ...
local.get 0
```

The cached register for local `0` can start undefined, because the first read
cannot observe the initial Wasm zero.

Why it belongs here:

- this needs structured whole-function control-flow analysis
- once lowering has turned locals into slot operations, the high-level proof is
  harder and less reliable

### Optimization: post-call selective reload skipping

The joint planner's per-block entry-region analysis decides which cached
locals a block needs resident on entry. At a call continuation, the
successor block's entry-cache requirement governs what is reloaded: if a
cached local will be overwritten before it is read, the successor simply
does not request it, so the edge repair in `rewrite/edge.rs` never emits
a reload for it.

What this achieves:

- cached locals that are dead-before-overwrite on the call continuation
  are not reloaded
- avoids useless "save before call, reload after call, overwrite immediately"
  traffic

Simple example:

```wat
call $foo
local.set 0, ...
```

If local `0` is cached and not read before that `local.set`, the continuation
block does not list it as an entry requirement, and the continuation edge
skips reloading local `0`.

Why it belongs here:

- it relies on semantic-local reads/writes in the straight-line region after
  the call
- later machine lowering only knows slots and registers, not the semantic read
  vs write intent

The volatile/preserved class of a cached local is decided by the residency
solver as a whole-function nomination: a local is preserved-class when its
trip-weighted survivable-call relief amortizes the backend's per-lane
save/restore overhead (`preserved_lane_save_overhead`), clamped to the bank's
preserved-lane capacity. Nominated residents pay a reduced call tax
(`algorithm4:pkeep=`) at survivable (direct local-JIT) calls, and the plan's
call model keeps them resident across those calls — no post-call re-ensures or
backedge repair loads are scheduled for them. The nomination is published as
`SsaProgram::preferred_preserved`.

Machine lowering executes the contract: new cache homes for nominated locals
prefer preserved dynamic lanes (inherited block-entry layouts are not forced
to switch banks, because that would create extra edge-copy code), and
`lower_call_internal()` carries nominated non-ref caches that sit in preserved
dynamic lanes as explicit `Call.success` edge arguments. Dirty carried caches
are frame-published before the call, then continue as clean preserved-lane
values on the success edge. Ref-typed cached locals are never carried across
the local-call safepoint; a plan-kept cache the machine could not place in a
preserved lane reloads lazily at its next use. Runtime/indirect/ref calls
remain full cache barriers with frame-publication semantics.

### Optimization-enabling structure: explicit spill/fill planning

`rewrite::rewrite_function` constrains the live transient window to the
configured GP/FP budgets and inserts explicit `Spill` / `Fill` actions when
needed, guided by the joint planner's decisions.

What it does:

- keeps only a bounded suffix of the Wasm stack live as SSA values
- publishes deeper stack values into canonical operand slots
- ensures calls and runtime boundaries see their arguments/results in slots

Simple example:

```wat
local.get 0
local.get 1
local.get 2
local.get 3
i32.add
```

If the transient budget can only keep two values live, older stack values get
spilled to operand slots before pressure becomes a backend problem.

Why it belongs here:

- only this stage still has exact Wasm stack-height information
- after this step, the backend can assume "deep stack already has a slot home"

How the pipeline enables later optimization:

- the backend never has to solve register pressure for the full Wasm stack
- constant folding and slot forwarding operate on an IR with explicit memory
  publication points

### Optimization: flat CFG construction + reachability pruning

`cfg::build_semantic_cfg` (using `build_block_ranges` + `retain_reachable_blocks`)
turns structured control into a flat basic-block CFG and drops blocks that
can never execute.

What it optimizes:

- removes dead lowering work
- shortens later optimization scans
- prepares the CFG for jump threading and block merging

### Optimization: selective slot publication at control-flow boundaries

`maybe_publish_live_window_for_targets()` and
`publish_taken_branch_payload_at()` only spill the portion of the live window
that the successor actually requires in canonical slots.

Simple example:

```text
current live window: [v0, v1, v2]
target entry state:  spill_depth requires only v0 spilled
```

Only `v0` is published to a slot; `v1` and `v2` can stay live if the successor
accepts them as block parameters.

Why it belongs here:

- it needs both current live-window state and precomputed target entry state
- later passes do not know the original stack-shape contract anymore

### Optimization-enabling structure: leaf/call split in SSA ops

Block lowering turns semantic ops into these SSA-IR variants:

- primitive value ops for pure-ish arithmetic / compare / convert work
- `Call` ops for compiled-call and runtime-dispatch sites
- `LocalGetSlot` / `LocalSetSlot` for frame-slot traffic
- `LocalGetCache` / `LocalSetCache` for cached-local traffic
- `LocalEnsureCache` / `LocalReserveCache` / `LocalDropCache` for explicit
  cache residency transitions
- `Spill` / `Fill` for deep-stack publish/refresh

Why that matters:

- calls become exact barriers for later legality checks
- the explicit cache-transition ops make the boundary contract visible in
  SSA rather than implicit in the backend
- pure value ops remain easy targets for constant folding and sink planning

### Optimization: CFG simplification after SSA construction

`cleanup::cleanup_program` iterates a handful of cheap CFG cleanups until
fixed point:

1. canonicalize cache-only instruction runs (`simplify_cache_only_runs`)
2. jump threading through trivial empty goto blocks
   (`thread_one_empty_goto_block`)
3. single-predecessor block merging (`merge_one_goto_successor`)
4. unreachable-block removal (`remove_unreachable_blocks`)

Simple example:

```text
b0 -> b1
b1: goto b2
```

becomes:

```text
b0 -> b2
```

If `b2` has only one predecessor, `b0` and `b2` may also merge into one block.

Why it belongs here:

- once block parameters and bindings exist, threading/merging is simple and
  local
- doing it earlier would require structured-control surgery
- doing it later would make MachineIR and native code carry useless block glue

### Optimization: constant folding and constant-operand absorption

`fold_constants_into_operands()` does three related rewrites:

1. evaluate pure ops with all-constant inputs
2. replace `Value(v_const)` operands with inline `Const(bits)` operands
3. remove dead constant definitions

Simple example:

```text
i32.const 40 -> v0
i32.const 2  -> v1
i32.add v0, v1 -> v2
```

becomes:

```text
i32.const 42 -> v2
```

And if a value is used once:

```text
i32.const 8 -> v0
i32.shl x, v0
```

becomes:

```text
i32.shl x, #8
```

Why it belongs here:

- arithmetic is still semantic enough to evaluate safely
- the backend has not yet committed to scratch-register placement
- the transient budget already guarantees a fallback register exists if a
  backend cannot encode the immediate natively

### Optimization: sink planning

`sink_plan::plan_sinks` finds `LocalSetCache` operations whose producer can
write directly into the local's cached home register, and annotates the SSA
program's `value_sink_local` table accordingly.

Simple example:

```text
local.get x -> v0
i32.add v0, #1 -> v1
local.set x, v1
```

If no call and no intervening read of the old `x` exists, the add result
can be placed directly in `x`'s cache register and the `LocalSetCache`
disappears (machine lowering consumes the sink annotation via
`apply_sink_premap`).

Why it belongs here:

- legality depends on local-version liveness, which is an SSA-layer fact
- the actual register choice is still deferred to Machine lowering

How the pipeline enables it:

- local versioning proves "old x is dead"
- cached-local analysis has already declared which locals have profitable
  register homes

## 3. Prepared SSA-IR -> Machine IR

Main code:

- `sf-nano-core/src/vm/machine/lower_module.rs`
- `sf-nano-core/src/vm/machine/lower_context.rs`
- `sf-nano-core/src/vm/machine/lower_regalloc.rs`
- `sf-nano-core/src/vm/machine/lower_cache_layout.rs`
- `sf-nano-core/src/vm/machine/lower_cached.rs`
- `sf-nano-core/src/vm/machine/lower_inst.rs`
- `sf-nano-core/src/vm/machine/lower_leaf_arith.rs`
- `sf-nano-core/src/vm/machine/lower_leaf_special.rs`
- `sf-nano-core/src/vm/machine/lower_call.rs`
- `sf-nano-core/src/vm/machine/lower_const_pool.rs`
- `sf-nano-core/src/vm/machine/lower_i64.rs`
- `sf-nano-core/src/vm/machine/lower_i64_gp64.rs`
- `sf-nano-core/src/vm/machine/gp32/lower_leaf.rs`
- `sf-nano-core/src/vm/machine/validate.rs`

This stage is where the "simple pipeline" claim becomes concrete. Because the
SSA stage already bounded transient pressure and assigned canonical homes, the
Machine lowerer can stay one-pass and local.

### Optimization-enabling structure: fixed register-file partition

`MachineRegFile::new()` partitions the virtual register space as:

```text
[fixed | gp_dynamic | fp_dynamic]
```

Fixed registers include:

- runtime context pointer
- frame pointer
- `mem0_base`
- `mem0_size`

The GP and FP dynamic banks are ordered pools with an abstract volatility split
supplied by `BackendConfig`: volatile lanes first, then preserved lanes (plus
any backend-only scratch tail for GP). Cached-local residency and linear-value
ownership are tracked in `BlockLowerContext` state rather than by register
number. `BackendConfig` also supplies the per-lane preserved save overhead the
residency solver prices preserved-class nominations with. The bank order is an
ABI preference, but ownership is authoritative.

Why that matters:

- no later pass has to rediscover ABI roles
- cached locals and transient temps share one pool under explicit metadata,
  so sink-pinning and alias tracking stay exact instead of implicit
- MachineIR can carry selected clean non-ref cached locals when they already
  reside in preserved lanes, without naming physical registers
- one-pass allocation becomes possible

### Optimization: entry cached-local initialization

The machine owns the physical lane layout: `lower_cache_layout::
compute_block_entry_cache_params` assigns each cached local a register,
biasing call-crossing preferred locals into preserved lanes. It reads two rows
the middle publishes on the `SsaProgram` (both derived over the final SSA in
`final_signals`, see stage 2):

- `block_entry_cache_requirements` — per entry slot, `Ensure` (materialize the
  value on entry) vs `Reserve` (the block writes it first, so no incoming value
  is needed); this is the machine's `needs_value` input and drives the edge
  `reserve()` markers
- `preferred_preserved` — the whole-function per-slot flag promoted into the
  layout's per-block preference (also consumed by binding-time register
  selection in `lower_context` for mid-block caches)

Together with `rewrite`-emitted `LocalEnsureCache` / `LocalReserveCache` ops,
this drives machine lowering to:

- load cached parameters from frame slots
- zero-initialize only locals that may be read before write
- skip untouched cached locals entirely when `reads_before_write == false`

This is the backend realization of the earlier joint-plan analysis; the middle
hands the machine the context-free preference + per-entry requirement, and the
machine places the lanes.

### Optimization: `LocalGet` source aliasing

When a local is cached, lowering a `LocalGetCache` does not emit a load. It
just maps the SSA value to the cache register.

Simple example:

```text
LocalGetCache x -> v0
i32.add v0, #1
```

If `x` is cached in `r5`, `v0` is simply "value in `r5`".

Why it belongs here:

- only this stage knows both the SSA value and the concrete cache-register map

### Optimization: `LocalSet` elision via sink pre-mapping

Before lowering a value-producing op, `apply_sink_premap()` can pre-assign its
destination to a cached local register when sink planning approved it.

Simple example:

```text
i32.add v0, #1 -> v1
LocalSetCache x, v1
```

becomes "emit the add directly into `x`'s cache register".

### Optimization: materialize cache aliases only when overwrite is imminent

Cached registers can alias multiple live SSA values. When a cache register is
about to be overwritten, `materialize_cache_aliases()` spills only the still
needed aliases into transients.

Why this is effective:

- aliasing is free while the cached value remains current
- the move cost is only paid when the cache register must change ownership

This is much cheaper than eagerly copying every cached local read into a fresh
transient.

### Optimization: one-pass allocation with dead-input reuse

Arithmetic lowering aggressively reuses dead input registers for results via
`alloc_*_reusing_dead_inputs()`.

Simple example:

```text
r7 = i32.add r7, r8
```

If the left input dies here, the result can stay in `r7` rather than needing a
new transient.

Why it belongs here:

- this is where actual register assignment exists
- the bounded transient window makes dead-input reuse sufficient for most cases

This is one of the main reasons the backend can avoid heavyweight register
allocation.

### Optimization: immediate-to-store coalescing

`try_coalesce_last_store_immediate()` looks for:

```text
move rTmp <- imm
store [slot] <- rTmp
```

and rewrites it to:

```text
store [slot] <- imm
```

Why it matters:

- reduces transient pressure around calls and long argument setup
- especially useful on 32-bit paths where GP transients are tighter

### Optimization: direct local calls reuse the caller's frame region

For internal calls, `lower_call_internal()` reuses the caller operand span as
the callee frame prefix.

What this avoids:

- argument reshuffling into a separate outgoing-call area
- extra copies for already-published call arguments

Why it belongs here:

- the canonical frame contract has already made both caller operands and callee
  local prefix concrete frame regions

### Optimization: save/reload only around true boundaries

Because `Call` ops were already made explicit in SSA, this stage can use a
simple rule:

- no live transient values may cross a call
- runtime helpers publish dirty cached locals before the helper boundary
- compiled direct local calls publish/drop non-selected cached locals and publish
  every dirty cached local; selected non-ref cached locals already in preserved
  dynamic regs are carried through the explicit `Call.success` edge, with dirty
  survivors published once before the call and then treated as clean
- cached locals are selectively reloaded after calls based on the successor
  block's entry-cache requirement (reloads the continuation does not need are
  never emitted)

The pipeline turns calls from "hard global allocation problem" into "make the
boundary state explicit: frame-publish what must be frame-visible, carry only
MIR-declared preserved-register survivors, reload what is still needed".

### Optimization: mem0 fixed-register fast path and guard-page-aware bounds checks

Memory `0` uses pinned fixed registers:

- `mem0_base`
- `mem0_size`

For `memidx == 0`, lowering can avoid repeated runtime-view loads. On guarded
64-bit configurations, many accesses can also skip explicit bounds checks and
rely on guard pages, except for cases that still need an explicit multiword GP
check.

Why it belongs here:

- this is the first stage that knows both the memory opcode shape and the fixed
  register plan

### Optimization: preserve address-generation shape for later fusion

Memory lowering intentionally emits address arithmetic in a recognizable form:

```text
extend index
add optional offset
add base
load/store
```

That is not accidental. It sets up the later MachineIR peephole to recover
indexed addressing modes.

### Optimization: 32-bit `i64` legalization stays compact

On 32-bit GP targets, `Gp32Lowering` keeps `i64` work in pair-aware MachineIR
ops such as:

- `Int64PairBinary`
- `Int64PairDivRem`
- `Int64PairShift`
- `Int64PairCompare`

Why this is better than exploding immediately:

- the shared MachineIR stays compact
- backends can lower pair ops directly with carry/borrow-aware sequences
- the one-pass lowerer avoids manufacturing many scratch temporaries too early

On 64-bit GP targets, `Gp64Lowering` is deliberately thin and uses the scalar
path directly.

### Optimization-enabling structure: machine constant pool

`ConstPoolBuilder` moves runtime-call metadata and other read-only records out
of the instruction stream into the machine module's constant pool.

What this optimizes:

- keeps MachineIR compact
- avoids repeating metadata payloads in every instruction
- lets runtime calls operate on canonical frame regions without bloating the ISA layer

## 4. Machine IR Optimization

Main code:

- `sf-nano-core/src/vm/machine/machine_ir/module.rs`
- `sf-nano-core/src/vm/machine/peephole/mod.rs`
- `sf-nano-core/src/vm/machine/peephole/*.rs`

This stage is intentionally small. It wins because MachineIR is already shaped
to expose profitable local patterns.

Pass order today (block-local unless noted):

1. constant deduplication (`deduplicate_constants`)
2. store-to-load forwarding (`forward_stored_values`)
3. load-to-load reuse (`reuse_loaded_values`)
4. indexed memory fusion (`fuse_indexed_memory`)
5. copy propagation (`copy_propagate`)
6. instruction-selection fusion (`fuse_isel`)
7. signed 32x32→64 multiply recovery on 32-bit GP targets
   (`fuse_smull_sign_ext`)
8. compare-and-branch fusion (`fuse_compare_branch`, runs across whole
   program because it needs successor-block liveness)

### Optimization: constant deduplication

Duplicate constant materializations in the same block are turned into copies
from the first materialization.

Example:

```text
move r1 <- 42
...
move r6 <- 42
```

becomes:

```text
move r1 <- 42
...
move r6 <- r1
```

### Optimization: store-to-load forwarding

Pattern:

```text
store [addr] <- x
...
load y <- [addr]
```

becomes:

```text
move y <- x
```

when no intervening op invalidates the address or source.

### Optimization: load-to-load reuse

Pattern:

```text
load r1 <- [addr]
...
load r2 <- [addr]
```

becomes:

```text
load r1 <- [addr]
...
move r2 <- r1
```

when no intervening store can alias.

### Optimization: indexed memory fusion

`fuse_indexed_memory()` recognizes the address-generation patterns emitted by
the lowerer and replaces them with `IndexedLoad` / `IndexedStore`.

Simple example:

```text
i64.extend_i32_u r3 <- idx
i64.add r3 <- r3, #16
i64.add r3 <- base, r3
load r4 <- [r3]
```

becomes:

```text
indexed_load r4 <- [base + uxtw(idx) + 16]
```

Why it works so well:

- the previous stage preserved this shape on purpose
- the peephole does not need alias analysis beyond short local checks

### Optimization: copy propagation

This pass removes transient-to-transient copies and rewrites later uses to the
original register.

It also folds single-use `move r <- Imm64(C)` into consumer operands as inline
immediates when legal.

### Optimization: instruction-selection fusion

`fuse_isel()` creates higher-level MachineIR ops that map well to real ISAs:

1. `ShrU + And(mask)` -> `BitfieldExtractU`
2. `shift + binop` -> `IntBinaryShifted`
3. `And(mask) + compare-with-zero` -> `TestBits`

Simple example:

```text
shr.u r1 <- src, #8
and   r2 <- r1, #255
```

becomes:

```text
ubfx-like r2 <- src, lsb=8, bits=8
```

Why it belongs here:

- the backend wants these fused nodes
- earlier stages should stay ISA-neutral
- later native emission can map them to one instruction on ARM64/ARM32/RV64
  where the ISA has a matching form, or decompose them if necessary

### Optimization: signed 32x32→64 multiply recovery (32-bit GP only)

`fuse_smull_sign_ext()` replaces an `Int64PairBinary{Mul}` whose operands
both come from `i64.extend_i32_s` with a single `Int64MulFromSignExt32` op,
so 32-bit GP backends can emit one signed-long-multiply instruction
(`SMULL` on ARM32) instead of a 64×64 pair multiply.

This only fires on targets that carry the `Int64PairBinary` form — 64-bit
GP targets skip it entirely.

### Optimization: compare-and-branch fusion

`fuse_compare_branch()` removes materialized boolean values when a compare feeds
only the branch.

Pattern:

```text
cmp r5 <- (a < b)
branch if r5
```

becomes:

```text
branch if (a < b)
```

This is run last because it needs successor-block liveness checks. In shared
MachineIR this fusion is intentionally limited to integer/test-bit conditions;
float compares are not fused here because x86_64 needs extra NaN-handling
steps that do not fit a single generic branch condition.

## 5. Machine IR -> Native Code

Main code:

- `sf-nano-core/src/vm/arch/common/pipeline.rs`
- `sf-nano-core/src/vm/arch/common/scratch_pool.rs`
- `sf-nano-core/src/vm/arch/abi.md`
- `sf-nano-core/src/vm/arch/common/core.rs`
- `sf-nano-core/src/vm/arch/arm64/*`
- `sf-nano-core/src/vm/arch/arm32/*` (shared by the `armv7a` and `thumbm`
  backends)
- `sf-nano-core/src/vm/arch/riscv/*` (shared RISC-V ABI, encoder, and
  register mapping)
- `sf-nano-core/src/vm/arch/riscv64/*`
- `sf-nano-core/src/vm/arch/riscv32/*`
- `sf-nano-core/src/vm/arch/x86_64/*`
- `sf-nano-core/src/vm/arch/emulator/*` (debug MachineIR execution backend,
  used for testing and the `emu64` / `emu32` configs)
- `sf-nano-core/src/vm/runtime/runtime_call/*`
- `sf-nano-core/src/vm/runtime/preserved/*`

This stage is late by design. The shared pipeline keeps enough semantic shape
alive so each backend can choose the best encoding at the last responsible
moment.

### Optimization: prologue loads fixed fast-path state once

The prologue loads the pinned runtime state (`ctx`, `fp`, `mem0_base`,
`mem0_size`) into fixed registers once per function entry.

That makes subsequent memory and runtime access cheaper without repeating setup
inside the body.

### Native lowering invariants

This stage is also where the backend ownership rules matter most. These rules
exist to keep native lowering honest: the backend is selecting encodings, not
reinterpreting MachineIR liveness or inventing a second register allocator.

Shared ABI background lives in `sf-nano-core/src/vm/arch/abi.md`. The rules
below are the practical coding invariants that current native backends are
expected to follow.

#### 1. MachineIR register classes are not interchangeable

- Fixed MachineIR registers (`ctx`, `fp`, `mem0_base`, `mem0_size`) are always
  live machine state. They are never free temporaries.
- Cached-local registers are register views of canonical frame-slot locals.
  They are an optimization, not disposable scratch.
- Transient registers are MachineIR-owned SSA values. If the backend clobbers
  them outside the agreed boundary protocol, it is corrupting JIT state.
- Scratch-only registers are the backend's real temp pool. If an operation
  needs an ad hoc temporary, it must come from scratch allocation rather than
  from a mapped transient or cached-local register.

The practical consequence is simple: backend lowering must not treat "any
caller-saved physical register" as free. Only registers that the backend's
`abi.rs` marks as scratch are free backend temps.

#### 2. Frame slots remain the canonical state

Frame slots are the source of truth for:

- locals
- spilled deep-stack values
- call arguments and results
- local call-link records

Registers are normally execution caches over that frame state. This is what
makes the boundary rules safe:

- MachineIR can publish cached locals before a boundary
- local calls can reuse the caller operand region as callee frame prefix
- runtime calls can read and write frame spans directly
- preserved helpers can save caller-clobbered JIT state and then operate on a
  native-stack I/O window

The one intentional exception is a compiled local-call success edge that names a
non-ref cached local in a preserved dynamic register. That is still explicit
MachineIR state, not hidden backend preservation. Ref-typed cached locals must
be frame-visible before any boundary where a callee/runtime helper could need
root visibility.

#### 3. Scratch registers must come from the scratch pool

`sf-nano-core/src/vm/arch/common/scratch_pool.rs` is the ownership mechanism
for backend scratch use.

Use it this way:

- Prefer `scoped_alloc()` for local, lexical scratch use.
- Use `detach()` when a scratch reservation must survive later `&mut self`
  emission calls but should still free itself by RAII.
- Use `alloc()` / `reg()` / `free_index()` only for the rare cases that
  genuinely need manual protocol-scoped ownership.

The important invariant is that "keep using this register after the lexical
guard ends" must still keep the pool slot reserved somehow. That is exactly
what `detach()` and the explicit `alloc/free_index()` path are for.

Backends should also keep the pool honest by calling `assert_all_free()`
between instructions, so leaks or accidental long-lived scratch capture fail
early in debug builds.

#### 4. Regular lowering must not spell physical registers directly

Higher-level backend lowering should work in terms of:

- mapped MachineIR registers
- scratch-pool allocations
- semantic encoder helpers

not in terms of hard-coded physical register names.

ARM64 and RISC-V are the current examples of the intended structure:

- the physical register plan lives in `abi.rs`
  (`sf-nano-core/src/vm/arch/arm64/abi.rs`,
  `sf-nano-core/src/vm/arch/riscv64/abi.rs`,
  `sf-nano-core/src/vm/arch/riscv32/abi.rs`, etc.)
- raw register construction is hidden there
- lowering code gets temps from the scratch pool
- zero/SP-like forms are expressed through semantic helpers instead of raw
  register spellings

This rule prevents a subtle but recurring class of bugs where backend code
"borrows" a register that is actually part of the mapped MachineIR register
file.

#### 5. Foreign ABI registers are boundary-only

Registers such as `C_ARG*` and `C_RET*` are foreign ABI facts, not extra
MachineIR register classes.

They may overlap caller-saved transient or scratch registers, but that overlap
is only safe at the actual foreign boundary, after MachineIR has already made
dynamic state unavailable there.

That means:

- regular lowering must not use `C_ARG*` / `C_RET*` as general temps
- runtime-call lowering may use them while entering the runtime-call
  entry
- preserved-helper lowering may use them while entering the preserved runtime
  entry
- once the boundary sequence ends, those registers go back to being ordinary
  physical registers with no special backend privilege

This is why ARM64 `X0` / `X1` / `X2`, or RV64 `A0` / `A1` / `A2`, are
dangerous outside real runtime-call glue even though the ISA itself would let
the backend use them freely.

#### 6. MachineIR is the shared / arch boundary

MachineIR is the bottom of the shared pipeline and the top of the
arch-dependent backend pipeline.

That means a feature belongs in MachineIR only when the semantics below it are
still shared across targets. A MachineIR op should represent Wasm or shared JIT
behavior, not one backend's current helper strategy.

Use this rule:

- if the operation's semantics are still platform-independent, keep it above or
  at MachineIR
- if different targets may choose between native instructions and helper
  fallback for the same MachineIR op, that choice belongs below MachineIR
- do not add a new MachineIR op just because one backend currently needs a
  helper call

Calls are the main place where this matters:

- `Call` is the MachineIR compiled-call form because it represents real Wasm
  call control transfer into another compiled frame
  - `Call { target: Direct(..) }` is for a compile-time-known compiled callee
  - `Call { target: Indirect { .. } }` is for a runtime-resolved compiled
    callee, whether that callee lives in the current module or in another
    linked compiled module
- `CallRuntime` is a MachineIR instruction because it still represents Wasm call
  semantics, just through the runtime-dispatch path that round-trips back to
  the next instruction
- preserved helpers such as `memory.grow`, `table.grow`, `ref.test`,
  `struct.get`, and similar helper-backed ops are not MachineIR call forms;
  they are backend lowering choices for ordinary MachineIR instructions

Current call-shape split:

| Endpoint after shared dispatch | MachineIR form |
| --- | --- |
| compile-time-known compiled callee | `Call { target: Direct(..) }` |
| runtime-resolved compiled callee | `Call { target: Indirect { .. } }` |
| host or runtime-dispatch path | `CallRuntime` |

#### 7. Runtime calls and preserved helpers are different boundary systems

The current JIT intentionally keeps two runtime boundary systems:

- Runtime-call system:
  - triggered only by Wasm `call` / `call_indirect` / `call_ref` sites that
    must round-trip through runtime dispatch instead of transferring into
    another compiled MachineIR function
  - this includes calls that resolve to host/runtime targets such as WASI, and
    any dynamic target that is not representable as a compiled-frame transfer
  - lowered by MachineIR as an inline runtime call
  - uses frame slots as its argument/result transport
  - implemented under `sf-nano-core/src/vm/runtime/runtime_call/`
- Preserved-helper system:
  - triggered by engine-internal helper-backed operations such as
    `memory.grow`, `memory.copy`, `table.grow`, `table.init`, `ref.test`,
    `ref.cast`, `struct.get`, `struct.set`, and similar ops
  - owned by native backends rather than MachineIR
  - uses a fixed native-stack I/O layout
  - implemented under `sf-nano-core/src/vm/runtime/preserved/`

They share some low-level status/error plumbing, but they are not one generic
"helper" mechanism. The JIT should keep those boundaries explicit so readers do
not confuse Wasm call dispatch with engine-internal preserved operations.

#### 8. Boundary-specific safety rules

The safe register/use policy is different at each boundary:

| Context | Safe backend-owned temps | What must already be true |
| --- | --- | --- |
| Regular instruction lowering | scratch-pool regs only | mapped fixed/cache/transient regs keep their MachineIR meaning |
| Local JIT-to-JIT call transfer | scratch-pool regs only | MachineIR already prepared frame setup and call-link state |
| Runtime-call entry sequence | scratch-pool regs plus foreign `C_ARG*`/`C_RET*` as part of the ABI sequence | transients are dead, cached locals published, fixed regs remain live |
| Preserved-helper entry sequence | scratch-pool regs plus foreign `C_ARG*`/`C_RET*` inside the preserved-helper protocol | preserved wrapper saves the caller-clobbered JIT state it needs before reusing those registers |

One subtle consequence is that "caller-saved" does not mean "free right now".
The backend may only treat a caller-saved physical register as disposable if it
is either:

- a dedicated scratch register, or
- being used inside a boundary protocol that has already made the relevant
  MachineIR state unavailable there

#### 9. Delicate paths should document why they are delicate

Some control-flow glue is sensitive because of overlapping frame regions. A
good example is local return lowering: the backend may need to capture
continuation and caller-frame information before copying results back, because
the caller result window can overlap and clobber call-link slots.

Those cases are exactly where the backend should use comments and explicit
scratch ownership rather than "obvious" rewrites. The goal is to make the
correctness condition visible to the next person touching the code.

### Optimization: backend immediate selection

Backends try native immediate forms before falling back to scratch
materialization.

Backends recognize target-specific immediate forms, among others:

- add/sub immediate forms on ARM64 and RV64
- logical immediates
- shift immediates
- multiply by powers of two as shift
- compare-immediate forms

Simple example:

```text
i64.add dst, lhs, #5
```

can become a single `ADD imm` instead of "materialize 5 into a temp; ADD reg".

Why it belongs here:

- only the backend knows exact encodable immediate patterns
- earlier IR should stay portable

### Optimization: fused MachineIR maps to native addressing/ALU forms

The fused MachineIR ops created earlier now pay off:

- `IndexedLoad` / `IndexedStore` -> native indexed addressing modes
- `BitfieldExtractU` -> `UBFX` on ARM64/ARM32, or shift/mask sequences where
  the ISA lacks a dedicated bitfield extract
- `IntBinaryShifted` -> barrel-shifter forms on ARM64/ARM32, or decomposed
  shift+ALU forms elsewhere
- `TestBits` -> `TEST`/`TST`
- branch conditions reuse compare/test flags directly

This is exactly why the middle of the pipeline preserves these patterns instead
of lowering everything into the smallest possible primitive ops too early.

### Optimization: ARM64 zero-store pair fusion

`zero_store_pair_fusion()` looks for adjacent zero stores to neighboring
8-byte-aligned addresses and emits one `STP XZR, XZR`.

Simple example:

```text
store [base+0]  <- 0
store [base+8]  <- 0
```

becomes one paired store.

Why it belongs here:

- this is a pure encoding-level win
- it depends on exact ISA store-pair constraints

### Optimization: edge stubs use parallel-move resolution

`emit_parallel_moves()` handles block-parameter transfers as true parallel
moves, including cycle breaking with scratch registers.

This minimizes extra block-entry shuffle code and avoids serializing moves in a
way that would clobber sources.

### Optimization: shared tail blocks reduce repeated epilogues and traps

The native pipeline emits shared tail regions for:

- normal return
- stack overflow trap
- error return
- deferred traps

This is mostly a code-size and locality optimization, but it also keeps block
emission simple and regular.

## What The Pipeline Is Optimizing For

The engine is not trying to outsmart Wasm with a large speculative optimizer.
It is trying to make the hot path cheap:

- hot locals stay in pinned registers
- deep stack values get canonical slots early
- calls become explicit publish/reload boundaries
- transient register pressure is capped before backend lowering starts
- late peepholes recover the native forms that the structured lowering set up
  on purpose

That combination is why the pipeline can stay simple while still producing very
competitive code: the design pushes expensive global problems upward into cheap
structural choices, and pushes ISA-specific cleverness downward into small local
rewrites.
