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
| Wasm -> Semantic IR | Preserve Wasm structure and types | Inline tiny leaf callees before frame/register decisions exist |
| Semantic IR -> Prepared SSA-IR | Make slots, transient budget, and call boundaries explicit | Cheap local forwarding, constant folding, sink planning, and low-cost CFG cleanup |
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

- `sf-nano-core/src/vm/wasm/decode.rs`
- `sf-nano-core/src/vm/wasm/inline.rs`
- `sf-nano-core/src/vm/wasm/sir/semantic_ir.rs`

### Optimization-enabling representation

`decode_to_semantic_ir()` keeps Wasm-specific structure intact:

- structured control markers (`Block`, `Loop`, `If`, `Else`, `End`)
- abstract locals (`LocalGet`, `LocalSet`, `LocalTee`)
- semantic calls (`CallInternal`, `CallExternal`, `CallIndirect`)
- typed result information (`local_types`, `result_types`, `op_result_types`)
- `max_stack_height`

This is not an optimization pass by itself. It is an optimization-friendly
representation choice.

Why it lives here:

- The frontend still knows exact Wasm structure.
- No frame slots, cache registers, or transient budgets exist yet.
- Later passes can reason about loops, calls, and locals without reverse
  engineering them from low-level code.

### Optimization: small leaf inlining

`inline_calls_in_function()` replaces eligible `CallInternal` sites with the
callee body.

Current policy:

- callee must be a leaf (no nested calls)
- at most `200` semantic ops
- at most `8` parameters
- fixed-point iteration is used, so transitive chains are fully inlined

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

Main code:

- `sf-nano-core/src/vm/middle/mod.rs`
- `sf-nano-core/src/vm/middle/frame.rs`
- `sf-nano-core/src/vm/middle/local_cache.rs`
- `sf-nano-core/src/vm/middle/spill_plan.rs`
- `sf-nano-core/src/vm/middle/lower_cfg.rs`
- `sf-nano-core/src/vm/middle/lower_ops.rs`
- `sf-nano-core/src/vm/middle/lower_term.rs`
- `sf-nano-core/src/vm/middle/thread_jumps.rs`
- `sf-nano-core/src/vm/middle/optimize.rs`
- `sf-nano-core/src/vm/middle/sink_plan.rs`
- `sf-nano-core/src/vm/middle/ssa_ir/ir.rs`

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

`analyze_local_cache_prefs()` decides which canonical locals are worth pinning
in dedicated cache registers.

What it does:

- scores locals by access frequency
- weights accesses inside loops more heavily
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

The same analysis computes `reads_before_write` for each cached local.

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

`continuation_skip_reload()` computes, for each call site, which cached locals
will be overwritten before they are read after the call.

What it does:

- marks cached locals that do not need to be reloaded at the continuation
- avoids useless "save before call, reload after call, overwrite immediately"
  traffic

Simple example:

```wat
call $foo
local.set 0, ...
```

If local `0` is cached and not read before that `local.set`, the continuation
can skip reloading local `0`.

Why it belongs here:

- it relies on semantic-local reads/writes in the straight-line region after
  the call
- later machine lowering only knows slots and registers, not the semantic read
  vs write intent

### Optimization-enabling structure: explicit spill/fill planning

`prepare_semantic_ops()` constrains the live transient window to the configured
GP/FP budgets and inserts explicit `Spill` / `Fill` actions when needed.

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

`build_block_ranges()` and `retain_reachable_blocks()` turn structured control
into a flat basic-block CFG and drop blocks that can never execute.

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

### Optimization-enabling structure: local versioning and leaf/boundary split

Block lowering turns semantic ops into:

- `Value` ops for pure-ish arithmetic / compare / convert work
- `Boundary` ops for calls and runtime helpers
- `LocalGet` / `LocalSet`
- `Spill` / `Fill`

Every `LocalSet` also carries a monotonically increasing local version.

Why that matters:

- calls become exact barriers for later legality checks
- local versioning makes it cheap to prove whether an old local value is still
  live
- pure value ops remain easy targets for constant folding and sink planning

### Optimization: CFG simplification after SSA construction

`thread_jumps::simplify_cfg()` runs two cheap CFG cleanups:

1. jump threading through trivial empty goto blocks
2. single-predecessor block merging

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

### Optimization: slot-value forwarding

`forward_slot_values()` removes redundant local reloads when the stored value is
still live.

Simple example:

```text
local.set x, v0
local.get x -> v1
i32.add v0, v1
```

If `v0` already lives long enough, the `local.get` becomes an alias of `v0`,
eliminating the reload.

Important limits:

- only canonical local slots are forwarded
- `Spill` / `Fill` traffic for deep stack values is intentionally left explicit
- forwarding is blocked across `Boundary` ops

Why it belongs here:

- this is the last stage where local slot traffic is still explicit and typed
- later MachineIR may have already converted the local into cached-register
  aliases or moves

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

`plan_sinks()` finds `LocalSet` operations whose producer can write directly
into the local's cached home register.

Simple example:

```text
local.get x -> v0
i32.add v0, #1 -> v1
local.set x, v1
```

If no boundary and no intervening read of the old `x` exists, the add result
can be placed directly in `x`'s cache register and the `local.set` disappears.

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
- `sf-nano-core/src/vm/machine/lower_cached.rs`
- `sf-nano-core/src/vm/machine/lower_inst.rs`
- `sf-nano-core/src/vm/machine/lower_leaf_arith.rs`
- `sf-nano-core/src/vm/machine/lower_leaf_special.rs`
- `sf-nano-core/src/vm/machine/lower_boundary.rs`
- `sf-nano-core/src/vm/machine/gp32/lower_leaf.rs`
- `sf-nano-core/src/vm/machine/lower_i64_gp64.rs`

This stage is where the "simple pipeline" claim becomes concrete. Because the
SSA stage already bounded transient pressure and assigned canonical homes, the
Machine lowerer can stay one-pass and local.

### Optimization-enabling structure: fixed register-file partition

`MachineRegFile::new()` partitions the virtual register space as:

```text
[fixed | gp_local_cache | gp_transient | fp_transient | fp_local_cache]
```

Fixed registers include:

- runtime context pointer
- frame pointer
- `mem0_base`
- `mem0_size`

Why that matters:

- no later pass has to rediscover ABI roles
- cached locals and transient temps do not compete in an unconstrained pool
- one-pass allocation becomes possible

### Optimization: entry cached-local initialization

At entry, `emit_entry_cached_locals()`:

- loads cached parameters from frame slots
- zero-initializes only locals that may be read before write
- skips untouched cached locals entirely when `reads_before_write == false`

This is the backend realization of the earlier analysis.

### Optimization: `LocalGet` source aliasing

When a local is cached, lowering a `LocalGet` does not emit a load. It just
maps the SSA value to the cache register.

Simple example:

```text
local.get x -> v0
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
local.set x, v1
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

Because `Boundary` ops were already made explicit, this stage can use a simple
rule:

- no live transient values may cross a boundary
- cached locals are saved before call/helper boundaries
- cached locals are selectively reloaded after calls using `skip_reload`

The pipeline turns calls from "hard global allocation problem" into "publish to
slots, call, reload what is still needed".

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

### Optimization-enabling structure: sidecar metadata

`SidecarBuilder` moves helper metadata and constants out of the instruction
stream into sidecar tables.

What this optimizes:

- keeps MachineIR compact
- avoids repeating helper payload descriptions in every instruction
- lets helpers operate on canonical frame regions without bloating the ISA layer

## 4. Machine IR Optimization

Main code:

- `sf-nano-core/src/vm/machine/machine_ir/module.rs`
- `sf-nano-core/src/vm/machine/peephole/mod.rs`
- `sf-nano-core/src/vm/machine/peephole/*.rs`

This stage is intentionally small. It wins because MachineIR is already shaped
to expose profitable local patterns.

Pass order today:

1. constant deduplication
2. store-to-load forwarding
3. load-to-load reuse
4. indexed memory fusion
5. copy propagation
6. instruction-selection fusion
7. compare-and-branch fusion

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
- later native emission can map them to one instruction on ARM64/ARMv7a, or
  decompose them if necessary

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
- `sf-nano-core/src/vm/arch/common/core.rs`
- `sf-nano-core/src/vm/arch/arm64/*`
- `sf-nano-core/src/vm/arch/armv7a/*`
- `sf-nano-core/src/vm/arch/x86_64/*`

This stage is late by design. The shared pipeline keeps enough semantic shape
alive so each backend can choose the best encoding at the last responsible
moment.

### Optimization: prologue loads fixed fast-path state once

The prologue loads the pinned runtime state (`ctx`, `fp`, `mem0_base`,
`mem0_size`) into fixed registers once per function entry.

That makes subsequent memory and runtime access cheaper without repeating setup
inside the body.

### Optimization: backend immediate selection

Backends try native immediate forms before falling back to scratch
materialization.

ARM64 currently recognizes, among others:

- add/sub immediate forms
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
- `BitfieldExtractU` -> `UBFX` on ARM64/ARMv7a
- `IntBinaryShifted` -> barrel-shifter forms on ARM64/ARMv7a
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
