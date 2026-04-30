# Block-Streaming Middle Plan

## Goal

Make the Wasm -> SSA middle stage usable in a block-streaming compile path for
large functions on memory-constrained targets.

The mode should preserve the existing middle architecture as much as possible:

- keep `SemanticOpKind`, `SsaInst`, `SsaBlock`, `SsaTerminator`, frame slots,
  and MachineIR lowering mostly unchanged
- keep the current full optimizer for normal functions
- add a lower-quality fallback for functions that exceed the compile RAM budget
- avoid full-function SSA, full-function MIR, and per-edge cache repair in the
  fallback path

The intended fallback is not "no local cache". It is the older uniform-boundary
model:

- one function-global local-cache set
- one function-global transient-stack register budget
- every block enters and exits with the same local-cache contract
- edges do not repair cache state because all blocks agree on it
- deeper Wasm stack values use canonical operand slots

This should recover most of the old performance profile while enabling
block-by-block compilation.

## Current Blockers

The current middle pipeline is internally close to block-at-a-time lowering, but
the surrounding contracts are whole-function:

- `cfg::build_semantic_cfg` builds a complete semantic CFG.
- `JointPlanner::build` computes per-block public cache sets.
- `rewrite::rewrite_function` lowers blocks in a loop, but accumulates all
  `SsaBlock`s, all value types, all block params, all cache-entry sets, and all
  call/primitive pools before returning.
- `insert_boundary_repair_blocks` needs the whole SSA program plus every
  predecessor exit cache set.
- `cleanup`, `sink_plan`, and some MIR peepholes are whole-CFG passes.
- Machine lowering computes `explicit_cached_locals`,
  `compute_block_entry_cache_params`, and block-entry dirty bits by scanning the
  complete `SsaProgram`.

The uniform-boundary fallback removes the hardest whole-function dependency:
per-edge cache reconciliation. It still needs compact block metadata, but it
does not need a full SSA body in memory.

## Streaming Mode Contract

Add an explicit preparation mode:

```rust
enum PrepareMode {
    Full,
    UniformBoundaryStreaming,
}
```

`Full` keeps the current path.

`UniformBoundaryStreaming` uses these invariants:

1. Every block has the same public cached-local set.
2. Every block opens with the same fixed split of dynamic registers:
   transient values first, local cache from the remaining budget.
3. Every block exits with the same cached-local residency as it entered.
4. SSA edge bindings carry only transient stack suffix values, never cache
   repair values.
5. Any Wasm stack value below the transient suffix is published in the canonical
   operand slot for its stack index.
6. Calls, runtime helpers, throws, tail calls, and other boundaries publish
   transient values exactly as they do today.
7. Whole-CFG cleanup and sink planning are skipped.

This means the target block does not care which predecessor was taken. The
local-cache part of the entry state is identical for every predecessor.

## Budget Split

The split must be target- and function-aware. The local cache cannot consume
registers that are needed to make a single Wasm operation legal, especially on
32-bit GP targets where `i64` values consume two GP units.

Treat this as two independent register classes:

- GP dynamic bank
- FP dynamic bank

For each class:

```text
total = backend middle-visible dynamic units
floor = max(backend_floor, function_operation_floor)
transient_units = min(total, floor + slack)
cache_units = total - transient_units
```

Where:

- GP `total` is `BackendConfig::allocatable_gp_dynamic_budget()`, not raw
  `gp_dynamic_budget`, because the backend already reserves a scratch tail.
- FP `total` is `BackendConfig::fp_dynamic_budget`.
- `backend_floor` is a small target-independent floor:
  - GP64: 3 units
  - GP32: 5 units
  - FP: 3 units when FP exists, otherwise 0
- `function_operation_floor` is computed by a lightweight type-stack scan over
  the function.
- `slack` starts small:
  - GP32: 1 unit
  - GP64: 1 unit
  - FP: 0 or 1 unit

The exact constants are tuning knobs. The important rule is transient-first:
if a function needs five GP units for a legal operation on a six-unit ARM32
allocatable bank, the local cache only gets one GP unit.

### Operation Floor

The operation floor is a correctness floor, not an optimization estimate. It
answers: how many transient units can a single lowered operation require at
once after the rewriter has spilled unrelated stack values?

Use the same stack-type simulation infrastructure that currently feeds
`compute_lightweight_plan`, but store only maxima.

For each semantic op:

1. Determine the operand types consumed at the top of the stack.
2. Determine the result types produced.
3. Convert types to GP/FP units using `gp_value_budget_units`.
4. Classify the lowering shape:
   - result can reuse dead operand register
   - result must be allocated in addition to all operands
   - no result
   - call-like / slot-backed boundary
5. Update per-bank maxima.

Useful initial approximations:

```text
ordinary unary/binary arithmetic:
    max(operand_units, result_units)

select:
    operand_units
    # GP32 i64 select is the important case:
    # true i64 pair + false i64 pair + i32 cond = 5 GP units.

store / memory.copy / memory.fill / table.copy:
    operand_units

current StructNew / ArrayNewFixed:
    operand_units + result_units
    # current MachineIR lowerer collects all fields/elements before allocating
    # the result and does not reuse dead inputs.

current ArrayFill / ArrayCopy / ArrayInit*:
    operand_units

call / return_call / throw:
    0 for streaming split purposes
    # operands are published to canonical slots before the boundary.
```

If `function_operation_floor > total`, the current register-based lowering
cannot guarantee compilation for that operation even with local cache disabled.
Do not silently pick an impossible split. The staged plan below handles this
with an oversized-op fallback.

### GP32 `i64`

On 32-bit GP targets, every `i64` transient consumes two GP units. Important
floors:

- `i64.const`: 2 GP
- `i64` unary: 2 GP
- `i64` binary: 4 GP
- `i64` compare: 4 GP
- `i64.select`: 5 GP
- `i64` load from an i32 address: at least 2 GP, normally legal by reusing the
  address register as one result half
- `i64` store: address plus value pair, usually 3 GP

The GP32 backend floor should therefore start at 5, not 2 or 3. That prevents
the first bring-up from allocating too many locals into cache and then failing
on a legal `i64.select`.

### Local Cache Selection

For bring-up, do not do hot-local analysis. Pick locals by index:

```text
for local_index in 0..local_count:
    if local type fits remaining cache units for its bank:
        cache it
```

This is simple and deterministic. Params come first because Wasm locals are
`params ++ non_param_locals`, which is a reasonable first approximation.

Unit cost:

- `i64` on GP32: 2 GP cache units
- all other GP/ref values: 1 GP cache unit
- `f32`/`f64`/`v128`: 1 FP cache unit

Avoid choosing a cached local if doing so would reduce transient units below
the computed floor.

## Middle Changes

### 1. Add `PrepareMode`

Replace or extend `PrepareInput.full_optimization` with a mode:

```rust
pub(crate) enum PrepareMode {
    Full,
    UniformBoundaryStreaming,
}
```

The RAM-budget decision in `vm/build.rs` should choose:

- `Full` when the function estimate fits
- `UniformBoundaryStreaming` when it does not

`full_optimization` can remain as a derived boolean internally during the first
patch if that keeps the diff small.

### 2. Add Uniform Planner

Add a second planner constructor:

```rust
JointPlanner::build_uniform_boundary(...)
```

It should produce the same `FunctionPlan` shape as today, but with different
contents:

- `gp_dynamic_budget = transient_gp_units`
- `fp_dynamic_budget = transient_fp_units`
- every block gets the same `tentative_entry_cached_locals`
- every block entry uses the uniform transient suffix policy

The planner can reuse the existing lightweight stack simulation to produce
block entry stack types. The difference is that `spill_depth` is not derived
from per-block pressure solving:

```text
entry_live_suffix = largest stack suffix that fits transient GP/FP units
spill_depth = stack_height - entry_live_suffix.len()
live_types = stack_types[spill_depth..]
```

This preserves block params for a bounded top-of-stack suffix. Deeper values
are always in frame operand slots.

### 3. Force Uniform Cache At Block Exit

At the end of each lowered block, before lowering the terminator edge, restore
the global cache contract:

```rust
ensure_uniform_cache_boundary(
    global_cached_slots,
    resident_cache,
    materialized_cache,
    state.ops,
)
```

Rules:

- For a slot in the global cached set that is not resident, emit
  `LocalEnsureCache(slot)` if the block entry requires a real value.
- If a write-first reserve mode is later added, emit `LocalReserveCache(slot)`
  instead when legal.
- For bring-up, use `Ensure` for all global cached locals. It is less optimal
  after calls, but simple and correct.
- Do not keep non-global cached locals live at exit. The initial uniform mode
  should never introduce them.

Calls currently clear `resident_cache` and `materialized_cache`. This exit
restore is what makes a block that contains a call still satisfy successor
entry requirements without a repair block.

### 4. Skip Repair Blocks

In `UniformBoundaryStreaming`, do not call:

```rust
insert_boundary_repair_blocks(...)
```

If every block entry and exit uses the same cached-local set, repair blocks are
dead weight. Keeping them would also force whole-function SSA retention.

### 5. Conservative Local Init

The full mode uses `locals_reads_before_write` to avoid unnecessary zero-init.
Streaming mode should initially skip that proof and mark every non-param local
as `reads_before_write = true`.

There is one important entry-cache ordering issue:

- `BlockLowerContext::new` currently materializes entry cached locals.
- `lower_function_into_sink` emits zero-init locals after constructing the
  `BlockLowerContext`.

If a cached non-param local is loaded from frame before zero-init, it can read
an uninitialized frame slot. Fix this before caching non-param locals at
function entry.

Acceptable fixes:

1. Move zero-init before entry cache materialization.
2. Teach entry cache materialization to synthesize zero directly for cached
   non-param locals that require zero-init.
3. Bring-up shortcut: only cache params until the ordering is fixed.

The final uniform mode can cache locals by index once this ordering is correct.

### 6. Keep Block-Local Optimization

Safe in streaming mode:

- inline spill/fill pressure handling
- block-local constant folding
- block-local MachineIR peepholes

Skip in streaming mode:

- `cleanup::cleanup_program`
- `sink_plan::plan_sinks`
- cross-block MachineIR peepholes
- per-edge cache repair
- region solver

## Machine Lowering Changes

The first milestone can still build a full `SsaProgram` with uniform cache
state. That proves correctness with minimal changes.

True streaming needs a block sink:

```rust
type SsaBlockSink<'a> =
    dyn FnMut(SsaBlock, SsaBlockMeta, &SsaSharedPools) -> Result<(), WasmError> + 'a;
```

Then lower:

```text
semantic block -> SsaBlock -> block-local SSA opt -> MIR block(s) -> MIR block opt -> arch sink
```

To avoid rebuilding whole-program cache planning in MachineIR, add a uniform
cache metadata path:

- `cached_locals`: known from the uniform planner, ordered by local index
- `entry_cache_params`: same for every non-entry block
- entry block materializes the same cached locals from frame or zero
- non-entry blocks receive cached locals as edge params
- dirty bits can be conservative: all incoming cached values start dirty

This bypasses these whole-program scans in the streaming path:

- `explicit_cached_locals(&SsaProgram)`
- `compute_block_entry_cache_params(...)`
- `compute_block_entry_cache_dirty(...)`

For bring-up, keeping those scans is acceptable if the path still builds a full
`SsaProgram`. For true MCU streaming, they need uniform metadata inputs.

## Semantic Frontend Plan

The shortest proof path still decodes a full `SemanticProgram`. That is useful
for validating the uniform planner and rewriter with existing tests.

The end-state does not have to decode twice. With the uniform boundary contract,
the preferred MCU path can decode once and emit block-by-block. Two-pass decode is
only an optional tuning path when the compiler wants function-specific summaries
before lowering block 0.

### Preferred Path: Single-Pass Decode And Emit

Use the Wasm function header and backend config to choose a conservative uniform
contract before decoding the body:

- local/result types are known from the function type and locals declaration
- local-cache set can be selected by local index without scanning the body
- transient floor can come from backend target floors plus conservative opcode
  class limits
- branch target stack shapes are available from the structured control stack and
  block types while decoding

Then decode the body once:

1. maintain the Wasm validation/control stack
2. decode semantic ops for the current basic block only
3. lower to one `SsaBlock`
4. run block-local SSA optimization
5. lower to MIR through the existing streaming MIR sink
6. run block-local MIR peephole
7. hand the block to arch streaming emit
8. drop the SSA/MIR block

Forward branches do not require a prior whole-function block directory in this
mode. Wasm's structured control stack tells the decoder the target frame for
`br`/`br_if`/`br_table`, and the uniform cache contract means every edge carries
the same cached-local shape. Any unresolved physical label can be patched by the
existing arch/MIR label machinery.

### Optional Path: Compact Pre-Scan

Use a compact pre-scan only if we want better per-function choices without
retaining full middle IR:

#### Pass A: Compact Scan

Build a compact `FunctionStreamSummary`:

- max stack height
- local/result types
- block directory: block id, semantic range, successor ids
- entry stack type snapshot per block
- uniform transient floor per bank
- local-cache set by index
- return-result span shape

This pass stores O(blocks + locals + control stack), not O(ops).

#### Pass B: Block Decode And Emit

Decode the function body again and lower one block at a time:

1. decode semantic ops for the current block only
2. lower to one `SsaBlock`
3. run block-local SSA optimization
4. lower to MIR through the existing streaming MIR sink
5. run block-local MIR peephole
6. hand the block to arch streaming emit
7. drop the SSA/MIR block

The Wasm bytes are already available in `FunctionSpec::code()`, so a second
decode pass is cheaper than retaining full middle IR on MCU targets when the
extra summary quality is worth the code complexity.

## Oversized Operation Fallback

Some operations can require more operands than a small backend can hold:

- `array.copy`: five GP-like operands
- `array.fill`: four operands plus possible `i64` payload
- `struct.new` with many fields
- `array.new_fixed` with many elements

If the computed operation floor exceeds available transient units, there are
three options:

1. For bring-up, disable streaming for that function and use the full path.
2. If the full path would still fail due to register pressure, route the op
   through a slot-backed runtime helper.
3. Add specialized lowerings that stream operands from operand slots instead
   of requiring every operand as an SSA register.

The robust MCU answer is option 2 or 3. The minimal initial answer is option 1
plus a clear diagnostic path, because this case is rare for C-like Wasm
workloads.

## Rollout Plan

### Milestone 1: Uniform Mode, Buffered SSA

Goal: prove the contract.

- Add `PrepareMode`.
- Add `JointPlanner::build_uniform_boundary`.
- Compute function transient floors.
- Select cached locals by local index.
- Make all blocks use the same cache-entry set.
- Restore uniform cache at block exits.
- Skip repair blocks in uniform mode.
- Keep returning a full `SsaProgram`.
- Run existing native and emulator tests.

Expected result: less optimized code, no streaming memory win yet, but cache
repair and region-solving are removed from the fallback mode.

### Milestone 2: SSA Block Sink

Goal: drop SSA blocks after use.

- Refactor `rewrite_function` so the block loop can call a sink.
- Keep current `SemanticProgram` input.
- Use uniform metadata so the MachineIR lowerer does not need whole-program
  cache planning.
- Reset or compact SSA value numbering per block if no cross-block value
  survives beyond edge params.

Expected result: middle no longer retains all prepared SSA blocks in fallback
mode.

### Milestone 3: Compact Semantic Scan + Re-decode

Goal: avoid retaining full semantic ops.

- Add pass-A summary builder.
- Add pass-B block decoder/lowerer.
- Keep full semantic decode for `Full` mode and tests.
- Use the uniform planner data from pass A.

Expected result: Wasm -> SSA -> MIR -> arch is block-streaming for large
functions.

### Milestone 4: Oversized Slot-Backed Slow Paths

Goal: guarantee compilation even when an op needs more registers than the
target has.

- Add slot-backed runtime/helper lowering for high-arity GC/table/memory ops.
- Use it only when `operation_floor > available_transient_units`.
- Keep normal register lowering otherwise.

Expected result: large or awkward functions compile instead of failing, with
localized performance loss.

## Test Plan

Add middle tests for:

- uniform mode inserts no cache repair blocks
- every block has identical `block_entry_cached_slots`
- a block containing a call restores the uniform cached-local set before edge
  transfer
- GP32 `i64.select` leaves enough transient budget and reduces cache count
- local-index cache selection handles `i64` locals as two GP units on GP32
- non-param cached locals are initialized correctly at function entry
- streaming mode skips cleanup/sink planning but still validates SSA

Add integration tests for:

- ARM32/RISC-V32 integer code with `i64` arithmetic and `i64.select`
- loops with cached params crossing backedges
- calls inside loops with cached locals
- memory/table ops with 3-5 operands
- a function forced into streaming mode via a tiny compiler RAM budget

## Expected Tradeoffs

Quality losses in uniform mode:

- no loop-specific cache set
- no edge-specific reserve/ensure repair
- no cross-block sink planning
- more spill/fill around block boundaries
- more cache reloads after calls
- no whole-CFG cleanup

Quality wins over no-cache fallback:

- hot early locals, especially params, still stay in registers
- loops with stable local access keep their cache state without repair edges
- block streaming is possible without throwing away the local-cache design

This matches the old pipeline's expected shape: lower quality than the current
region-planned cache, but much better than disabling cached locals entirely.
