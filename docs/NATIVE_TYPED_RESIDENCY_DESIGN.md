# Typed Frontend Residency Design

This document proposes the frontend and lowering changes needed to make
floating-point values stay in FP transients all the way from prepared LIR
through MachineIR lowering.

The target is not a general register allocator. The target is a typed,
bank-aware version of the existing prepared pipeline:

- one ordered Wasm value stack
- canonical frame homes for locals and deep stack values
- a bounded GP transient bank
- a bounded FP transient bank
- explicit slot traffic when values leave the transient banks

This design is motivated by the current `c-ray` gap, where many float values
reenter the pipeline as untyped `u64` values and therefore bounce through GP
registers before the ARM64 backend can use them as FP operands.

## Status

This is a design proposal, not an implemented feature.

The first implementation stage is intentionally conservative:

- preserve exact value types through decode and prepare
- keep float values out of GP machine regs except at explicit semantic
  cross-bank operations
- add strong validation and assertions
- do not add new optimization heuristics yet

## Goals

- Preserve Wasm value types through decode, prepare, LIR, and LIR-to-MachineIR
  lowering.
- Ensure `f32`/`f64` values live only in FP transients or canonical frame
  slots.
- Ensure `i32`/`i64`/refs live only in GP transients or canonical frame slots.
- Remove the current representation-driven GP<->FP churn from ordinary float
  code.
- Keep the pipeline simple, explicit, and `no_std` friendly.

## Non-Goals

- No general register allocation.
- No speculative optimization or value numbering framework.
- No attempt to keep float values live across helpers, calls, or other existing
  canonical boundaries in the first phase.
- No benchmark-specific tuning.

## Core Rule

The semantic stack remains one ordered stack.

It must not become two independent semantic stacks.

WebAssembly is defined around one typed operand stack, and operations like
`drop`, `select`, mixed-arity calls, branches, and structured control all rely
on preserving stack order. The right model is:

- one ordered stack of entries
- each entry has a known Wasm type
- each entry is either resident in a GP register, resident in an FP register,
  or spilled to its canonical frame slot

## Current Problem

Today the prepare pipeline is fundamentally untyped and single-bank:

- one `Vec<LirValue>` live window
- one `spill_depth`
- one scalar transient width limit

See:

- [state.rs](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/plan/prepare/state.rs)
- [steps.rs](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/plan/prepare/steps.rs)
- [ir.rs](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/lir/ir.rs)

That model only works cleanly because resident values are treated as one
contiguous suffix of the stack. Once a float value is spilled and later
reloaded through an untyped `LoadSlot`, native lowering often recreates it in a
GP transient. The ARM64 backend then has to use scratch FP registers and
`fmov` shuffles to execute ordinary float operations.

## Proposed Model

### 1. Typed Stack Entries

Prepared execution state should track explicit typed stack entries rather than
only stack height and one untyped live suffix.

Conceptually:

```rust
struct StackEntry {
    ty: ValueType,
    residency: Residency,
}

enum Residency {
    Resident(LirValue),
    Spilled,
}
```

The stack index still determines the canonical operand slot:

- stack entry `i` spills to `frame.operand_slot(i as u16)`

Locals keep their existing canonical slots:

- local `i` lives in `frame.local_slot(i)`

### 2. Banked Residency

Each value type maps to exactly one transient bank:

- `I32`, `I64`, refs => GP bank
- `F32`, `F64` => FP bank

Each bank has its own budget:

- `gp_lanes`
- `fp_lanes`

Budget pressure is handled per bank, not by one combined lane count.

### 3. Ordered Stack, Separate Banks

Residency does not change stack order.

Example:

```text
[ i32(resident GP), f64(spilled), i64(resident GP), f32(resident FP) ]
```

This is legal and expected.

The stack remains ordered for semantics; bank selection only affects where each
entry is carried while live.

## Value-Type Invariants

The pipeline should enforce these invariants:

1. A `LirValue` has one exact Wasm value type for its whole lifetime.
2. A float `LirValue` may only be:
   - in an FP transient machine reg
   - in a canonical frame slot
3. A non-float `LirValue` may only be:
   - in a GP transient machine reg
   - in a canonical frame slot
4. Generic `LoadSlot` and `StoreSlot` behavior must respect the value type of
   the carried value.
5. Generic bank crossing is illegal.

The only legal GP<->FP crossings are explicit semantic crossings such as:

- int-to-float conversions
- float-to-int conversions
- bit reinterpret operations
- float compare producing `i32`
- explicit helper/runtime ABI crossings, if any remain

## Pipeline Changes

### 1. Decode Must Preserve Types

The current semantic decode keeps stack height but not stack types. That is not
enough for typed residency.

See:

- [decode.rs](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/wasm/decode.rs)
- [semantic_ir.rs](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/wasm/semantic_ir.rs)
- [context.rs](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/wasm/context.rs)

The frontend must preserve:

- full local types for the function, including params
- exact block param/result types
- exact call param/result types
- exact type of each pushed SSA value

This may be represented either by:

- enriching `SemanticProgram`, or
- attaching a typed sidecar consumed by prepare

The rule is more important than the container:

- prepare must consume exact value types
- it must not infer float-ness later from ad hoc opcode checks alone

### 2. Prepared LIR Must Carry Value Types

`LirValue` can remain an integer id, but prepared LIR needs a value-type side
table:

```rust
struct LirValueInfo {
    ty: ValueType,
}
```

`PreparedFunction` or `LirProgram` should own this side table.

Required consequences:

- `make_block_params` creates typed block params
- `Fill` recreates typed SSA values
- `LoadSlot` results are typed
- `StoreSlot` sources are type-checked

### 3. Prepare State Must Replace `spill_depth`

The current `spill_depth` model assumes one contiguous resident suffix.
Typed banked residency should replace that with explicit stack-entry state.

Instead of:

- `stack_height`
- `spill_depth`
- `live: Vec<LirValue>`

use:

- ordered stack entries
- per-entry type
- per-entry residency
- explicit resident counts by bank

Conceptually:

```rust
struct BlockState {
    gp_budget: u8,
    fp_budget: u8,
    stack: Vec<StackEntry>,
    gp_live: u8,
    fp_live: u8,
    ops: Vec<LirInst>,
}
```

### 4. Residency Policy

The first implementation should use a simple deterministic policy.

Recommended initial policy:

- on push:
  - try to keep the new value resident in its bank
  - if the bank is full, spill the deepest resident value of the same bank
- on operand use:
  - if a required operand is spilled, reload it into its bank
  - if the bank is full, spill the deepest resident value of the same bank
- on pop/consume:
  - free resident bank slots immediately

This is not meant to be globally optimal. It is meant to be correct, explicit,
and bank-aware.

### 5. Prefix Actions Must Become Typed

Current `PrepAction::Spill` and `PrepAction::Fill` operate only on contiguous
prefix counts.

See [ops.rs](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/plan/prepare/ops.rs).

The typed design should move away from count-only spill/fill and toward
explicit entry transitions. The exact API can vary, but it must be able to say:

- spill these specific stack entries
- reload these specific stack entries with their types

Count-only prefix fill/spill can remain as an internal optimization later if it
falls out naturally from the typed model, but it should not be the primary
state representation anymore.

### 6. Block Entry And Edge Contracts Must Be Typed

A block entry should describe which stack entries are resident and which are
canonical in slots.

Block params should exist only for resident live-ins. Spilled live-ins remain
implicit in frame slots.

This means entry-state metadata must preserve:

- stack order
- entry types
- which entries are resident
- which resident entries need block params

### 7. Native Lowering Must Respect Types Strictly

Once prepared LIR values are typed:

- float locals loaded from slots must use `alloc_float_value`
- float fills must create FP-bank values
- float results must never silently fall back to GP registers

See current fallback path in
[context.rs](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/lower/context.rs).

The new rule should be:

- if a float value is live, it is either in an FP transient or already spilled
- lowering must not rescue bank pressure by placing float SSA values in GP regs

### 8. MachineIR Invariants

MachineIR already has a GP/FP split:

- GP machine regs below `first_fp_reg`
- FP-only machine regs in `[first_fp_reg, reg_count)`

See:

- [module.rs](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/ir/machine/module.rs)
- [regfile.rs](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/lower/regfile.rs)

The redesign should strengthen the contract above MachineIR:

- ordinary float values must arrive in FP machine regs
- typed slot loads for floats should lower to FP machine regs
- GP float aliases should stop being a normal representation shape

### 9. ARM64 Backend Scratch Registers

This proposal does not forbid backend scratch FP registers such as `D0`, `D1`,
and `D2`.

It forbids logical value flow from being represented as GP values when the
value is semantically float.

So the distinction is:

- disallowed: float SSA value carried in a GP MachineIR reg for ordinary flow
- allowed: backend uses scratch FP regs to materialize a legal FP MachineIR op

Over time, ordinary float code should stop generating representation-driven
`fmov x<->d` traffic. Explicit conversions and unavoidable ISA-specific
materialization may still need targeted scratch moves.

## Conservative First Phase

The first implementation phase should not try to be clever.

It should do only this:

1. preserve exact types through decode and prepare
2. rebuild prepared state as typed ordered entries
3. separate GP and FP residency budgets
4. remove float fallback to GP in lowering
5. add strict validation and assertions
6. keep all existing canonical call/helper/boundary behavior conservative

This phase should be correctness-first, not benchmark-first.

## Migration Plan

### Phase 0: Documentation And Invariants

- land this design
- document the invariants in `NATIVE_DESIGN.md`

### Phase 1: Typed Frontend Plumbing

- add local-type and signature typing to the prepare input path
- add a prepared-LIR value-type side table
- make `Fill`, `LoadSlot`, block params, and result creation typed
- add tests for typed locals, typed fills, typed branches, and mixed stacks

### Phase 2: Typed Residency State

- replace `spill_depth` and untyped live suffix with explicit typed stack
  entries
- add separate GP and FP budgets
- implement the simple deterministic residency policy

### Phase 3: Strict Lowering

- forbid float fallback to GP regs in native lowering
- ensure float slot loads and fills always target FP machine regs
- add assertions that non-explicit GP<->FP crossings do not occur

### Phase 4: Measurement And Expansion

- re-dump LIR, MachineIR, and assembly
- verify that hot float code no longer reloads float values into GP regs by
  default
- only after correctness and codegen proof, consider increasing FP transient
  budget beyond the initial value

## Validation Plan

Required checks:

- unit tests for typed prepare state with mixed `i32`/`f64` stack shapes
- unit tests for typed `local.get`, `local.set`, `drop`, `select`, `br_if`,
  loop headers, and call signatures
- native lowering tests proving float slot loads become FP machine regs
- MachineIR validation rejecting illegal bank misuse
- ARM64 compile tests proving ordinary float flow does not fall back to GP regs
- full `spectest`
- full `benchmarks/wasi/run_tests.py`

Required dump checks before trusting performance numbers:

- LIR shows typed float fills/local loads feeding float ops
- MachineIR shows float values in FP machine regs
- disassembly shows removal of representation-driven `fmov x<->d` churn in hot
  float blocks

## Open Design Questions

These do not block the direction, but they must be resolved during
implementation:

1. Where should typed semantic metadata live?
   - directly in `SemanticProgram`
   - or in a typed sidecar owned by prepare
2. How much typed entry metadata should be stored on CFG edges?
3. Should the initial typed residency policy spill the deepest resident entry
   in the same bank, or the oldest by last-use order?
4. When should FP transient count increase beyond the current small bank?

The recommendation is to keep the answers conservative until the typed pipeline
is stable.

## Summary

The main change is not "add more FP registers".

The main change is:

- preserve types in the frontend
- replace the untyped rotating-TOS suffix with typed ordered residency
- make bank selection a property of value type

Once that exists, FP values can stay FP from prepared LIR through MachineIR,
and the ARM64 backend can stop paying the current representation tax for
ordinary float code.
