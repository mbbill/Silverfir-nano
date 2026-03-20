# Native MachineIR ABI Guide

This document defines the platform-agnostic native ABI contract used by
MachineIR lowering and native backends.

It is a guide for future native platform bring-up. It does **not** define how
any specific target maps physical registers. Per-target facts belong in that
target's `abi.rs`, which is the source of truth for the actual mapping.

## Scope

This document defines:

- the shared MachineIR register roles
- what each register class is allowed to assume at call boundaries
- how values are saved and restored across helper and local-call boundaries
- the conservative rules to use when a platform ABI is ambiguous

This document does **not** define:

- which physical register a target uses for any role
- how many registers a target budgets by default
- target-specific prologue or encoding details

## The Two Rules

### Rule 1: Only fixed MachineIR registers require free preservation

Only the fixed MachineIR registers are required to survive foreign-call
boundaries without compiler-inserted spills.

Those fixed roles are:

- `MACHINE_CTX_REG`
- `MACHINE_FP_REG`
- `MACHINE_MEM0_BASE_REG`
- `MACHINE_MEM0_SIZE_REG`

Therefore, each backend must map these fixed roles to physical registers that
are unquestionably preserved across the target's foreign ABI boundary.

Cached locals and transients do **not** require free preservation. They may be
saved and restored explicitly by lowering.

### Rule 2: If a register is not unquestionably callee-saved, treat it as caller-saved

If a platform ABI, OS ABI, toolchain convention, or platform register rule is
ambiguous, the backend must not rely on that register for fixed MachineIR
state.

Use the conservative policy:

- fixed MachineIR roles only in unquestionably preserved registers
- ambiguous or platform-sensitive registers treated as caller-saved
- caller-saved registers are still usable for cached locals or transients if
  lowering already spills them at the relevant boundaries

This rule is intentionally conservative. It keeps new platform bring-up safe
even when the target ABI has special platform registers.

## Shared MachineIR Register Roles

The first four MachineIR GP registers are fixed roles shared by all native
backends:

- `MachineReg(0)` = runtime context pointer
- `MachineReg(1)` = current frame base
- `MachineReg(2)` = cached `mem0` base pointer
- `MachineReg(3)` = cached `mem0` size

These are defined in [`regs.rs`](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/ir/machine/regs.rs).

All remaining machine registers are dynamically allocated by lowering.

The logical regfile layout is:

`[fixed | gp_local_cache | gp_transient | fp_transient | fp_local_cache]`

This layout is defined by [`regfile.rs`](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/lower/regfile.rs).

Implications:

- fixed registers are a shared contract, not a backend choice
- GP cached locals are allocated before GP transients
- FP bank starts at `first_fp_reg`
- any target-specific native lowering stage that rewrites dynamic GP regs must
  preserve this shared contract

## Meaning Of Each Register Class

### Fixed registers

These hold machine state that must remain live across helper calls and local
native calls:

- runtime context
- frame base
- cached `mem0` base
- cached `mem0` size

Fixed registers are part of the native ABI contract. A backend must not model
them as disposable temporaries.

### GP local-cache registers

These hold cached locals that also have canonical home frame slots.

Important property:

- a cached local is an optimization, not the canonical source of truth

Therefore cached locals may live in:

- callee-saved registers
- caller-saved registers
- any mixture of the two

as long as lowering publishes them back to their frame slots before a boundary
that may clobber them.

### GP transient registers

These hold temporary SSA values that are not allowed to remain live across call
boundaries.

They are expected to be dead before:

- local native calls
- helper calls
- other helper-backed runtime boundaries

Because of that, GP transients should preferentially use caller-saved physical
registers.

### FP transient and FP local-cache registers

The same logic applies to FP values:

- FP transients are dead at boundaries
- FP cached locals have canonical frame-slot homes
- FP cached locals may use caller-saved FP registers if lowering spills and
  reloads them appropriately

## Boundary Semantics

### Foreign helper / runtime boundaries

At a helper-backed boundary, lowering owns synchronization of non-fixed dynamic
state.

The shared model is:

1. publish all cached locals to their canonical frame slots
2. issue `CallHelper`
3. reload any fixed cached views that are modeled as explicit MachineIR state
4. reload cached locals from their frame slots

In the current code, this is implemented by:

- [`emit_save_all_cached_locals`](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/lower/context.rs)
- [`emit_reload_mem0_cache_regs`](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/lower/context.rs)
- [`emit_reload_cached_locals`](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/lower/context.rs)
- helper boundary lowering in [`boundary.rs`](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/lower/boundary.rs)

This means:

- helper-time cache synchronization is shared MachineIR behavior
- backends should not invent hidden cache-reload policy beyond what the
  MachineIR contract requires

### Local native calls

Transients must already be dead at local-call boundaries.

Cached locals are also publishable state, so lowering may spill them before the
call and let the callee freely reuse those dynamic registers.

The engine defines its own JIT-to-JIT ABI. Therefore:

- transients are not required to survive local calls
- cached locals are not required to survive local calls in registers
- fixed registers are the only always-live machine state across local calls

## Frame Slots Are Canonical State

Frame slots are the canonical storage for:

- locals
- spilled cached locals
- call arguments/results
- machine call-link metadata

Registers are an execution cache over that canonical frame state.

Consequences:

- cached locals may always be synchronized through frame slots
- a backend must not assume that keeping a local only in a register is
  sufficient at a call boundary
- local-cache placement is an optimization policy, not an ABI fact

## What `abi.rs` Must Define Per Platform

Each backend `abi.rs` must define:

- which physical registers back the 4 fixed MachineIR roles
- which physical GP/FP registers are allocatable for dynamic MachineIR regs
- the ordering of those dynamic registers
- which physical registers are scratch-only
- what the backend must save/restore in the shared prologue/epilogue
- stack-alignment and other target ABI facts needed by code generation

It must not change the shared meaning of the MachineIR register classes.

## Conservative Bring-Up Checklist

When bringing up a new native target:

1. Identify the target's unquestionably preserved registers at the foreign ABI
   boundary.
2. Place the 4 fixed MachineIR roles only in that unquestionably preserved set.
3. Prefer caller-saved registers for transients.
4. Place cached locals in whatever remaining registers are useful, including
   caller-saved ones if lowering already spills them.
5. If a register has platform-specific meaning or uncertain preservation, do
   not use it for fixed MachineIR state.
6. Keep any target-specific exception in the target's `abi.rs`, not in shared
   runtime structures.

## 32-Bit Note

For 32-bit targets, `emu32` and real 32-bit native backends are expected to
consume the same 32-bit-target MachineIR contract.

That means shared 32-bit lowering bugs should be exposed by `emu32`.
Backend-specific instruction-selection or encoding bugs may still be
target-only.
