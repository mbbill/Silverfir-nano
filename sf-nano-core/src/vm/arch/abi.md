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

Cached locals and linear values do **not** require free preservation. They may be
saved and restored explicitly by lowering.

### Rule 2: If a register is not unquestionably callee-saved, treat it as caller-saved

If a platform ABI, OS ABI, toolchain convention, or platform register rule is
ambiguous, the backend must not rely on that register for fixed MachineIR
state.

Use the conservative policy:

- fixed MachineIR roles only in unquestionably preserved registers
- ambiguous or platform-sensitive registers treated as caller-saved
- caller-saved registers are still usable for cached locals or linear values if
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

`[fixed | gp_dynamic | fp_dynamic]`

This layout is defined by [`lower_regalloc.rs`](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/machine/lower_regalloc.rs).

Implications:

- fixed registers are a shared contract, not a backend choice
- the GP/FP bank split is real, but there is no static local-cache/linear-value split inside either bank
- semantic linear-value ownership is tracked by lowering state, not by machine-register number
- FP bank starts at `first_fp_reg`
- 32-bit legalization/finalization may rewrite GP regs, but must preserve this
  shared contract

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

### Dynamic GP / FP banks

Each backend exposes one ordered GP dynamic bank and one ordered FP dynamic
bank.

Important properties:

- the order is an allocation preference, not a semantic class boundary
- a dynamic register may hold either an SSA value or a cached local
- only the lowering state knows which dynamic regs currently hold SSA values
- cached locals still have canonical frame-slot homes

Therefore cached locals and SSA values may both use:

- callee-saved registers
- caller-saved registers
- any mixture of the two

as long as lowering publishes frame-backed state before a boundary that may
clobber it.

## Boundary Semantics

### Foreign helper / runtime boundaries

At a helper-backed boundary, lowering owns synchronization of non-fixed dynamic
state.

The shared model is:

1. publish all cached locals to their canonical frame slots
2. issue `CallExternal`
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

- SSA values are not required to survive local calls
- cached locals are not required to survive local calls in registers
- fixed registers are the only always-live machine state across local calls

### Foreign C ABI argument and return registers

Per-target `abi.rs` may also define physical C ABI boundary registers such as
`C_ARG*` and `C_RET*`.

These are **not** extra MachineIR register classes. They are foreign-call ABI
facts used only while entering or leaving a helper or other foreign boundary.

Rules:

- `C_ARG*` / `C_RET*` may overlap caller-clobbered dynamic or scratch registers
- `C_ARG*` / `C_RET*` must not overlap the 4 fixed MachineIR roles
- non-boundary backend code must not treat `C_ARG*` / `C_RET*` as general
  allocatable registers

Why the overlap is safe:

- SSA values are required to be dead before the foreign boundary
- cached locals are published to frame slots before the foreign boundary
- fixed registers remain live across the boundary

So the foreign ABI temporarily reuses caller-saved physical registers only
after shared lowering has made dynamic MachineIR state unavailable there.

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
- dynamic-register placement is an optimization policy, not an ABI fact

## What `abi.rs` Must Define Per Platform

Each backend `abi.rs` must define:

- which physical registers back the 4 fixed MachineIR roles
- which physical GP/FP registers are allocatable for dynamic MachineIR regs
- the ordering of those dynamic registers
- which subset of the dynamic bank is caller-clobbered vs callee-saved, when
  that matters for helper wrappers or shared prologues
- which physical registers are scratch-only
- what the backend must save/restore in the shared prologue/epilogue
- stack-alignment and other target ABI facts needed by code generation

It must not reintroduce a static local-cache/linear-value split inside the dynamic
banks.

## Conservative Bring-Up Checklist

When bringing up a new native target:

1. Identify the target's unquestionably preserved registers at the foreign ABI
   boundary.
2. Place the 4 fixed MachineIR roles only in that unquestionably preserved set.
3. Prefer caller-saved registers for linear values.
4. Place cached locals in whatever remaining registers are useful, including
   caller-saved ones if lowering already spills them.
5. If a register has platform-specific meaning or uncertain preservation, do
   not use it for fixed MachineIR state.
6. Keep any target-specific exception in the target's `abi.rs`, not in shared
   runtime structures.

## 32-Bit Note

For 32-bit targets, `emu32` and real 32-bit native backends are expected to
consume the same finalized legal 32-bit MachineIR contract.

That means shared 32-bit lowering, legalization, and finalization bugs should
be exposed by `emu32`. Backend-specific instruction-selection or encoding bugs
may still be target-only.
