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

### Rule 1: Only fixed MachineIR registers require mandatory call-boundary preservation

Only the fixed MachineIR registers are required to stay live across helper
calls and local native calls as part of the ABI contract.

Those fixed roles are:

- `MACHINE_CTX_REG`
- `MACHINE_FP_REG`
- `MACHINE_MEM0_BASE_REG`
- `MACHINE_MEM0_SIZE_REG`

Each backend must define how these four roles survive both boundary classes:

- **Foreign helper / runtime boundaries**: preferably map them to physical
  registers that are unquestionably preserved by the platform C ABI. If that
  is impossible on a target, the boundary wrapper must save and restore the
  fixed roles explicitly.
- **Local native calls**: the engine's JIT-to-JIT ABI must preserve them,
  again preferably by keeping them in preserved host registers and otherwise
  by an explicit save/restore sequence in the local-call boundary.

Cached locals and linear values do **not** require this mandatory
preservation. They may be published to frame slots and reloaded explicitly by
lowering.

### Rule 2: If a register is not unquestionably callee-saved at the foreign ABI boundary, treat it as caller-saved there

If a platform ABI, OS ABI, toolchain convention, or platform register rule is
ambiguous, the backend must not rely on that register for fixed MachineIR
state across foreign calls.

Use the conservative policy:

- fixed MachineIR roles only in unquestionably preserved registers, unless the
  foreign-boundary wrapper saves and restores them explicitly
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

Backends should preserve them in the cheapest robust way:

- preferred: choose host registers that already survive the relevant boundary
  (for example, host callee-saved registers across the C ABI)
- fallback: save and restore the fixed roles explicitly in the backend's
  helper-boundary wrapper or local-call sequence

Callers should observe the fixed roles as continuously live either way.

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

## Backend Lowering Discipline

The rules above define which register classes exist. This section defines how
backend lowering is allowed to use them.

These are backend-authoring invariants, not style suggestions.

### Ordinary instruction lowering may only touch owned registers

When lowering a normal MachineIR instruction or terminator, the backend may
only touch:

- the physical registers named by the MIR operands being lowered
- the 4 fixed MachineIR registers, when the instruction's semantics require
  them
- backend-owned temporary registers that were explicitly claimed from the
  relevant ownership tracker
- foreign ABI argument / return registers, but only while lowering the
  boundary itself

What a backend must **not** do during ordinary lowering:

- pick a convenient dynamic GP register as an extra temp
- pick a convenient dynamic FP register as an extra temp
- use `C_ARG*` / `C_RET*` as general-purpose temporaries away from the
  boundary
- touch backend-owned temp registers without first claiming them

If a lowering sequence needs one more temporary than its explicit operands
provide, it must claim one through the backend's ownership mechanism. For many
targets that is the shared scratch pool; for ISA-specific registers it may be a
target-local owner. "This register looks dead here" is not a valid reason to
bypass ownership tracking.

### Backend-owned temporary registers require explicit ownership

Backend-owned temporary registers exist specifically so backends have a place
to put temporary values while lowering. Some targets use the shared generic
scratch pool for this. Others may also need target-local ownership for
architecturally fixed registers that certain instruction forms require.

Whatever mechanism a backend uses, it must provide the same ownership
guarantees:

- reserve before use
- keep the reservation for the whole period the temp is live
- release when the temp is dead

The purpose of the ownership tracker is not just convenience. It prevents
unrelated lowering helpers from accidentally clobbering each other when they
are composed, nested, or refactored.

Backend code should therefore treat naked backend-owned-temp use as a
correctness bug, not as an optimization shortcut.

### Dynamic registers are not backend temporaries

The dynamic GP / FP banks belong to MachineIR lowering and register allocation.
They are not a backend-private scratch area.

Important consequences:

- the backend must not assume a dynamic register is free just because the
  current helper does not mention it
- the backend must not infer "this dynamic reg probably holds only a transient"
  from register number or allocation order
- the backend must not treat caller-saved dynamic regs as implicit scratch regs
- the backend must not treat callee-saved dynamic regs as implicitly preserved
  local-cache regs

Only lowering state knows whether a dynamic register currently holds:

- a transient SSA value
- a cached local
- a frame-backed reloadable value

Backend code does not get to guess.

### Call and helper boundaries do not justify hidden register preservation

At helper boundaries and local-call boundaries, shared lowering already defines
how dynamic state becomes safe:

- transients are dead before the boundary
- cached locals are published back to their canonical frame slots before the
  boundary
- fixed registers remain live across the boundary because the backend preserves
  them there

Therefore the backend must **not** paper over a bug by conservatively saving
and restoring the dynamic GP bank, the dynamic FP bank, or some guessed subset
of them around a helper or local call.

If such a save/restore seems necessary, the bug is almost certainly one of:

- ordinary lowering touched a register it did not own
- a backend-owned temporary register was used without explicit ownership
- boundary lowering failed to follow the shared publish / reload protocol
- the target's `abi.rs` mapped a fixed role to the wrong physical register or
  forgot to preserve it at the boundary

The fix is to repair ownership or boundary semantics at the root cause, not to
promote dynamic registers into hidden ABI-preserved state.

### Why these rules are strict

These constraints exist for three reasons:

- **Correctness**: only shared lowering knows which dynamic regs are live and
  what they mean. A backend cannot safely infer that from the physical
  register.
- **Composability**: lowering helpers need to remain safe when combined,
  reordered, or refactored. Hidden clobbers make that impossible.
- **Performance**: blanket save/restore of dynamic banks hides the real bug and
  adds hot-path overhead to every boundary.

The arm64 backend is the simplest reference implementation for this discipline:
it mostly confines itself to explicit operands plus common-pool temps, and it
asserts that scratch reservations are fully released between instructions. x64
adds target-local ownership for ISA-mandated GP registers, but the rule is the
same: no naked touching of non-operand temporaries. Other backends are expected
to follow the same model.

## Boundary Semantics

### Foreign helper / runtime boundaries

At a helper-backed boundary, lowering owns synchronization of non-fixed dynamic
state.

Independently, the backend must preserve the 4 fixed MachineIR roles across the
foreign boundary. The preferred strategy is to keep them in host
callee-saved/preserved registers; the fallback is an explicit save/restore in
the helper-boundary wrapper. Shared cache publication does **not** preserve
fixed regs for the backend.

The shared model is:

1. publish all cached locals to their canonical frame slots
2. issue `CallExternal`
3. refresh any fixed cached views whose semantic contents may have changed
   across the helper (for example `mem0` base/size after helper side effects)
4. reload cached locals from their frame slots

In the current code, this is implemented by:

- [`emit_save_all_cached_locals`](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/lower/context.rs)
- [`emit_reload_mem0_cache_regs`](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/lower/context.rs)
- [`emit_reload_cached_locals`](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/lower/context.rs)
- helper boundary lowering in [`boundary.rs`](/Users/bytedance/Dev/Silverfir-nano/sf-nano-core/src/vm/native/lower/boundary.rs)

This means:

- helper-time cache synchronization is shared MachineIR behavior
- refreshing `mem0` cache regs after a helper is a semantic cache update, not
  compensation for fixed-register clobbering
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

The local-call ABI must preserve those fixed registers. The preferred strategy
is again to keep them in host registers that the callee does not clobber. If a
target cannot do that, its local-call sequence must save and restore the fixed
roles explicitly. Either way, callers should observe the fixed roles as live on
return.

### Foreign C ABI argument and return registers

Per-target `abi.rs` may also define physical C ABI boundary registers such as
`C_ARG*` and `C_RET*`.

These are **not** extra MachineIR register classes. They are foreign-call ABI
facts used only while entering or leaving a helper or other foreign boundary.

Rules:

- `C_ARG*` / `C_RET*` may overlap caller-clobbered dynamic or backend-owned
  temp registers
- `C_ARG*` / `C_RET*` must not overlap the 4 fixed MachineIR roles
- non-boundary backend code must not treat `C_ARG*` / `C_RET*` as general
  allocatable registers

Why the overlap is safe:

- SSA values are required to be dead before the foreign boundary
- cached locals are published to frame slots before the foreign boundary
- fixed registers are preserved across the boundary by the backend's fixed-role
  strategy

So the foreign ABI temporarily reuses caller-saved physical registers only
after shared lowering has made dynamic MachineIR state unavailable there.

## Frame Slots Are Canonical State

Frame slots are the canonical storage for:

- locals
- spilled cached locals
- call arguments/results
- backend-private local-call record metadata

Registers are an execution cache over that canonical frame state.

Consequences:

- cached locals may always be synchronized through frame slots
- a backend must not assume that keeping a local only in a register is
  sufficient at a call boundary
- dynamic-register placement is an optimization policy, not an ABI fact

## What `abi.rs` Must Define Per Platform

Each backend `abi.rs` must define:

- which physical registers back the 4 fixed MachineIR roles
- how those 4 fixed roles are preserved across foreign helper boundaries and
  local native calls
- which physical GP/FP registers are allocatable for dynamic MachineIR regs
- the ordering of those dynamic registers
- which subset of the dynamic bank is caller-clobbered vs callee-saved, when
  that matters for helper wrappers or shared prologues
- which physical registers are backend-owned temporaries (generic scratch,
  ISA-mandated temps, or both)
- what the backend must save/restore in the shared prologue/epilogue
- stack-alignment and other target ABI facts needed by code generation

It must not reintroduce a static local-cache/linear-value split inside the dynamic
banks.

## Conservative Bring-Up Checklist

When bringing up a new native target:

1. Identify the target's unquestionably preserved registers at the foreign ABI
   boundary.
2. Prefer to place the 4 fixed MachineIR roles only in that unquestionably
   preserved set.
3. If that is impossible, implement explicit save/restore of the 4 fixed roles
   in the foreign-boundary wrapper and document the choice in `abi.rs`.
4. Define how local native calls preserve the 4 fixed roles as well; reuse the
   same physical mapping when possible.
5. Prefer caller-saved registers for linear values.
6. Place cached locals in whatever remaining registers are useful, including
   caller-saved ones if lowering already spills them.
7. If a register has platform-specific meaning or uncertain preservation, do
   not use it for fixed MachineIR state.
8. Keep any target-specific exception in the target's `abi.rs`, not in shared
   runtime structures.

## 32-Bit Note

For 32-bit targets, `emu32` and real 32-bit native backends are expected to
consume the same finalized legal 32-bit MachineIR contract.

That means shared 32-bit lowering, legalization, and finalization bugs should
be exposed by `emu32`. Backend-specific instruction-selection or encoding bugs
may still be target-only.

## Local Call ABI

Direct and indirect local calls between MachineIR functions use a
backend-private host-stack call record. The MIR-level contract is just
the `CallDirect` / `CallIndirect` terminator with `callee_frame_base`,
`caller_result_base`, and `continuation`; the call record layout itself
is target-private and never appears in the abstract MIR.

### Public entry vs internal entry

Each function gets two entry points in the emitted code:

- The **public entry** is the function start. It runs the C-ABI prologue
  (save callee-saved registers, move C arg regs into the fixed MachineIR
  roles), then the **public-entry caller stub** that builds a "root"
  call record on the host stack and `bl/call`s the internal entry. When
  the body returns, the stub falls through to the C-ABI epilogue and
  the platform `ret`.
- The **internal entry** is the body's true start, bound at
  `internal_entry_label` (allocated in `CompilerCore::new`). All local
  call sites and direct-call relocations resolve to this label, never
  to "the byte after the prologue".

This split keeps direct local calls free of the C-ABI prologue cost
while leaving the public entry usable from C code (root invocation,
external callbacks).

### Body prelude and terminal sequences

Today, native backends emit a backend-specific **body prelude** between
`internal_entry_label` and the first body block. This prelude is what makes
nested native calls safe in the body; leaf bodies may eventually skip it as an
optimization, but that gating is not part of the current ABI metadata yet.

| Backend  | Body prelude                                |
|----------|---------------------------------------------|
| arm64    | `stp x29, x30, [sp, #-16]!` — link save     |
| armv7a   | link save                                   |
| x86_64   | `sub rsp, 8` — alignment shim               |
| emulator | none                                        |

The current implementation always emits the prelude on native backends and
never emits one in the emulator. If we later add a per-function
`body_emits_native_call` bit, it should mean "this body actually needs the
native-call prelude"; until then, readers should treat the prelude as
unconditional on native backends.

The body has exactly two terminal sequences:

1. **Success Return** — emitted inline at every `Return` terminator.
   Pops the body prelude link save, pops the caller's call record,
   copies `MachineFunctionAbi::return_results` from the callee frame to
   `*caller_result_base`, restores `MACHINE_FP_REG`, sets `C_RET0 = 0`,
   and executes the platform `ret`.
2. **`body_local_error_label`** — bound by the pipeline tail, reached
   from every trap stub and from every post-call status check. A
   near-copy of the success path: same prelude pop and same call-record
   pop, but **no** result copy and **no** touch of `C_RET0` (which
   already holds the trap kind set by the trap stub or inherited from a
   trapped descendant's BL). Trap stubs end with `bl raise_trap; b
   body_local_error_label`. There is no shared `return_error_label`.

### Call sequence (arm64, reference implementation)

For `CallDirect` (with `caller_result_base` and `callee_frame_base`
already materialised in registers by earlier MIR ops):

```
stp caller_result_base, fp_reg, [sp, #-16]!   ; push call record
mov fp_reg, callee_fp                         ; switch frame pointer
ldr s0, =callee_literal                       ; deferred literal pool
blr s0                                        ; native call
cbnz w0, body_local_error_label               ; status check on C_RET0
                                              ; (continuation falls through
                                              ;  if it is the next emitted
                                              ;  block — see "Block layout"
                                              ;  below)
```

`CallIndirect` is identical except it uses the runtime register
`callee_entry` directly instead of a deferred literal.

The 16-byte record pushed before each call contains:

| offset | content              |
|-------:|----------------------|
|    0   | caller_result_base   |
|    8   | caller fp            |

The body prelude link save (a separate 16-byte `stp x29, x30`) sits
above this record. Both the success Return and `body_local_error_label`
pop them in the matching order.

### Trap propagation via C_RET0

Local-call trap status flows through `C_RET0`:

- **Success**: the body's Return path sets `C_RET0 = 0` before its
  native return.
- **Error**: trap stubs execute `bl raise_trap` (which records the
  WasmError on the runtime context and sets `C_RET0` to a non-zero
  trap kind), then `b body_local_error_label`. `body_local_error_label`
  preserves `C_RET0` across its prelude/record pop and through the
  native return, so the caller's post-BL `cbnz w0` sees the propagated
  error code and re-branches to its own `body_local_error_label`. The
  whole chain unwinds to the public-entry caller stub, whose epilogue
  hands the C_RET0 value back to the C caller as the function return
  value.

`CallExternal`'s post-helper `cbnz w0` targets the same
`body_local_error_label`, so external-helper failures use the same
unwind path.

### Block layout — continuation fall-through

The shared block layout pass in
`CompilerCore::extend_block_trace`/`block_layout` treats every
`CallDirect` / `CallIndirect` continuation as a preferred fall-through
target. When the layout pass succeeds in placing the continuation block
immediately after the call site, the backend's `lower_call_direct` /
`lower_call_indirect` elides the trailing `b continuation_label` (or
`jmp` / `b` on other architectures).

For backends that emit per-call inline literals (arm64 uses an
`ldr_lit_64` to load the patchable callee address), the literal must
not sit in the fall-through path between the call and the continuation
block, or it would be executed as garbage instructions. The arm64
backend handles this by **deferring** each call's literal to a
per-function literal pool flushed after edge stubs and before the body
tail labels — see `Arm64Backend::lower_function_literal_pool` and the
`lower_function_literal_pool` hook on `ArchBackend`. The default impl is
a no-op; backends without inline-literal call sequences (x86_64,
emulator) do not need it.

### Public-entry caller stub

The root call record built by `lower_root_caller_stub` uses
`caller_fp = caller_result_base = MACHINE_FP_REG` (the root frame
pointer). That makes the body's unified Return copy results into the
same bytes that `eval.rs::collect_native_results_from_stack` reads
afterwards, so root invocation needs no special-case wiring.

### Frame metadata

`MachineFunctionAbi` carries the static call-side data each function
publishes:

- `frame_prefix_slots` / `total_frame_slots` — the function's frame
  geometry produced by `FrameLayoutPlan`.
- `helper_scratch: Option<MachineFrameRegion>` — frame region for
  runtime helpers that take a frame-relative scratch base. This is the
  only frame slot region the call ABI now reserves; the older
  `call_scratch` (which used to hold a MIR-visible call link) is gone.
- `return_results: Option<MachineFrameRegion>` — region the callee
  writes its return values into; the unified Return copies from here
  to `*caller_result_base`.
- `init_locals` — non-param local slots that may be read before being
  written. The callee zero-initialises these at function entry; locals
  not listed here are written before any read.

### Host-stack-depth caveat

The local-call ABI consumes host stack proportional to native call
depth: each nested local call pushes a 16-byte backend-private call
record, and each native body also pays its backend-specific body
prelude cost (currently 16 bytes on arm64/armv7a, 8 bytes on x86_64,
0 in the emulator). For a wasm program that recurses W levels deep,
host stack consumption is therefore roughly "call-record cost per edge
+ body-prelude cost per active native body" along the active path.

`vm/runtime/context.rs` does **not** maintain a `native_call_depth`
guard today. The current assumption is that the wasm-side stack
overflow check (still emitted on indirect/SCC-internal calls and at
function entry) bounds nesting to whatever fits in the wasm stack. If
a future embedding ships with a small thread stack or runs untrusted
deeply recursive modules, the right fix is a separate host-stack-depth
counter on `NativeContext`, decremented at the body-entry prelude and
checked against a configurable cap.
