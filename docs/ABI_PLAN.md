# Local Call ABI Redesign — Plan

Status: design locked (pending implementation).
Audience: anyone touching `vm/machine/lower_call.rs`, `vm/machine/lower_module.rs`,
`vm/machine/machine_ir/{abi,cfg,inst}.rs`, the per-arch `control.rs` files,
`vm/arch/common/{pipeline,eval}.rs`, or `vm/runtime/{code,layout}.rs`.

## 1. Why

The current local-call ABI uses a software-managed return mechanism: every
call site stores a continuation pointer, the caller frame pointer, and a
result-base offset into a frame "call link" record, plus a per-call wasm-stack
overflow check, plus an unconditional spill of every dirty cached local. The
return path mirrors that record. On `arm64` this costs roughly 12 fixed ops
per call site and 6 + 2N ops per return (N = result count), where LLVM uses
~3 ops per call and ~2 per return.

Measured on coremark today: roughly 70 local call sites in the hot
functions (`func 6`/`func 8`/`func 9`) are paying this tax. Categorising
`func 6`'s 1006 instructions against LLVM's 433 shows the gap is dominated by
**memory traffic and call boilerplate**, not by missed peepholes:

| category        | SF  | LLVM | Δ    |
|-----------------|-----|------|------|
| memory loads    | 199 | 54   | +145 |
| memory stores   | 165 | 26   | +139 |
| reg→reg moves   | 160 | 64   |  +96 |
| uncond branches |  87 | 23   |  +64 |

Per call site, the round-trip waste is ~13 ops; over the hot functions that
totals roughly 700 static instructions and a much larger dynamic count (every
coremark iteration drives many of these calls). The two earlier patches in
this thread (MOVN encoding, Eqz→IntCompare fusion) saved ~380 static
instructions but barely moved coremark, because the call boilerplate was
absorbing the wins. This redesign is the structural fix.

## 2. Goals

1. Per local-call site overhead ≤ 5 native ops on arm64 (from ~12),
   including the in-band post-BL status check (a single `cbnz w0`).
   Comparable reduction on x86_64 (≤ 7 ops including the `test eax/jnz`).
2. Per return overhead ≤ 4 + N ops (from ~6 + 2N), including a single
   `mov w0, #0` (`xor eax, eax` on x86_64) for the success status.
3. MachineIR stays portable. No arch-specific variants of `CallDirect` /
   `CallIndirect` / `Return`. Per-platform details live in the backend's
   terminator lowering.
4. Cached locals are spilled at a call site only if the callee can actually
   clobber the registers they live in (per-callee clobber set).
5. The wasm-stack overflow check is amortised to one check per function entry
   for non-recursive direct-call DAGs. SCC members and indirect-call sites
   still pay per-call.
6. Lowerable on `arm64`, `x86_64`, `armv7a`, and the emulator with no
   correctness gaps in any one backend before the others land.
7. Trap propagation works correctly from locally-entered callees (i.e., the
   shared C-ABI error tail does not get reused by callees that never ran the
   public prologue — see §9).
8. Status propagation rides the existing C ABI return register
   (`C_RET0`). No new `NativeContext` field, no host-Rust-layout coupling
   in generated code.

## 3. Non-goals (v1)

- No result aliasing (callee return-results region overlapping caller's
  result slots). Keep the explicit copy.
- No new fixed result-area register. `caller_result_base` is a normal MIR reg.
- No special-casing of call lowering per architecture inside MachineIR.
- No `CallExternal` ABI changes. Helper boundary stays as it is today.

## 4. The shape change at a glance

### MachineIR

```rust
// vm/machine/machine_ir/cfg.rs

pub(crate) enum MachineTerminator {
    // ...
    CallDirect {
        callee:             MachineFuncId,
        callee_frame_base:  MachineReg, // caller_fp + args_offset
        caller_result_base: MachineReg, // caller_fp + results_offset (absolute)
        continuation:       MachineBlockId, // CFG edge — abstract
    },
    CallIndirect {
        callee_target:      MachineReg,
        callee_entry:       MachineReg,
        callee_frame_base:  MachineReg,
        caller_result_base: MachineReg,
        continuation:       MachineBlockId,
    },
    Return,
    // ...
}
```

**Deletions**

- `CallDirect::call_link_base`, `CallIndirect::call_link_base`
- `MachineCallLinkLayout` (and its field on `MachineModuleAbi`)
- `FrameLayoutPlan::call_scratch` is **split** (see §6 — *not* deleted by
  name, the live half is renamed)

**Additions**

- `caller_result_base: MachineReg` on both `CallDirect` and `CallIndirect`
- `MachineFunctionAbi::body_emits_native_call: bool` (per-function metadata; the
  bit asks a property question about the *function*, not the *backend* —
  see §8 for backend interpretation)

### What `caller_result_base` means

Caller computes it once: `caller_result_base = caller_fp + results_offset_imm`.
The address is absolute. The callee's `Return` writes results to
`*caller_result_base`. The caller does not read this register again after
emitting the call terminator.

### What `continuation` means

It is the abstract CFG successor of the call site. The emulator and any
future non-native backend uses it directly to drive control flow. Native
backends use the platform return mechanism (LR or stack) and let the
continuation block fall through; the native return mechanism ignores the
`continuation` field at runtime, but the field stays in the MIR for layout,
debug, and validation.

## 5. Public entry vs internal entry

This is the part that bit my first draft and needs an explicit story.

### Today

`vm/arch/common/pipeline.rs:38–40` records:

```rust
let prologue_start = b.core().text.len();
b.lower_prologue();
let internal_entry_offset = b.core().text.len();
```

i.e. `internal_entry_offset` is *defined as* "the byte right after
`lower_prologue()` returns". Direct call patches in
`pipeline.rs:299–305` resolve to `internal_entry_addrs[i] = base + internal_entry_offset`.
Everything works because the prologue is the *only* thing between the public
entry and the body — falling through from the prologue is the same as
entering the body.

### After 1A

The public entry runs the **C-ABI prologue + a real caller stub + the
C-ABI epilogue**. The internal entry is no longer "just after the prologue";
it is a *separate* labelled location somewhere later in the function text.

Layout:

```
public_entry_label:               ; <-- C ABI lands here
    lower_prologue                 ; save C callee-saved (x19–x30, d8–d15, ...)
    set up MACHINE_CTX_REG, MACHINE_FP_REG, ...
    ; --- root caller stub ---
    compute caller_result_base = stack_base
    push backend-private call record (caller_fp = stack_base,
                                      caller_result_base = stack_base)
    bl/call internal_entry_label
    ; --- callee returns here ---
    ; C_RET0 already holds 0 (success) or trap kind (error) — see §9
    lower_epilogue                 ; restore C callee-saved
    ret

internal_entry_label:             ; <-- direct local calls patch to this
    [synthetic body-prelude block]
        per-backend non-leaf setup gated on body_emits_native_call (§8):
          arm64/armv7a: backend-private link save
          x86_64: alignment shim (sub rsp, 8)
        (1D will add the hoisted stack check here)
    block 0
    block 1
    ...
    return_ok_label:              ; unified Return; sets C_RET0 = 0
    body_local_error_label:       ; trap propagation tail (§9); leaves C_RET0 alone
```

### Pipeline changes

`vm/arch/common/pipeline.rs` must:

1. Allocate an explicit `internal_entry_label` at backend creation time.
2. After `lower_prologue()`, emit the **caller stub** (a small per-backend
   helper, e.g. `lower_root_caller_stub()`) followed directly by
   `lower_epilogue()` and `ret`. The stub does the `bl/call internal_entry`
   and then falls into the existing epilogue; no status check is needed
   in the stub because `C_RET0` already holds the right value (see §9).
   Do *not* record `internal_entry_offset` from the current text position.
3. Bind `internal_entry_label` *after* the caller stub + epilogue, before
   the body prelude. Record its offset for direct call patching.
4. Emit the body-prelude block (per-backend non-leaf setup in 1A; stack
   check in 1D).
5. Lower body blocks as today.
6. **Replace** the function-wide `return_error_label` with
   `body_local_error_label`: a synthetic block reached from trap stubs,
   `CallExternal`'s post-helper status check, and every caller-side
   post-BL status check. It lowers as a near-copy of the unified Return
   sequence with two differences: it skips the result copy and it does
   not touch `C_RET0`. The trap stub for each kind does
   `bl raise_trap; mov C_RET0, #<trap_kind>; b body_local_error_label`.
   The deferred per-trap-kind tail blocks all funnel through
   `body_local_error_label` — there is no surviving function-wide
   `return_error_label` after 1A.

### What goes away

- `seed_root_call_link()` in `eval.rs`
- `native_root_return()` in `eval.rs` and `runtime/code.rs`
- `NativeCode::root_return: Option<NativeCodePtr>` field
- `NativeCode::set_native_entry(..., root_return: ...)` parameter
- `EmittedFunction::root_return_offset`
- `pipeline.rs:126–138` root return offset computation
- The `b14` / call-link-based root return continuation block synthesis
  (whatever currently emits the "after this root call returns, set status,
  jump to epilogue" handshake)

### Emulator note

The emulator does *not* need to literally simulate the public-entry caller
stub. It can keep its existing root special case and only emulate the new
local-call ABI shape (push logical call frame on `CallDirect`, pop on
`Return`).

## 6. Splitting `call_scratch`

The `call_scratch` field on `FrameLayoutPlan` and on `MachineFunctionAbi` is
doing two unrelated jobs:

- **Dead job**: storing the local-call call link (continuation, caller fp,
  result base). These are the slots indexed by `MachineCallLinkLayout::*_offset`.
- **Live job**: helper-call scratch and frame-prefix padding plumbing. This
  is what `lower_module.rs:148–155` is computing as `helper_scratch` —
  whatever is left after the call-link prefix.

If 1A naively removes `call_scratch`, the live half disappears too.

**Plan**:

1. Rename the live half. `FrameLayoutPlan::call_scratch` →
   `FrameLayoutPlan::helper_scratch`. The frame planner allocates only the
   helper-scratch slots; it no longer reserves room for a call link.
2. `MachineFunctionAbi::call_scratch` is **deleted**. `helper_scratch` stays.
3. `lower_module.rs:144–183` (`lower_function_runtime`) drops the
   `call_link.slot_count` arithmetic and just exposes `helper_scratch`
   directly from `input.frame.helper_scratch`.
4. `vm/runtime/dispatch_view.rs::NativeLocalCallInfo{32,64}` loses the
   `call_scratch_base_slot` field. `entry`, `total_frame_bytes`, and
   `frame_prefix_slots` stay — indirect local calls still need them.
5. `vm/runtime/layout.rs::LocalCallInfoAbiLayout` loses the
   `call_scratch_base_slot_offset` field (the offset is *not* on
   `CallDispatchView`; that struct only carries `kind`, `type_canon`,
   `local_target`). The corresponding assertions in the
   `runtime::layout::tests` module are removed in lockstep.
6. `lower_module.rs:1340–1390` (the indirect-call call-link write path) is
   rewritten to compute `caller_result_base` directly instead of
   `call_link_base`, matching the new MIR shape.

After this, every place that referenced `call_scratch_base_slot` either
stops mentioning it or moves to the new `caller_result_base` plumbing.

## 7. Per-backend lowering contracts

The MIR is the same on every backend. The arch-specific work is in
`vm/arch/<arch>/control.rs::lower_call_direct`,
`lower_call_indirect`, and `lower_return_sequence`, plus a new
`lower_root_caller_stub` in the pipeline-side helper file.

The shared description: at `CallDirect` lowering time the backend must (a)
arrange for the platform's hardware return mechanism to bring control back
to the continuation block, (b) make the caller's `MACHINE_FP_REG` value and
`caller_result_base` recoverable inside the callee's `Return`, and (c)
switch `MACHINE_FP_REG` to `callee_frame_base` before transferring control.

Backends are free to choose any aligned, backend-private layout for the
call record. The MachineIR contract does not name `x29`, `x30`, the host
SP, or any specific instruction.

### arm64

Caller (per `CallDirect`, including the register-based post-BL status
check from §9):

```
add  result_base_reg, fp_reg, #results_offset_imm   ; 1
stp  result_base_reg, fp_reg, [sp, #-16]!           ; 2  call record
add  fp_reg, fp_reg, #args_offset_imm                ; 3  advance in place
bl   callee_internal_entry                            ; 4
cbnz w0, body_local_error_label                       ; 5  status check
; --- continuation block placed here as fallthrough if layout permits ---
; (otherwise: b continuation_label)
```

= 5 ops + (optional) trailing branch. The cbnz is a single instruction and
predicts not-taken on every successful run.

Internal-entry body prelude (when `body_emits_native_call`):

```
stp  x29, x30, [sp, #-16]!
```

(Backend-private; the abstract ABI just says "save link state".)

`Return` lowering (success path → `return_ok_label`):

```
if body_emits_native_call:
    ldp  x29, x30, [sp], #16
ldp  result_base_scratch, caller_fp_scratch, [sp], #16
; copy return_results region into [result_base_scratch]:
str  res0, [result_base_scratch, #0]
str  res1, [result_base_scratch, #8]
...
mov  fp_reg, caller_fp_scratch
mov  w0, #0                                  ; success status (§9)
ret
```

`body_local_error_label` lowering (trap propagation path):

```
if body_emits_native_call:
    ldp  x29, x30, [sp], #16
ldp  result_base_scratch, caller_fp_scratch, [sp], #16
; (no result copy — results are undefined on trap)
mov  fp_reg, caller_fp_scratch
; w0 is already non-zero (set by trap stub or inherited from a trapped
; descendant's BL via the cbnz fallthrough at the call site)
ret
```

The body-prelude `stp` and both terminal sequences' `ldp` are paired. They
run on *both* the public-entry path (the public stub does
`bl internal_entry`, which leaves LR pointing into the stub) and the
local-entry path (a caller's `bl` left LR pointing into the caller). In
both cases the internal entry's prelude saves LR before any nested call,
and the terminal sequence restores it before its `ret`.

### x86_64

Two facts to keep straight:

1. **`call` pushes the return address on top of the caller's pushes.**
   Naive `pop / pop / ret` is wrong: the popped values would be the return
   address and one of the caller's pushes, not both pushes.
2. **The body of a function expects `rsp` to be 16-byte aligned** (SysV ABI:
   `rsp ≡ 0 (mod 16)` at the point any helper `call` is issued). The current
   `lower_prologue()` establishes that alignment by pushing N callee-saved
   regs and then `sub rsp, STACK_PADDING`. A locally-entered function
   bypasses the prologue entirely, so the body inherits whatever `rsp`
   alignment the caller stub left behind — and the caller stub's three
   pushes (`result_base`, `caller_fp`, return-address-from-call) leave `rsp`
   at `≡ 8 (mod 16)`, *not* the post-prologue aligned state.

Both problems are solved by giving x86_64 an **internal-entry alignment
shim** in the body prelude (§8) and using **`ret 24`** at `Return` time:

- Body prelude on x86_64 always emits `sub rsp, 8` when `body_emits_native_call`
  is set. This lands the body at `rsp ≡ 0 (mod 16)`, matching what the
  C-ABI prologue would have established.
- `Return` does `ret 24` — pops the return address (8) and additionally
  drops `8 + 16 = 24` bytes (alignment slot + 16-byte call record). One
  instruction discards everything.

Pure leaves on x86_64 (no `CallDirect` / `CallIndirect` / `CallExternal`)
don't need the shim: there's no helper or trap call inside the body that
would observe the misalignment. They use `ret 16` (no alignment slot to
discard) and skip the prelude `sub`.

The stack at the start of the callee body looks like:

```
[rsp + 0]   = return address (pushed by call)
[rsp + 8]   = caller_fp        (pushed by caller before call)
[rsp + 16]  = result_base      (pushed by caller first)
```

After the body-prelude `sub rsp, 8` (when `body_emits_native_call`):

```
[rsp + 0]   = unused alignment slot
[rsp + 8]   = return address
[rsp + 16]  = caller_fp
[rsp + 24]  = result_base
```

Caller (per `CallDirect`, including the register-based post-BL status
check from §9):

```
lea   result_base_reg, [fp_reg + results_offset]
push  result_base_reg
push  caller_fp_reg
lea   fp_reg, [fp_reg + args_offset]   ; advance fp_reg in place
call  callee_internal_entry
test  eax, eax                          ; status check
jnz   body_local_error_label
; --- continuation falls through ---
```

`Return` (success path, `body_emits_native_call` true):

```
mov   caller_fp_scratch,   [rsp + 16]
mov   result_base_scratch, [rsp + 24]
; copy return_results region to [result_base_scratch]
mov   fp_reg, caller_fp_scratch
xor   eax, eax                        ; success status (§9)
ret   24                              ; pops ret-addr + alignment + 16-byte record
```

`Return` (success path, leaf — no shim):

```
mov   caller_fp_scratch,   [rsp + 8]
mov   result_base_scratch, [rsp + 16]
; copy return_results region to [result_base_scratch]
mov   fp_reg, caller_fp_scratch
xor   eax, eax
ret   16
```

`body_local_error_label` (`body_emits_native_call` true, alignment shim):

```
mov   caller_fp_scratch,   [rsp + 16]
mov   result_base_scratch, [rsp + 24]
; (no result copy)
mov   fp_reg, caller_fp_scratch
; eax is already non-zero (set by trap stub or inherited)
ret   24
```

`body_local_error_label` (leaf form):

```
mov   caller_fp_scratch,   [rsp + 8]
mov   result_base_scratch, [rsp + 16]
mov   fp_reg, caller_fp_scratch
ret   16
```

x86_64 does not need a *register-based* link save (`call`/`ret` use the
host stack natively for the return address), but it *does* need the
alignment shim and the matching `ret 24`. The same `body_emits_native_call`
bit gates both — see §8.

### armv7a

Same shape as arm64. `bl` writes `lr` (r14), nested `bl`s clobber it,
non-leaf functions need to save it at the body prelude. `Return` uses
`bx lr` after restoring.

Backend-private layout decisions (e.g. whether to use `push {r4, lr}` or
two `str`s) live in `vm/arch/armv7a/control.rs`; they are not part of the
MachineIR contract.

### emulator

The emulator backend ignores all of the hardware-return discussion and
maintains a logical call stack. On `CallDirect` it pushes
`(caller_fp, caller_result_base, continuation_block_id)` onto its own
`Vec<...>`; on `Return` it pops one frame, copies results, restores
`fp_reg`, and resumes execution at the popped continuation block. The
public-entry path can keep its current "this is the root call" special
case — it does not need to construct the same shape as native backends.

## 8. `body_emits_native_call` and the body prelude

The shared per-function metadata field is named `body_emits_native_call`,
*not* `needs_link_save`. The reason: the bit asks a property question about
the function ("does this function's MIR contain anything the backend will
lower as a host bl/call?"), which is computable from shared lowering inputs
alone. What each backend *does* with the bit is a backend interpretation,
not part of the shared MIR contract. Storing the property question in
shared metadata, and letting each backend translate it to its own action,
keeps the MachineIR contract clean and prevents backend semantics from
leaking into shared lowering.

Definition (shared):

> `body_emits_native_call` is true iff the function's MIR contains any of
> the following ops, all of which lower as a native `bl`/`call` on at least
> one supported backend:
>
> - `CallDirect` (terminator) — local SF→SF call
> - `CallIndirect` (terminator) — local SF→SF call via table
> - `CallExternal` (instruction) — runtime helper boundary
> - `Trap` (terminator) — `lower_trap_dispatch` calls `raise_trap` via `bl`
> - `TrapIf` (instruction) — conditional trap; the not-taken side falls
>   through but the taken side branches to a trap stub that calls
>   `raise_trap` via `bl`
> - any future MIR op whose backend lowering emits a host call must be
>   added to this list

`CallExternal` counts because the runtime helper boundary on arm64/armv7a
goes through `bl <runtime helper>` (clobbering the link register), and on
x86_64 goes through `call <runtime helper>` (requiring 16-byte alignment).
`Trap`/`TrapIf` count for the same reason — they reach `raise_trap` via the
same host-call mechanism, and a function whose only "call" is a trap path
still needs the prelude in order to unwind correctly.

Computed during machine lowering by scanning each function's MIR. Stored as
`MachineFunctionAbi::body_emits_native_call: bool`. The shared lowering
walks the MIR ops once and ORs the bits together; the list above lives in
one place (call it `mir_op_emits_host_call(&MachineInstKind) -> bool`) so
that adding a new MIR op that lowers as a host call only requires updating
one predicate, not chasing every per-function metadata computation.

### Per-backend interpretation

| backend | `body_emits_native_call = true` | `body_emits_native_call = false` (leaf) |
|---|---|---|
| arm64   | body prelude `stp x29, x30, [sp, #-16]!`; both terminal sequences `ldp` before `ret` | nothing in prelude; terminal sequences are just `ret` |
| armv7a  | backend-private link save in prelude; restore before `bx lr` | nothing in prelude; terminal sequences are `bx lr` |
| x86_64  | body prelude `sub rsp, 8` (alignment shim); terminal sequences use `ret 24` | no shim; terminal sequences use `ret 16` |
| emulator| nothing (logical call stack handles everything) | nothing |

Each backend reads the shared bit and decides what its body prelude /
terminal sequence emission needs to do. None of these decisions are visible
in the MIR. The success terminal sequence (`return_ok_label`) and the trap
propagation tail (`body_local_error_label`) share the same prelude pairing
— both apply the same backend-specific cleanup before the native return.

Note that the **status convention** (success → `C_RET0 = 0`, error → leave
`C_RET0` non-zero) is uniform across all four backends and is *not*
dispatched on `body_emits_native_call`. It is part of the shared
local-call ABI contract; see §9.

### The synthetic body-prelude block

The body prelude is a *synthetic* MIR-level block, not user wasm code. The
common pipeline emits it after binding `internal_entry_label` and before
lowering the first user block. This keeps the prelude out of the user CFG
and out of the block-layout pass — it is always physically first inside the
internal entry, on every entry path (public-stub `bl internal_entry` and
local-call `bl callee_internal_entry`).

In 1A the prelude block carries the per-backend non-leaf setup (link save
on arm64/armv7a, alignment shim on x86_64). In 1D it also carries the
hoisted stack check. The block contents are decided per-backend based on
`body_emits_native_call`; the shared lowering only emits an empty prelude block
slot that the backend's own emission code populates.

## 9. Trap propagation and the shared error tail

This is the part that is *not* "the existing tail can stay roughly as is" —
it has to change.

### The problem

Today's tail layout (`vm/arch/common/pipeline.rs:92–112`) emits a single
function-wide `return_error_label` whose body is just `lower_epilogue()`
followed by `ret`. Trap helpers (`lower_trap_dispatch`) and the
`CallExternal` post-call status check (`lower_cbnz` to `return_error_label`)
all branch there. This works because today every function is entered via
the C-ABI prologue, so by the time anything reaches `return_error_label`
the C callee-saved registers are sitting on the host stack ready to be
restored.

Under the new ABI, local SF→SF calls land at `internal_entry_label`,
*skipping* the prologue. If a locally-entered callee then trapped and
branched to `return_error_label`, the C-ABI epilogue would pop garbage off
the host stack (or worse, pop the call record left behind by the local
caller) and `ret` through whatever LR was last set, almost certainly
crashing or returning to the wrong place. Reusing the public-entry error
tail for locally-entered callees is a correctness bug.

### The design — register-based status, two-label split

Trap propagation works the same way function returns work: every level
unwinds via the unified `Return` mechanism (§7), and propagation between
levels uses an **in-band status return register**. There is no new
`NativeContext` field, no host-Rust-layout coupling, and no extra memory
load per call site.

**Status register**: the existing C ABI return register `C_RET0` (i.e. `w0`
on arm64, `r0` on armv7a, `eax` on x86_64). It is caller-saved, already
clobbered across any call, and already happens to be the value the public
entry must return to its C caller. We hijack the same register for
SF↔SF status return.

**Convention**:

> After any local `bl`/`call`, `C_RET0 == 0` means the callee returned
> normally. `C_RET0 != 0` means the callee (or one of its descendants)
> trapped, and `NativeContext.error` holds the diagnostic `WasmError`
> (the helper-based `raise_trap` path still populates the Rust-side
> `Option<WasmError>` exactly as today; the register only carries
> "something went wrong, propagate").

**Inside the body**:

- The trap stubs (`lower_trap_dispatch` family) still call `raise_trap` via
  the runtime helper to populate `NativeContext.error`. *After* that
  helper returns, the stub explicitly sets `C_RET0` to a non-zero value
  (concretely, the `MachineTrapKind` discriminant; any non-zero will do
  for propagation but the kind value gives a cheap debug breadcrumb), then
  branches to `body_local_error_label`.
- `CallExternal`'s existing post-helper status check (`cbnz x0, ...` on
  the helper return value) now branches to `body_local_error_label`
  instead of `return_error_label`. The helper's return value is already
  the propagation status, so no extra `mov` is needed.

`body_local_error_label` lowers as: pop the call record (if
`body_emits_native_call`, also restore the body-prelude link save /
alignment shim) → restore `fp_reg` → native return. **Crucially, it does
not touch `C_RET0`.** The non-zero status set by the trap stub (or
inherited from a trapped callee's BL) flows through unchanged. v1 also
skips the result-region copy on this path (results are undefined on
trap; the caller filters them via the status check anyway).

The success-path `Return` lowering is identical *except* it sets
`C_RET0 = 0` immediately before the native return. That's the only
difference between the two terminal sequences.

**At every caller-side `CallDirect` / `CallIndirect`**, after the BL/CALL:

```
arm64:    cbnz w0, body_local_error_label
armv7a:   cmp r0, #0; bne body_local_error_label
x86_64:   test eax, eax; jnz body_local_error_label
```

One op on arm64 (cbnz), two on x86_64 / armv7a. The branch is overwhelmingly
not-taken on a successful run, so the dynamic cost is essentially the
single fall-through prediction.

**At the public entry**, no separate post-BL status check is needed.
`C_RET0` already holds the value the C ABI wants — `0` on success, the
trap kind on error — and the C-ABI epilogue preserves `C_RET0` across the
register restores by construction (it is the return value register on
every supported platform). The public stub is therefore just:

```
public_entry_label:
    [lower_prologue]                 ; save C callee-saved
    [build root call record]
    [bl/call internal_entry_label]
    ; C_RET0 already holds 0/error from the body's terminal sequence
    [lower_epilogue]                 ; restore C callee-saved
    ret
```

There is no `public_return_ok_label` / `public_return_error_label` split.
Both paths fall through the same epilogue. The host C side reads
`C_RET0` as the public entry's return value and reads
`NativeContext.error` if the return value is non-zero. (That is exactly
the contract `eval.rs` already has today, just driven by `C_RET0` instead
of by reading a flag the runtime sets manually.)

### Labels per function (after 1A)

| label | reachable from | runs | terminates with |
|---|---|---|---|
| `public_entry_label` | C ABI | C-ABI prologue + caller stub | `bl internal_entry`, then `lower_epilogue` (no status check) |
| `internal_entry_label` | local `bl/call`, public stub `bl/call` | body prelude + body blocks | various; success exits via `Return` terminator → `return_ok_label` |
| `return_ok_label` | every `Return` terminator | unified Return: pop record, copy results, restore `fp_reg`, `mov C_RET0, #0` | native return |
| `body_local_error_label` | trap stubs, post-BL status check, `CallExternal` post-call check | unified Return without the result copy and without touching `C_RET0` | native return |

The deferred per-trap-kind tail blocks (the `lower_trap_dispatch` codegen
that materializes a trap kind constant and calls `raise_trap`) stay in the
function tail. Each one's final sequence becomes
`bl raise_trap; mov C_RET0, #<trap_kind>; b body_local_error_label` instead
of `bl raise_trap; b return_error_label`. There is no longer a function-wide
`return_error_label` at all — every error path goes through
`body_local_error_label`, and `body_local_error_label` is the same shape
as a normal `Return` (just three small differences: skip result copy,
don't touch `C_RET0`, and reach it via a different control predecessor).

### Cost note

The dynamic cost of the propagation check is one cbnz/test+jnz per call
site, predicted not-taken on successful runs. The branch predictor on
modern arm64 / x86_64 cores handles this for free. We do not pay any extra
load. Across the hot functions this is well under 1% of cycles even in the
absolute worst case, and the design is implementable across all four
backends without depending on host Rust struct layout.

A future revision could skip the check at call sites whose callee is
statically known to never trap (per-function `can_trap` bit, computed
bottom-up over the call graph). Out of scope for v1.

## 10. Block layout: continuation as a preferred fall-through

The block layout pass already orders blocks to maximise fall-through for
conditional branches. Extend it to treat the `continuation` edge of
`CallDirect`/`CallIndirect` as a preferred (not forced) fall-through.

Backend lowering of the call terminator must handle both cases:

- continuation is the next physical block: emit nothing after the `bl/call`.
- continuation is elsewhere: emit one `b/jmp continuation_label` after the
  `bl/call`.

Adjacency is an optimisation, never a correctness condition.

## 11. Host-stack-depth caveat

This ABI makes host-stack consumption proportional to native call depth:
each nested local call pushes a 16-byte call record (and on arm64/armv7a a
non-leaf body also pushes 16 bytes for the link save). For a wasm module
that recurses W levels deep, the host stack consumption is roughly
`16 * W + 16 * (non_leaves on the path)` bytes.

`vm/runtime/context.rs` does **not** maintain a `native_call_depth` guard
today. v1 accepts this on the assumption that the wasm-side stack overflow
check (which still runs on indirect/SCC-internal calls and at function
entry once 1D lands) bounds nesting to whatever fits in the wasm stack.

If a future workload (deep recursion, untrusted module, embedded host with
a small thread stack) makes this fragile, the right fix is a separate
host-stack-depth counter on `NativeContext`, decremented at the body-entry
prelude and checked against a configurable cap.

Documenting this here so the assumption is explicit, not implicit.

## 12. Phases

### Phase 1A — single landed PR

Must land complete and correct. The recommended subtask order:

1. **Split public entry vs internal entry in the common pipeline.** Add an
   explicit `internal_entry_label`, allocate it at backend creation, bind
   it after the caller stub. Stop equating "internal entry" with "post
   prologue".
2. **Split helper scratch from dead call-link scratch in frame/runtime
   metadata.** Rename `FrameLayoutPlan::call_scratch` →
   `helper_scratch`; drop `MachineFunctionAbi::call_scratch`; drop
   `NativeLocalCallInfo*::call_scratch_base_slot` and
   `LocalCallInfoAbiLayout::call_scratch_base_slot_offset` (the offset
   field lives on `LocalCallInfoAbiLayout`, *not* `CallDispatchView`).
   Update `runtime::layout::tests`. Frame planning no longer reserves
   call-link slots.
3. **Change MIR shape.** Drop `call_link_base` from `CallDirect`/
   `CallIndirect`. Add `caller_result_base: MachineReg`. Drop
   `MachineCallLinkLayout` from `MachineModuleAbi`. Update `validate.rs`,
   `ownership.rs`, the regalloc helpers, and the MIR pretty-printer.
4. **Change local indirect-call metadata layout.** Rewrite
   `lower_module.rs:1340–1390` (the indirect-call call-link write path)
   to compute `caller_result_base` and emit the new `CallIndirect` shape.
5. **Add the public-entry caller stub.** Per backend: a
   `lower_root_caller_stub()` helper that builds the root call record,
   does the platform `bl/call` to `internal_entry_label`, then falls
   through to `lower_epilogue()` and `ret`. No status check inside the
   stub — `C_RET0` already holds 0/error from the body's terminal
   sequence (§9). Delete `seed_root_call_link()`, `native_root_return()`,
   and the `NativeCode::root_return` field. Adjust `eval.rs` to no
   longer pass `root_return` and to read results directly from
   `stack_base` (which is where the unified `Return` will copy them).
6. **Add backend-private call-record lowering and register-based status
   check.** Per arch `lower_call_direct` / `lower_call_indirect`: push
   the record, advance `fp_reg`, native `bl/call`,
   `cbnz w0, body_local_error_label` (arm64) /
   `test eax,eax; jnz body_local_error_label` (x86_64), optional post-
   call `b` if continuation is not adjacent. Implement on arm64 first,
   then x86_64 (with `ret 24` matched on the return side and the
   alignment shim in §7), then armv7a, then emulator.
7. **Unified `Return` lowering and the `body_local_error_label` twin.**
   Per arch: emit the success terminal sequence (`return_ok_label`) which
   pops the call record (or reads from it on x86_64), copies
   `return_results` to `*caller_result_base`, restores `fp_reg`, sets
   `C_RET0 = 0`, and does the native return. Emit `body_local_error_label`
   as a near-copy: same record pop and `fp_reg` restore, *no* result copy,
   and *no* touch of `C_RET0` (the trap stub or upstream BL has already
   set it non-zero). Trap stubs end with
   `bl raise_trap; mov C_RET0, #<trap_kind>; b body_local_error_label`.
   `CallExternal`'s post-helper `cbnz` now targets `body_local_error_label`.
   The shared `return_error_label` is deleted.
8. **`body_emits_native_call` + body prelude.** Compute the bit during
   machine lowering by walking the MIR and asking
   `mir_op_emits_host_call(&kind)` for each op. The predicate returns
   true for `CallDirect`, `CallIndirect`, `CallExternal`, `Trap`, and
   `TrapIf`. Stored in `MachineFunctionAbi::body_emits_native_call`. Emit
   the synthetic body prelude in the common pipeline. Each backend
   interprets the bit per the table in §8: arm64/armv7a `stp` link
   state, x86_64 `sub rsp, 8` alignment shim, emulator nothing. Pair
   with matching backend action at the start of both terminal sequences
   (`return_ok_label` and `body_local_error_label`).
9. **Block layout extension.** Treat call continuation as preferred
   fall-through in the layout pass. Backend lowering emits a trailing `b`
   only when adjacency is not achieved.
10. **Doc update.** This file. Plus a short note appended to
    `sf-nano-core/src/vm/arch/abi.md` that links here for the call-ABI
    details and states the host-stack-depth caveat.
11. **Tests, spectest, coremark.** Run
    `cargo test -p sf-nano-core --features jit --lib`,
    `cargo run --bin sf-nano-spectest -- --backend native`, regenerate
    the coremark dump, run `scripts/compare_llvm.py`. Update
    `lower_tests.rs` fixtures touched by the call-frame-zero work in
    earlier sessions; their expected MIR op counts and shapes will move.
    Confirm the coremark crc still matches (modulo iteration count
    sensitivity), and report instruction-count and CoreMark deltas.

L1 (body-entry link save) is folded into 1A. L2 was rejected because it
would land a BL/RET ABI with a known LR-clobber bug for any window.

#### Backend gap audit (1A.6b / 1A.6c — deferred)

The 1A landing prioritized arm64 (1A.6a) and the emulator's MIR-level
plumbing because that is what coremark and the spectest exercise on the
darwin-arm64 dev machine. The x86_64 and armv7a backends have NOT been
brought current with the new local-call ABI yet, and additionally have
some pre-existing backlog mixed in with the 1A work. They do not compile
under their respective `--target` cross-compile checks. Documenting the
gap here so a follow-up session (ideally one with access to actual
target hardware for runtime verification) can finish them.

Snapshot of `cargo check -p sf-nano-core --target ... --features jit`
errors as of the 1A.9 / 1A.10 / 1C-revert checkpoint:

**x86_64 — 26 errors total**

1A.6b call-ABI port (~13 errors):
- `MachineTerminator::CallDirect` / `CallIndirect` patterns still bind the
  removed `call_link_base` field; need `caller_result_base` instead
  (`x86_64/control.rs:62, 71`).
- `lower_call_direct` / `lower_call_indirect` still write a continuation
  literal into a callee frame slot via `call_link.continuation_offset`
  and `JMP` to the callee. Need to be rewritten to `push caller_result_base;
  push caller_fp; mov fp_reg, callee_fp; call callee; test eax,eax; jne
  body_local_error_label;` plus the optional `jmp continuation` elision.
- `lower_return_sequence` reads continuation/caller_fp/caller_result_base
  from frame `call_scratch` slots. Need to read from the host-stack call
  record, copy `return_results` to `*caller_result_base`, restore caller
  fp, set `RAX = 0`, and `ret 16` (the `ret imm16` form pops the 16-byte
  call record after popping the return address).
- `body_emits_native_call` body prelude `sub rsp, 8` alignment shim and
  matching `add rsp, 8` undo at the start of both terminal sequences
  (success Return and `body_local_error_label`). The plan's "ret 24"
  shorthand is `ret 16` once the alignment shim is paid by the prelude
  rather than the per-call setup.
- New trait methods missing: `lower_root_caller_stub`,
  `lower_body_prelude`, `lower_body_local_error_tail`
  (`x86_64/backend.rs:105`).
- Obsolete `lower_return_ok_status` method still defined; not on
  `ArchBackend` anymore (`x86_64/backend.rs:184`).
- Stale `return_error_label` references in `lower_trap`, trap stub
  emission, and external-call status check (`backend.rs:211`,
  `control.rs:310`, `inst.rs:2106`, `inst.rs:2198`). Need to retarget
  to `body_local_error_label`.
- Stale `EmittedFunction::root_return_offset` /
  `return_error_offset` reads in `make_entry`
  (`backend.rs:305, 307`). The unified Return rewrite removed those
  fields; `CompiledX86_64Entry` no longer needs `root_return` /
  `return_error` either.

Pre-existing non-1A backlog (~13 errors, surfaced because the file
no longer compiles):
- `MachineValue::ReservedReg(_)` not covered in 8 match sites across
  `x86_64/backend.rs`, `control.rs`, `inst.rs`. Need uniform handling
  matching arm64's pattern (treat as identity edge / publish to frame).
- `MachineInstKind::MemoryGrow`, `MemoryFill`, `MemoryCopy` plus 7 more
  not covered in `lower_inst_dispatch` (`inst.rs:36`). The instruction
  vocabulary grew; x86_64 hasn't filled in handlers.
- `Cc::Ne` not found (`inst.rs:1254`). Likely `Cc::NE` after a rename.
- `crate::vm::arch::common::helpers::convert_op_code` import unresolved
  (`inst.rs:30`). Helper was removed/renamed; arm64 doesn't use it but
  x86_64's saturating/trapping truncation paths still call it from
  `inst.rs:2189, 2208, 2235`.

**armv7a — 97 errors total**

1A.6c call-ABI port (~30 errors):
- Same `call_link_base` / `caller_result_base` MIR shape mismatches as
  x86_64, in `armv7a/control.rs` and `armv7a/backend.rs`.
- `emit_call_direct` and `emit_call_indirect` still write continuation
  + caller_fp + caller_result_base into the callee's `call_link` frame
  slots and `BX` (no link). Need conversion to host-stack call record
  + `BL` + status check, mirroring arm64.
- `emit_return_sequence` reads from `call_scratch` slots; needs the
  same rewrite as x86_64's return path but using the armv7a `LDMIA`/
  `LDR` pattern to pop the host-stack call record.
- Missing trait methods: `lower_root_caller_stub`, `lower_body_prelude`,
  `lower_body_local_error_tail`.
- Obsolete `lower_return_ok_status` impl (no longer on the trait).
- Stale `return_error_label`, `root_return_offset`, `call_link`,
  `call_scratch`, `call_scratch_base_slot` references throughout
  `armv7a/backend.rs`, `armv7a/mod.rs`.
- `MachineTrapKind::InvalidConversion` not covered in armv7a's
  trap_code mapping (this is partly armv7a backlog, but matches the
  trap-kind set the new ABI relies on).

Pre-existing non-1A backlog (~67 errors):
- `MachineValue::ReservedReg(_)` not covered in 36 match sites. Larger
  surface than x86_64 because armv7a has more value-shape match arms
  for legalised 32-bit ops.
- `OwnedPreparedFp` is missing arithmetic ops (`Mul<i32>`, `PartialEq`).
  17 errors of the form "cannot multiply `OwnedPreparedFp` by integer".
  Looks like an FP operand prep wrapper that was renamed but the
  implementation wasn't refreshed.
- ~13 mismatched-types errors (various refactors that touched function
  signatures armv7a calls into).

**Why these were left for later**

- darwin-arm64 dev machine cannot runtime-test x86_64 or arm32 code;
  only `cargo check --target ...` is available. The cache-state coupling
  bug in 1C was only caught at coremark runtime, so blind backend ports
  carry real risk.
- Both ports are larger than they look on the surface — the unrelated
  backlog needs to be untangled from the 1A.6 work, and several
  changes (`ret imm16` size, alignment shim placement, push order, status
  reg) have target-specific subtleties.
- Coremark and the WebAssembly spectest currently pass on arm64 with
  the 1A landing as it stands; nothing in production depends on x86_64
  or armv7a today.

**When picking these back up**

- Use a session run on (or able to ssh to) actual x86_64 / arm32
  hardware so the spectest and coremark can validate the unified
  Return path end-to-end.
- Bring x86_64 forward first; it is meaningfully smaller and the
  alignment / `ret imm16` choices generalise cleanly to the armv7a
  port that follows.
- Treat the non-1A backlog (ReservedReg coverage, `MemoryGrow` family,
  `OwnedPreparedFp` arithmetic) as separate prerequisite work — get
  the file compiling on its own merits first, then layer the 1A.6
  call-ABI rewrite on top.
- Cross-reference arm64's `lower_call_direct`, `lower_call_indirect`,
  `lower_return_sequence`, `lower_root_caller_stub`,
  `lower_body_prelude`, and `lower_body_local_error_tail` as the
  reference implementations.

### Phase 1C — clobber sets

After 1A lands and is measured.

- Add `MachineFunctionAbi::clobber_set: u64` (or BitSet if the dynamic
  bank ever exceeds 64 regs).
- After machine lowering, scan each function's MIR for the dynamic regs it
  directly defs.
- Compute the transitive closure over the SCC-condensed direct call
  graph: `clobber(SCC) = ∪ direct_clobber(f) for f in SCC ∪ ∪ clobber(g)
  for g in SCC's direct callees`.
- Indirect call sites: assume the entire dynamic bank.
- At each direct call site, replace `emit_save_dirty_cached_locals()` with
  a selective spill that only writes back cached locals living in registers
  in the callee's `clobber_set`. Indirect calls and SCC-internal calls keep
  the conservative path.

#### Status: REVERTED — first attempt was architecturally incompatible

A first attempt was made and reverted in 2026-04. Notes for the next try:

**What was built (and reverted):**

1. `clobber_set.rs` analysis pass: scans every function's MIR ops/block
   params for GP-bank defs, runs Tarjan SCC over the direct call graph,
   computes the transitive closure, and stores a `u64` bitmask. Functions
   containing `CallExternal`, `CallIndirect`, or living in a non-trivial
   SCC are marked full-clobber (`u64::MAX`).
2. New MIR variant `CallSiteCacheSpill { ty, addr, width, src }` to
   distinguish call-site spill stores from `LocalDropCache` flushes (which
   look identical at the MIR level but have permanently-evict semantics
   that must NOT be removed). `lower_cached::emit_save_dirty_cached_locals`
   was changed to emit this variant for the GpWord path; the new variant
   was threaded through validate, ownership, regalloc, peephole, ir_dump,
   and the arm64+emulator backends (both lower it as a plain store).
3. `remove_redundant_call_site_spills` post-pass: walks every block ending
   in a direct `CallDirect`, finds the consecutive `CallSiteCacheSpill`
   prefix, and removes the entries whose source register is not in the
   callee's clobber set.

**Why it broke (this is the load-bearing finding):**

The post-pass approach is fundamentally incompatible with the cached-local
state machine in `lower_inst.rs`. After every call,
`begin_continuation_block_selective()` calls
`clear_cache_live()` and `clear_cache_dirty()` for ALL cached locals,
unconditionally. The post-call code that needs the local then re-reads from
the frame slot via `LocalGetCache` (which on a miss generates a `Load`).

That sequence is correct ONLY because the conservative spill earlier in the
block has flushed the live register value to the frame slot. If 1C removes
that spill — even though the callee provably doesn't clobber the register —
the post-call reload reads stale data from the frame slot and the program
takes a hard wrong turn (CoreMark reproduced this as an OOB memory access
on the very first removal, even with `SF_CLOBBER_MAX_REMOVE=1`).

The optimization cannot be done as a post-MIR rewrite over the conservative
emission. The spill skip is *coupled* to keeping the cached register live
across the call AND to teaching `begin_continuation_block_selective` to
preserve those specific cache entries instead of clearing all of them.

**What 1C v2 actually needs:**

1. **Lowering-time decision, not post-pass.** The clobber set must be
   known at the moment `emit_save_dirty_cached_locals` runs. That requires
   computing clobber sets in topological order over the SCC condensation
   *before* lowering descendants — chicken-and-egg with SSA→MIR lowering
   that already wants per-function info up front. Two practical options:
   - **Two-pass lowering.** Pass 1 lowers every function with the
     conservative spill set so we can scan defs; pass 2 reruns lowering
     for any function whose callees gained refined clobber info. Wasteful.
   - **MIR-first analysis with cooperative re-lowering.** Lower once, run
     the SCC analysis, then for each direct call site use a backend hook
     that consults the precomputed clobber set when emitting the spill
     and the matching cache-state preservation. The hook lives inside
     `lower_inst.rs` so it can update `cache_live`/`cache_dirty` in the
     same step that decides to skip the spill.
2. **Selective continuation clear.** `begin_continuation_block_selective`
   must take the (now per-call-site) "preserved register set" and clear
   only the cache entries whose backing register is NOT in that set.
3. **Edge-stub parallel moves.** Block-edge parallel moves emitted at the
   arch level (arm64 `block.params` materialization in `arm64/control.rs`)
   are invisible to a MIR-level def scan. A redesign should either move
   those edge moves into MIR (so the def scan sees them) or fold the
   block-param register set into the per-function direct clobber.
4. **Drop-cache vs spill distinction.** The `CallSiteCacheSpill` variant
   IS the right way to keep `LocalDropCache` flushes safe from this
   optimization. Reintroduce it when 1C v2 lands.

**Files touched in the reverted attempt** (for reference when re-attempting):
`machine/clobber_set.rs` (new), `machine/machine_ir/inst.rs`,
`machine/lower_cached.rs`, `machine/lower_module.rs`, `machine/mod.rs`,
`machine/ownership.rs`, `machine/lower_regalloc.rs`,
`machine/peephole/{copy_propagate,helpers}.rs`, `machine/validate.rs`,
`machine/lower_tests.rs`, `arch/arm64/inst.rs`, `arch/emulator/mod.rs`,
`debug/ir_dump.rs`.

### Phase 1D — hoisted stack check

After 1C lands.

- Add `MachineFunctionAbi::max_descendant_frame_bytes: u32`.
- Compute via reverse-topological walk over the SCC-condensed direct call
  graph: `max_descendant(f) = own_frame(f) + max(max_descendant(g) for g in
  direct_callees(f))`.
- SCCs and indirect-call sites use a conservative bound and keep the
  per-call check. Everyone else has the per-call check removed.
- The body prelude gains a single check at the top: `fp_reg + max_descendant
  > stack_end → trap`. The check goes in the same synthetic prelude block
  added by 1A.

## 13. Open questions / things to revisit

- **Indirect call clobber set.** v1 treats indirect calls as full-clobber.
  If profiling shows hot indirect calls in a closed function table, we can
  later compute the union of all candidates' clobber sets and use that.
  Out of scope for 1A–1D.
- **Result aliasing.** The unified `Return` copies results from the callee
  frame into `*caller_result_base`. For 1- and 2-result returns this is one
  or two `str`s. If profiling shows multi-value-heavy workloads, a future
  pass can extend the joint planner to overlap callee return slots with
  caller result-receive slots, eliminating the copy. Out of scope for v1.
- **Host-stack-depth guard.** Documented in §11. Not added in v1.
- **`Return` and the body-prelude pairing.** The current expectation is
  that every `Return` in a `body_emits_native_call` function emits the matching
  backend-specific cleanup before the native return (arm64/armv7a `ldp`,
  x86_64 implicit via `ret 24`). There is exactly one `Return` terminator
  kind in MachineIR, so this is uniform. If we ever introduce early-exit
  returns from inside the body, the same emission rule applies.
  `body_local_error_label` is the same lowering, just reached via a
  different control-flow predecessor.
