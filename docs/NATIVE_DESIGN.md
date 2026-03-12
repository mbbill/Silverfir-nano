# Native Backend Design Rules

This document records the current design rules for the native backend refactor.

The point of these rules is simple:

- keep VM semantics above `NativeIR`
- keep ISA details below `NativeIR`
- avoid transitional shortcuts becoming permanent architecture

Each section states one rule and why it exists.

## Rule 1: Reference Is Debug-Only

**Rule:** The reference backend exists only to validate correctness above the ISA layer. It must be compiled only for debug builds, and it must never participate in production execution.

**Reason:** The reference backend is a debugging oracle for `NativeIR` semantics. It is not a fallback production engine, not a mixed-mode executor, and not something compiled native code may jump into.

## Rule 2: No Shared-Entry Fallback In Production Native Code

**Rule:** A compiled native function must not be a stub that hands execution to the reference machine or any other non-native executor.

**Reason:** Once a function is compiled as native code, its execution model must be entirely native. A per-function fallback path destroys the backend boundary, hides correctness bugs, and makes performance numbers meaningless.

## Rule 3: Enter Native Once, Exit Native Once

**Rule:** For a root invocation, execution enters the native ABI once from Rust and stays there until final return or trap.

**Reason:** The native backend should behave like one native world. Local calls, indirect local calls, and returns must remain inside that world. Re-entering Rust in the middle of normal native control flow is a design failure, not a normal path.

## Rule 4: Most Functions Have Only One Real Entry

**Rule:** The real entry for a compiled function is its internal native-ABI entry. Most functions should expose only that entry to other native functions.

**Reason:** Internal functions are not host entrypoints. They should not carry host-facing prologue logic or runtime boundary setup just because they might be called by other native functions.

## Rule 5: Public Shims Exist Only At The Runtime Boundary

**Rule:** A separate public shim may exist only for functions that must be callable from Rust or from the host boundary, such as `_start` or exported roots.

**Reason:** The public shim is a runtime bridge, not a normal call target. Its job is to establish the native ABI, seed the exit path, and hand control to native code. It must not become the normal local-call mechanism.

## Rule 6: Local Calls Stay In Native ABI

**Rule:** `call_local` and local `call_indirect` must remain inside the native ABI. They must not route through Rust, a public shim, or the reference executor.

**Reason:** Native local control transfer is part of the backend contract. If a local call leaves native execution, then the backend still owns VM control-flow semantics it was supposed to finish.

## Rule 7: Helper Calls Are Explicit Runtime Boundaries

**Rule:** Helper-backed operations are allowed as explicit runtime boundaries, but they must resume native execution directly instead of reinvoking a local function through a host-facing entry.

**Reason:** Helpers are different from local calls. They may be needed for runtime services, traps, host callbacks, or cold paths. Even so, they must return enough continuation state to resume native execution cleanly.

## Rule 8: NativeIR Contains Only ISA-Relevant Concepts

**Rule:** A concept belongs below `NativeIR` only if it truly differs by ISA.

**Reason:** This is the main boundary test. If ARM64 and x86_64 would implement the same semantics the same way except for instruction encoding and register choices, then that concept should not live below `NativeIR`.

## Rule 9: VM Semantics Must Be Lowered Above NativeIR

**Rule:** Wasm-level semantics such as `GlobalGet`, `GlobalSet`, `MemorySize`, `MemoryGrow`, table access semantics, and runtime object layout knowledge should be lowered above `NativeIR`.

**Reason:** These are VM semantics, not ISA semantics. The backend should not know about "global 7", "memory 0", or other Wasm object concepts. It should only see concrete addresses, memory operations, control flow, and explicit runtime boundaries.

## Rule 10: Memory And Global Access Become Addressing Plus Memory Ops

**Rule:** By the time code reaches `NativeIR`, normal memory and global access should already be expressed as explicit address computation, bounds checks, and memory reads or writes.

**Reason:** This keeps the backend mechanical. ARM64 and x86_64 may encode the final load/store differently, but they should not be responsible for inventing where the address comes from or what runtime structure is being accessed.

## Rule 11: `memory.grow` Is Not An ISA Concern

**Rule:** `memory.grow` itself is not backend-specific. Its effect on subsequent memory base and length must be represented above `NativeIR`.

**Reason:** The fact that `memory.grow` invalidates previous memory views does not differ by ISA. The backend may need to materialize the resulting loads and stores, but it should not own the semantic policy for when memory state changes.

## Rule 12: Runtime State In `ctx` Is Fine; Long-Lived Register Caching Is Not Default

**Rule:** Canonical runtime state may live in `ctx`, but backend register caching of that state must be short-lived and validity-scoped, not assumed across an entire function by default.

**Reason:** `ctx` is the source of truth. Register caching is only an optimization. Whole-function caching of memory or global views is fragile because helpers, growth operations, and other boundaries can invalidate it. The safer default is to cache within a region whose validity is explicit.

## Rule 13: Cache Invalidation Policy Belongs Above The ISA Layer

**Rule:** If a value may become stale because of runtime semantics, that invalidation policy should be explicit above `NativeIR`, not guessed inside one architecture backend.

**Reason:** Backend-specific heuristics such as "this callee probably does not change views" leak VM policy into the ISA layer. The backend should consume an explicit contract, not infer semantic facts ad hoc.

## Rule 14: Control Flow Still Belongs Below NativeIR

**Rule:** Branches, jump tables, returns, traps, and the final shape of native call/continuation transfer remain valid `NativeIR` concerns.

**Reason:** These are exactly the kinds of things that do differ by ISA. ARM64 and x86_64 encode and organize them differently, so they belong at or below the native backend boundary.

## Rule 15: Below NativeIR, The Backend Should Be Mechanical

**Rule:** The architecture backend should only:

- choose machine registers
- emit loads, stores, arithmetic, and branches
- implement the concrete native ABI and continuation protocol
- honor explicit helper and trap boundaries

**Reason:** The backend should not own Wasm semantics, object-model policy, or runtime meaning that should already have been decided above it.

## Rule 16: Mixed-Mode Execution Is Forbidden

**Rule:** One native module execution must not silently mix direct native execution, reference execution, and helper-driven re-entry in ways that change the control model.

**Reason:** Mixed execution makes correctness bugs hard to localize and performance hard to reason about. The runtime model must be obvious: either this function is running as native code, or it is not.

## Rule 17: Design Names Must Stay Stable

**Rule:** The runtime model should use a small, consistent vocabulary:

- `public shim` for the root or exported runtime bridge
- `internal entry` for the real local-call target
- `exit/trap stub` for leaving native execution

**Reason:** Design drift often starts with naming drift. Consistent names make it easier to tell whether a path is a true runtime boundary or just another internal branch.

## Rule 18: Every New Backend Rule Must Pass The ISA Test

**Rule:** When adding anything below `NativeIR`, ask: "Does this truly differ between ARM64 and x86_64, or am I leaking VM semantics into the backend?"

**Reason:** This is the simplest guardrail for the refactor. If the answer is "no", the logic belongs above `NativeIR`.

## Rule 19: IR Operand Types Belong With IR

**Rule:** Any type that is a direct operand, place, value, or address inside `NativeIR` must live with the IR definitions, not in an ABI-only module.

**Reason:** Structural ownership should match semantic ownership. If a type is validated, traversed, dumped, and transformed as IR, then it is part of IR.

## Rule 20: NativeIR Must Not Preserve Planning Provenance

**Rule:** `NativeIR` must not distinguish storage by planner-origin names such as "frame slot" versus "spill slot".

**Reason:** By the time lowering reaches `NativeIR`, those are no longer semantic categories. They are just native memory locations. Carrying planning provenance into the final machine-facing IR leaks higher-level lowering details past the boundary.

## Rule 21: NativeIR Should Converge To MachineIR

**Rule:** The final form of `NativeIR` should be a true machine IR. It should model only generic registers, explicit addresses, memory operations, control flow, and explicit call or helper boundaries.

**Reason:** ARM64, x86_64, and other real ISAs do not have VM concepts like context, hot locals, TOS lanes, frame homes, or spills. They have registers and memory. The IR should mirror that reality.

## Rule 22: Memory Traffic Must Be Explicit In NativeIR

**Rule:** Loads and stores must be explicit `NativeIR` operations. Memory access must not be hidden inside generic place-to-place moves.

**Reason:** Explicit memory operations make dataflow, clobbers, and backend responsibilities visible. They also keep the ISA layer honest: a backend lowers a load or store, not an abstract "move from frame place to tmp place".

## Rule 23: LIR Slot Types Must Not Leak Into Final NativeIR Operands

**Rule:** LIR or planning slot types such as `FrameSlot` must not appear as first-class storage operands in finalized `NativeIR`.

**Reason:** Once lowering crosses into `NativeIR`, the backend should no longer depend on LIR storage vocabulary. Any remaining memory reference should already be an explicit native address or offset form.

## Rule 24: ABI Modules Define Contracts, Not IR Storage Nodes

**Rule:** `abi.rs` should define ABI contracts such as pinned inputs, call-facing register assignments, register budgets, and entry or edge contracts. It should not define final IR storage nodes like places, values, or addresses.

**Reason:** ABI configuration and IR structure are related, but they are not the same thing. Keeping them separate makes it easier to see whether a concept is a contract the backend consumes or an IR node the compiler transforms.

## Rule 25: VM-Flavored Register Kinds Must Not Survive Into MachineIR

**Rule:** Register kinds such as `Ctx`, `Fp`, `Hot`, `Tos`, and `Tmp` must not survive as semantic register classes in the final machine-facing IR.

**Reason:** These names encode VM meaning or lowering history. A machine IR should use generic registers. If some registers are pinned by ABI, that belongs in ABI metadata, not in the semantic meaning of every IR operand.

## Rule 26: Pinned ABI Inputs Are Opaque To The Backend

**Rule:** If the runtime passes a distinguished input register into native code, that register is opaque below the IR boundary. The backend may know its assigned machine register, but not its VM meaning.

**Reason:** For example, a runtime base pointer may be passed in a pinned input register. Above the IR boundary, lowering may use it to derive addresses. Below the boundary, the backend should only see ordinary register-based address computation and memory operations.

## Rule 27: Context Is An ABI Contract, Not An ISA Concept

**Rule:** Runtime notions such as "context" must exist only as ABI contracts or as address derivation performed above the machine IR. They must not appear as semantic concepts inside the ISA backend.

**Reason:** An ARM64 backend should not "know about context" any more than it knows about globals or memories. It should simply lower instructions that read registers, compute addresses, and perform loads, stores, branches, and calls.

## Rule 28: Prefer Generic Registers Over Named VM Registers

**Rule:** The default register namespace in the final machine IR should be generic, such as `R0`, `R1`, and so on. Any special calling-convention meaning should be attached by separate ABI metadata, not by distinct IR register enums.

**Reason:** This keeps the IR honest. It becomes a real abstraction over multiple ISAs instead of a VM-specific IR that happens to be lowered into machine code later.

## Rule 29: Control-Transfer Calls Must Be Explicit In CFG

**Rule:** Calls that transfer control to another local function and later resume in the caller must be explicit CFG constructs in the machine IR, with explicit continuation blocks.

**Reason:** A local call is not just another arithmetic instruction. It leaves the current block and resumes later. Hiding that control flow inside a normal instruction forces the backend to invent hidden labels and implicit continuation structure.

## Rule 30: Continuations Are First-Class MachineIR Blocks

**Rule:** Any post-call continuation must exist as an explicit machine IR block, not as a hidden backend-only patch target.

**Reason:** Making continuations explicit keeps call and return structure visible to validation, debugging, and backend lowering. It also prevents architecture-specific "special continuation code" from becoming a back door for missing IR structure.

## Rule 31: Local Return Uses One Uniform Call-Link Contract

**Rule:** All compiled local returns use the same call-link protocol: publish results, restore caller frame state from the planned call-link area, and jump to the stored continuation.

**Reason:** The root function is not a special return model. The public shim should simply seed the root call-link area so that ordinary local return logic works at the root as well.

## Rule 32: Public Shims Only Seed Native Execution

**Rule:** A public shim may establish the root native ABI, initialize the root call-link state, and transfer control to the function body. It must not contain extra execution semantics.

**Reason:** The public shim is a boundary adapter. If it starts owning special behavior, the real function body no longer has a uniform native calling model.

## Rule 33: Helpers Must Use Typed Machine-Level ABI Contracts

**Rule:** Helper calls must be described by typed machine-level call contracts: explicit argument values, explicit result values, explicit clobbers, and explicit continuation behavior.

**Reason:** A helper boundary is still a call boundary. It must not smuggle VM semantics through generic metadata blobs or implicit frame conventions that only one backend understands.

## Rule 34: Transparent Helpers Are Normal Calls, Not Semantic Escape Hatches

**Rule:** A helper that falls through in the same function must lower as an ordinary machine-level call with explicit clobber behavior and explicit result registers.

**Reason:** Transparent helpers are still backend-visible calls. They should not force hidden spilling policy or hidden runtime-state refresh rules inside one architecture backend.

## Rule 35: Control-Transfer Helpers Must Return Native Resume State

**Rule:** If a helper may resume execution at a different function body or frame, it must return explicit native resume state, such as the next entry and next frame base.

**Reason:** This keeps control transfer in the native world. The helper computes the next native target; the backend resumes there directly instead of reinvoking a function through a host-facing entry.

## Rule 36: Table And Global Layout Knowledge Belongs Above MachineIR

**Rule:** Knowledge of table entry shape, global layout, memory view layout, and runtime object offsets must be resolved above the machine IR into explicit address computations and checks.

**Reason:** The backend should not know how a table entry or global is laid out semantically. It should only lower the resulting loads, stores, compares, and branches.

## Rule 37: MachineIR Must Be Sufficient To Lower The Whole Function

**Rule:** Once machine IR is formed, every function must be fully lowerable to an ISA backend without falling back to another executor or inventing architecture-local semantics.

**Reason:** Machine IR is the final shared contract. If a backend still needs a debug executor, a shared interpreter, or architecture-only semantic helpers to finish the job, then the IR boundary is still wrong.

