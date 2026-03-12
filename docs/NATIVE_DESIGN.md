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

**Rule:** `abi.rs` should define ABI contracts such as external boundary register assignments, call-facing register assignments, register budgets, and entry or edge contracts. It should not define final IR storage nodes like places, values, or addresses.

**Reason:** ABI configuration and IR structure are related, but they are not the same thing. Keeping them separate makes it easier to see whether a concept is a contract the backend consumes or an IR node the compiler transforms.

## Rule 25: VM-Flavored Register Kinds Must Not Survive Into MachineIR

**Rule:** Register kinds such as `Ctx`, `Fp`, `Hot`, `Tos`, and `Tmp` must not survive as semantic register classes in the final machine-facing IR.

**Reason:** These names encode VM meaning or lowering history. A machine IR should use generic registers. If some registers are pinned by ABI, that belongs in ABI metadata, not in the semantic meaning of every IR operand.

## Rule 26: External Boundary Registers Are Opaque To The Backend

**Rule:** If the runtime passes distinguished registers into native code at an external boundary, those registers are opaque below the IR boundary. The backend may know their assigned machine registers, but not their VM meaning.

**Reason:** For example, a runtime base pointer may be passed in a fixed external-boundary register. Above the IR boundary, lowering may use it to derive addresses. Below the boundary, the backend should only see ordinary register-based address computation and memory operations.

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

## Rule 33: Helpers Use One Uniform Logical ABI

**Rule:** All helpers must share one logical call shape: `helper(runtime, metadata, scratch) -> status`. The machine IR must not allow per-helper calling conventions.

**Reason:** Per-helper signatures create another hidden ABI surface that can drift out of sync with runtime code. A single logical helper shape keeps helper lowering mechanical and reviewable.

## Rule 34: MachineIR Uses Opaque External Targets

**Rule:** The machine IR must refer to helpers and other external native targets through opaque external target ids. Machine IR must not encode helper symbols or backend addresses directly.

**Reason:** Helper meaning and symbol resolution are sidecar link data, not machine code semantics. Keeping external target identity opaque prevents helper policy from leaking into the machine layer.

## Rule 35: External Binding Data Owns Helper Meaning

**Rule:** Closed helper symbols and their mapping to real helper wrapper addresses belong in external binding metadata beside the machine module, not in the machine IR itself.

**Reason:** The backend needs a static resolver table, but machine IR should not know what a given external target id means. Sidecar binding data is the correct home for that meaning.

## Rule 36: Helper Metadata Is Read-Only And Helper Scratch Is Writable

**Rule:** Helper-specific immutable data must live in read-only sidecar metadata. Helper-specific inputs and outputs must live in writable native scratch memory. These are different things and must not be conflated.

**Reason:** Read-only metadata belongs beside the code. Writable helper data must not live in Wasm memory, Wasm globals, or code metadata. Keeping the two separate makes the contract explicit and avoids hidden aliasing or lifetime problems.

## Rule 37: Helper Scratch Lives In Native Frame Memory

**Rule:** Writable helper scratch must come from planned native frame space, not from Wasm memory, Wasm globals, or backend-invented ad hoc host stack blobs.

**Reason:** Helper scratch is part of the native execution contract. Planning it above machine IR keeps helper calls uniform and prevents one backend from inventing a private scratch allocation model.

## Rule 38: Helpers Return Only Status In ABI Registers

**Rule:** Helper calls return only a status code in the helper ABI. Any helper-specific outputs must be written into writable scratch memory.

**Reason:** Multi-value helper returns complicate backend lowering and make the ABI harder to keep stable. A single status return keeps the ABI small and lets the machine IR treat helper inputs and outputs as ordinary memory traffic.

## Rule 39: Call Indirect And Local Control Transfer Stay Native

**Rule:** `call_local` and local `call_indirect` must stay in native control flow. They must not rely on helper-driven control transfer.

**Reason:** If local control transfer leaves native execution and comes back through a helper, then the machine IR still is not carrying enough control structure. Helpers are for true runtime boundaries, not for ordinary local calls.

## Rule 40: Table And Global Layout Knowledge Belongs Above MachineIR

**Rule:** Knowledge of table entry shape, global layout, memory view layout, and runtime object offsets must be resolved above the machine IR into explicit address computations and checks.

**Reason:** The backend should not know how a table entry or global is laid out semantically. It should only lower the resulting loads, stores, compares, and branches.

## Rule 41: MachineIR Must Be Sufficient To Lower The Whole Function

**Rule:** Once machine IR is formed, every function must be fully lowerable to an ISA backend without falling back to another executor or inventing architecture-local semantics.

**Reason:** Machine IR is the final shared contract. If a backend still needs a debug executor, a shared interpreter, or architecture-only semantic helpers to finish the job, then the IR boundary is still wrong.

## Rule 42: LIR Is The Prepared Frontend Handoff

**Rule:** LIR is the prepared output of the frontend and the input to the native backend. It must already encode the stack-window and canonical-slot guarantees the backend relies on.

**Reason:** This engine is not a traditional optimizing compiler with a later general register allocator. The backend depends on frontend preparation to make register fit trivial.

## Rule 43: LIR Must Preserve The Fixed TOS-Window Contract

**Rule:** LIR must preserve the target-independent TOS-window contract. The number of transient live SSA stack values visible to the backend must never exceed the configured TOS register budget.

**Reason:** The fixed TOS window is not an optional optimization detail. It is the guarantee that lets the backend map transient values into a small fixed register set without running a traditional register allocator.

## Rule 44: Frontend Spills And Fills To Enforce Backend Fit

**Rule:** If stack depth or transient live values exceed the available TOS window, the frontend must emit spill and fill operations against canonical operand slots before the program reaches the backend.

**Reason:** Backend fit is guaranteed by preparation, not discovered by backend allocation. The backend should never need to invent new spills just to make transient stack values fit.

## Rule 45: LIR Must Leverage Stack Discipline, Not Erase It

**Rule:** LIR may use SSA values for transient stack values, but it must still preserve the crucial Wasm stack property that operations consume the top of the stack and that the top window is the hottest near-term state.

**Reason:** This stack discipline is what makes the engine fast without a traditional optimizer or lifetime-based register allocator. If LIR erases that structure into arbitrary full-stack SSA, the backend loses the main advantage of the design.

## Rule 46: Locals Keep Canonical Slot Homes

**Rule:** Local accesses in LIR use canonical frame-slot identity such as `ReadSlot` and `WriteSlot`. Hot-local caching must not change the canonical local slot layout.

**Reason:** Calls, returns, and frame layout still rely on stable slot identity. Register-cached locals are mirrors of those canonical homes, not replacements for them.

## Rule 47: Hot-Local Policy Travels As Metadata, Not As LIR Storage Kinds

**Rule:** The frontend may pass down hot-local cache budget and preferred local-slot ranking, but LIR itself must not encode local accesses as special hot-local storage kinds.

**Reason:** The backend needs to know how many locals it may cache and which locals are preferred, but canonical local identity must stay slot-based. The cache assignment is execution policy, not semantic storage.

## Rule 48: True Runtime Boundaries Must Be Split Out In LIR

**Rule:** LIR must distinguish normal leaf operations from true runtime-boundary operations such as growth and segment lifecycle ops.

**Reason:** The layer between LIR and machine IR must not rediscover helper policy by pattern-matching arbitrary primitives. The runtime boundary needs to already be explicit in LIR.

## Rule 49: LIR Calls Stay Semantic But Respect Canonical Stack Layout

**Rule:** Direct local calls, indirect calls, and external calls remain semantic LIR operations, but their argument and result publication still follows canonical stack and frame layout prepared above the backend.

**Reason:** Call-link layout and machine continuation structure belong below LIR, but stack-layout guarantees must already be stable by the time LIR is handed to the backend.

## Rule 50: Backend Register Assignment Must Stay Trivial

**Rule:** The native backend must only need to:

- assign fixed registers for runtime anchors such as `ctx` and `fp`
- fit transient live TOS values into the fixed TOS register window
- map selected local slots into the fixed local-cache registers

It must not require a general-purpose lifetime-based register allocator.

**Reason:** The whole engine is designed around a prepared stack-window handoff so code generation remains simple, fast, and predictable.

## Current Design Decisions

### Helper ABI

**Decision:** The helper ABI is `helper(runtime, metadata, scratch) -> status`.

**Reason:** This keeps the register budget small, avoids helper-id dispatch, and avoids multi-value Rust ABI returns. It is a documented logical contract, not a separate code-level ABI type.

### External Targets

**Decision:** Machine IR stores opaque external target ids, not helper kinds and not raw addresses.

**Reason:** Real helper addresses are pointer-sized finalization details, and helper meaning belongs in sidecar binding data. The shared machine IR should stay ISA-neutral and should not encode either backend addresses or helper symbols directly.

### Compare And Select

**Decision:** Machine IR includes value-producing compare operations and `select` as straight-line instructions.

**Reason:** Wasm relational operators and `select` produce values. Forcing them into CFG structure would expand the program unnecessarily and would leak a frontend lowering choice into machine IR.

### Module Ownership

**Decision:** Function ids, constant ids, and external target ids are owned by a top-level machine module artifact, not by isolated function bodies.

**Reason:** Direct-call targets, helper metadata ids, and opaque external targets need one canonical allocation domain. A machine module makes those references explicit and validates them in one place.

### Runtime Layout

**Decision:** Final machine runtime contract does not carry memory/global/table layout offsets.

**Reason:** Those offsets are runtime semantics that must be consumed above machine IR when lowering into explicit address computation.

### Helper Metadata

**Decision:** Helper metadata is a read-only sidecar record referenced from machine IR by a symbolic constant id.

**Reason:** The backend only needs to materialize a metadata pointer. It must not understand helper-specific metadata structure.

### Helper Scratch

**Decision:** Helper scratch is writable native frame memory planned above machine IR.

**Reason:** Helper inputs and outputs need writable storage, but that storage must not come from Wasm memory, Wasm globals, or read-only code metadata.

### Helper Scope

**Decision:** Helpers are reserved for true runtime boundaries such as host calls, growth operations, and segment-lifecycle operations.

**Reason:** Normal memory/global/table access and local control transfer should lower above machine IR into explicit loads, stores, checks, and CFG.

### External Bindings

**Decision:** Closed helper symbols live in sidecar external binding data and are resolved to real helper wrapper addresses during backend finalization.

**Reason:** The backend needs a static resolver table, but that table is part of link/finalization data, not part of machine IR semantics.

### LIR Lowering Stages

**Decision:** LIR lowering is structurally split into four steps: semantic/planned alignment, block-boundary shaping, straight-line body lowering, and terminator lowering.

**Reason:** Stack-height reconstruction and CFG boundary shaping are semantic concerns. Keeping them separate from body and terminator lowering makes the pipeline easier to audit and prevents one file from silently owning all lowering policy.

### LIR Stack Errors

**Decision:** LIR lowering treats stack underflow or stack-shape mismatch as an explicit internal error. It must not silently clamp stack reads or pops.

**Reason:** Silent stack clamping hides semantic/planning mismatches and produces misleading IR. A semantic lowering layer should fail loudly when its stack model becomes inconsistent.

### Engine Style

**Decision:** The engine is intentionally a prepared single-pass compiler. It does not rely on traditional optimization passes or a traditional general register allocator.

**Reason:** The design gets its efficiency from Wasm stack discipline and from frontend preparation rather than from heavyweight backend analysis.

### Prepared LIR Contract

**Decision:** LIR is not arbitrary full SSA over the whole stack. It is a prepared IR where transient stack values are already bounded to the configured TOS window through explicit spill/fill against canonical operand slots.

**Reason:** This is the contract that makes backend lowering easy. The backend can assume transient value fit instead of computing it.

### TOS Register Budget

**Decision:** The number of transient value registers is a frontend-known contract derived from backend budget. The frontend must prepare LIR so that transient live SSA values always fit inside that fixed TOS window.

**Reason:** Without this contract, the backend would need real register allocation or ad hoc spilling, which is explicitly not the intended design.

### Local Cache Budget

**Decision:** The number of local-cache registers and the preferred local-slot ranking are passed down from frontend analysis, but locals keep canonical frame-slot homes in LIR.

**Reason:** Calls and frame layout require stable slot identity. Cached locals are just fast mirrors of canonical slot homes.

### Backend Simplicity

**Decision:** Backend lowering should be close to mechanical:

- assign fixed registers for runtime anchors
- place the bounded TOS window into the fixed transient-value registers
- swap the chosen canonical local slots into the fixed local-cache registers

No general register allocation should be required.

**Reason:** This simplicity is the whole point of preparing LIR around stack discipline and frontend-enforced fit.

## Current Helper Classification

### Must Stay In Rust

**Decision:** These operations remain true Rust/runtime helpers:

- `call_external`
- `memory.grow`
- `table.grow`
- `memory.init`
- `data.drop`
- `table.init`
- `elem.drop`

**Reason:** These are real runtime boundaries. They depend on host callbacks, runtime-owned allocation growth, or runtime-owned segment lifecycle, so keeping them as helpers is a real contract boundary rather than a lowering shortcut.

### Must Lower Above MachineIR

**Decision:** These operations must not remain helpers in the final design:

- normal memory loads and stores
- `memory.size`
- `global.get`
- `global.set`
- `table.get`
- `table.set`
- `table.size`
- `call_local`
- local `call_indirect`
- float arithmetic and conversions
- ordinary integer/float compares that produce values

**Reason:** These do not fundamentally differ by ISA except in encoding and register choice. They should lower above machine IR into explicit address computation, loads, stores, checks, arithmetic, and CFG.

### Gray Area: Temporary Bring-Up Helpers Only

**Decision:** These may temporarily remain helpers during refactor bring-up, but they are not considered fundamental Rust boundaries:

- `memory.copy`
- `memory.fill`
- `table.copy`
- `table.fill`

**Reason:** These can be lowered later as explicit loops, memcpy/memmove-style lowering, or narrow runtime primitives. They are not inherently helper-shaped in the final design, but they may remain helpers briefly if that speeds staged refactoring without changing the intended boundary.

## Conceptual Walkthrough

This section is not another set of rules. It is a walkthrough of how the whole pipeline is supposed to work end to end.

The most important point is that this engine is not designed like a traditional compiler backend. It does not try to preserve arbitrary full SSA and then solve register pressure with a general-purpose register allocator. The engine gets its efficiency from the Wasm stack discipline:

- stack values are used from the top down
- the top of the stack is the hottest near-term state
- if we keep a fixed top-of-stack window in registers, we get a very simple and very effective execution model

That is why frontend preparation matters so much.

### 1. Wasm Instructions To Semantic Form

The first step removes bytecode syntax and keeps only program meaning:

- structured control flow
- explicit branch targets
- local/global/table/memory operations
- call shape: local, indirect, external
- runtime-boundary operations

At this point we still care about Wasm semantics, not about registers or native code shape.

### 2. Semantic Form To Prepared LIR

This is the important engine-specific preparation stage.

The backend passes down configuration such as:

- `tos_lanes`
- `cached_locals`
- other fixed native budget information

This stage does not "choose" those numbers. They are provided by the target/backend configuration.

This stage also does not invent a new local layout. Local layout follows the Wasm/frame contract. In particular, arguments still appear where the callee expects them in its frame layout, and the frame shape still follows the calling convention the engine already relies on.

What this stage does do:

- preserve the canonical frame layout
- preserve canonical local slots
- analyze which locals are good local-cache candidates
- pass down that hot-local preference metadata for later local-cache swapping
- enforce the fixed TOS-window contract

The key job here is to emit explicit `spill` / `load` in LIR so that transient live stack values never exceed `tos_lanes`.

That means if a block would otherwise have 100 live stack values, this stage does not hand 100 live SSA values to the backend. Instead:

- it keeps only the top `tos_lanes` transient values live in the TOS window
- it spills deeper transient values to their canonical operand slots
- it reloads them when they come back into the top window

This is how the frontend guarantees backend fit without a traditional register allocator.

### 3. What LIR Focuses On

Prepared LIR is the frontend/backend contract.

It should focus on:

- CFG
- SSA values for the current TOS window
- canonical frame slots for locals and spilled stack values
- explicit spill/load against operand slots when the TOS window is not enough
- semantic calls
- semantic runtime-boundary ops

It should not focus on:

- physical register names
- general register allocation
- ISA-specific addressing
- machine-level continuation encoding

For locals, LIR keeps canonical slot identity:

- `local.set 0` writes `slot0`
- `local.set 4` writes `slot4`

The hot-local analysis does not change that. It only passes down which canonical local slots should later be swapped into the fixed local-cache registers.

For transient stack values, LIR does carry the prepared TOS discipline. This is different from local caching. The TOS-window preparation must already be reflected in LIR so the backend never sees more transient live values than it can hold.

### 4. Calls In Prepared LIR

Calls need special care.

The current design uses stack/frame layout to pass call arguments. That means before a call, this frontend-to-LIR preparation stage must:

- spill any still-live TOS values to the correct `fp[...]` operand slots if they are needed there
- make sure call arguments are in the canonical stack/frame locations the callee expects

So LIR must be able to contain explicit operations like:

- `spill(v1, slot3)`
- `load(slot3, v2)`

or whatever exact naming we settle on for those prepared stack-window operations.

This is important: LIR does not need to say "SSA value `v1` lives in register X". But it does need to say when a transient value is published to its canonical frame slot so that calls and later reloads are correct.

Local cache is different. We do not need the same kind of spill/load modeling in LIR for local-cache registers. Local-cache handling happens later, between LIR and MachineIR.

### 5. Between LIR And MachineIR

This is not another public IR. It is the lowering stage that consumes prepared LIR and produces machine-shaped IR.

Its job is simple because LIR has already done the hard fit work.

It needs to:

- assign the bounded TOS live set into the fixed transient-value register window
- assign fixed runtime registers such as `ctx` and `fp`
- use the hot-local analysis metadata to swap selected canonical local slots into the fixed local-cache registers
- save modified local-cache registers before a call if the callee may overwrite them
- reload those cached locals back into their cache registers after return
- lower semantic memory/global/table operations into explicit address computation, checks, loads, and stores
- lower semantic calls into explicit machine control flow and call-link protocol

This stage should still stay simple. It is not doing general RA. It is mostly a mechanical use of:

- bounded TOS live values
- fixed local-cache budget
- fixed runtime registers

### 6. What MachineIR Focuses On

MachineIR is where VM-shaped concepts disappear.

MachineIR should contain:

- generic registers
- explicit addresses
- loads/stores
- arithmetic
- branches
- jump tables
- call/return/control-transfer structure
- explicit helper boundaries

MachineIR should not contain:

- TOS lanes
- hot-local ops
- Wasm-local/global/table semantic names
- planner spill provenance
- VM register kinds

By this point, the useful stack preparation has already done its job. MachineIR is just the architecture-neutral machine view of the already-prepared program.

### 7. MachineIR To Machine Code Or Emulator

From MachineIR, the final step is straightforward.

The real backend:

- maps generic machine registers to ISA registers
- encodes instructions
- lays out blocks
- patches direct-call targets
- resolves helper/external targets
- emits code and sidecar data

The emulator:

- executes the same MachineIR
- uses the same runtime contract
- serves as the debug oracle for the machine layer

It must not invent a different execution model from the one the real backend consumes.

### Example Walkthrough

Assume:

- `tos_lanes = 4`
- `cached_locals = 3`
- local `0` is hot
- local `4` is cold

Wasm:

```wat
local.get 0
i32.const 1
i32.add
local.set 0
local.get 4
call $g
return
```

Prepared LIR conceptually looks like:

```text
v0 = ReadSlot(local_slot(0))
v1 = Leaf(I32Const 1)
v2 = Leaf(I32Add, [v0, v1])
WriteSlot(local_slot(0), v2)

v3 = ReadSlot(local_slot(4))
spill(v3, operand_slot(0))
CallInternal g
load(operand_slot(result0), v4)
Return [v4]
```

The exact operation names can change, but the important ideas are:

- locals use canonical local slots
- call arguments are published through canonical stack/frame layout
- transient stack values are explicitly spilled/loaded when needed
- backend fit is already guaranteed

Then the LIR-to-MachineIR lowering does the simple backend work:

- map the bounded live TOS values into the fixed TOS registers
- map selected local slots into the fixed local-cache registers
- save dirty cached locals before call if needed
- restore/reload them after return
- turn the call into machine-level control transfer

This is the core design intention of the engine:

- frontend preparation uses Wasm stack discipline to guarantee fit
- LIR is prepared, not arbitrary full SSA
- backend stays simple and single-pass
- no traditional optimizer or general register allocator is required
