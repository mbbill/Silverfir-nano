- Native emission is the last pipeline stage and is the only stage that commits
  to ISA-specific encodings; MachineIR is the boundary between the shared
  pipeline above and the per-target backends below.

- The engine carries six native backends behind one MachineIR contract:
  x86_64, ARM64, RISC-V 64, RISC-V 32, ARMv7-A (A32), and ARMv7-M/Thumb-2; the
  ARM32 backend shares one codebase that emits either A32 or Thumb-2 encodings.

- A non-native emulator backend executes MachineIR directly and is used for
  testing and the `emu64` / `emu32` configurations.

- Each backend declares its register policy as a `BackendConfig` preset (budget
  unit size, GP/FP volatile and preserved lane counts, argument lanes, call
  scratch); the physical register mapping and ABI constraints stay in the
  backend's own ABI code, which selects policy, not the shared compiler.

- Backend lowering works only in terms of mapped MachineIR registers,
  scratch-pool allocations, and semantic encoder helpers, never hard-coded
  physical register names; ad hoc temporaries come from a per-backend scratch
  pool with explicit ownership (`scratch_pool`).

- The prologue loads the pinned fast-path state (context, frame pointer,
  `mem0_base`, `mem0_size`) into fixed registers once per function entry, and
  the native pipeline emits shared tail regions for normal return, stack-
  overflow trap, error return, and deferred traps.

- Block-parameter transfers at edges are resolved as true parallel moves with
  scratch-based cycle breaking (`emit_parallel_moves`).

## Facts

- 2026-06-14 invariant: two kinds of runtime boundary must NEVER be conflated,
  and the whole pipeline depends on keeping them apart. (1) RUNTIME CALLS — JIT
  code calling into the runtime (e.g. WASI) — happen ONLY on functions marked
  external/imported, and are visible to every stage because they come from the
  original wasm import instruction from the start; no stage invents one. (2)
  SPECIAL HANDLERS (memory.grow, memory.copy, table.grow, the GC/ref ops, …)
  must stay GENERIC instructions until arch: upstream stages cannot know whether
  a given arch can lower one directly — emit instructions inline to bump the
  memory limit like an ordinary memory write, with no runtime call — so that
  decision belongs to arch alone, which sees every instruction and decides per
  op how to lower it. Special-casing these early in the pipeline is the recurring
  failure mode. The PRESERVED-HELPER system is ONLY for arch-lowered instructions
  that an arch chose to implement via a helper — nothing else routes through it
  (sourced).

- 2026-03-14 (ecf26a68) measurement: the native backend's standing across the
  WASI suite — CoreMark 3.59x wasm3 / ~1.03x Cranelift on M4, competitive with
  or ahead of Winch, trailing Cranelift only on float-heavy kernels and STREAM
  bandwidth — is recorded in [[backends.fact/native-benchmark-standing]] (code).

- 2026-06-14 rationale: giving every kept instruction a native entry address and
  threading control from one instruction's entry to the next was a TRANSITIONAL
  shape inherited while the native backend was still escaping the
  handler-threaded interpreter; straight-line code-to-code chaining (a function
  is one stream of native code falling through instruction to instruction, with
  bridge stubs confined to cold transitions) was always the intended end state,
  not per-instruction native-entry threading (sourced).

- 2026-05-13 (adc74515) statement: the backend ABI contract (`arch/abi.md`)
  formalizes the dynamic-register volatility classes — each dynamic bank is an
  abstract `[volatile][preserved][backend-internal scratch]` (FP omits scratch)
  whose lane counts `BackendConfig` publishes without naming physical
  registers; volatile lanes are caller-saved across local wasm-to-wasm calls
  and host the argument-lane prefix, preserved lanes are callee-saved and a
  function records a per-function preserved-clobber mask its prelude and every
  exit path save/restore; "cached local" is explicitly not an ABI class — the
  class is the abstract register's volatility, independent of what it currently
  holds (code).

- 2026-04-22 (e34767dd) pitfall: a function entry pointer on Thumb-2 carries
  the interworking bit (LSB=1), so the native instruction-dump region slicer
  must mask the low bit (`entry & !1`) before using the pointer as the region
  base; reading from the tagged pointer shifts every function's disassembly by
  one byte (code).

- 2026-03-20 (111163ac) rationale: the ABI guide states two preservation rules
  for platform bring-up — only the four fixed MachineIR registers (context, frame
  base, mem0 base, mem0 size) need free preservation across foreign-call
  boundaries (cached locals and transients are instead published to / reloaded
  from canonical frame slots by lowering at each boundary), and when a platform
  ABI is ambiguous a register not unquestionably callee-saved is treated as
  caller-saved — never backing a fixed MachineIR role with an ambiguous register
  (sourced).

- 2026-04-08 (b6424fed) rationale: the ABI guide's Backend Lowering Discipline
  section codifies that ordinary MachineIR lowering may touch only the registers
  named by the MIR operands, the four fixed roles when semantically required,
  backend-owned temps explicitly claimed from an ownership tracker, and foreign
  C-ABI arg/return regs only while lowering the boundary itself; a backend must
  never grab a convenient dynamic register as a temp or paper over a bug with
  blanket save/restore of a dynamic bank around a call (sourced).

- 2026-04-10 (8225d7a7) pitfall: the fallthrough-edge identity test must treat a
  MachineValue::ReservedReg(r) the same as Reg(r): a reserved cached-local lane
  whose register already equals the target block param is in place, so the edge
  is a no-op fallthrough; once edges began carrying ReservedReg, a predicate that
  only matched Reg silently classified those edges as non-fallthrough and emitted
  a needless move/branch (code).

## Moves

- 2026-03-24 (b4808682) replaced [[monolithic-per-arch-backends]]: each
  monolithic backend re-implemented the same per-function pipeline (prologue,
  block walk, edge parallel-move stubs, and the shared
  return-ok/stack-overflow/error/deferred-trap tail) so the orchestration was
  duplicated across every arch; factoring it into one CompilerCore plus a
  generic compile_function leaves only truly arch-specific behaviour
  (encoding, register mapping, prologue/epilogue, branch mechanics) on the
  ArchBackend trait (code).

- 2026-03-24 (b4808682) replaced [[per-arch-native-entry-typedefs]]: every
  architecture's native entry has the identical extern-C signature
  (NativeContext*, u64*) -> u32, so the three cfg-gated per-arch typedef +
  entry/return field pairs and their per-arch with_*_entry setters were
  redundant and collapse to a single NativeRootEntry/NativeCodePtr and one
  with_entry (code).

- 2026-04-07 (12aa736b) replaced [[inline-per-arch-dispatch]]: the module-build
  and eval paths each branched on the active backend with inline sf_arch_* cfgs
  in vm/build.rs and native_eval, leaking ISA gating into the shared pipeline;
  projecting every backend's per-function compile result into one
  CompiledArchEntry shape and routing compile/eval through arch::dispatch_*
  keeps every caller free of sf_arch_* cfgs and confines arch selection to
  arch/mod.rs (code).

- 2026-03-17 (33128cd7) dropped: per-call-boundary call-depth recursion
  limiter — the stack overflow check alone limits recursion, so the separate
  call-depth counter (incremented on every call, decremented on every return,
  capped at min(MAX_CALL_STACK_DEPTH, 300)) and its NativeContext.call_depth
  field are redundant and removed across every backend (code).
