- The single IR layer between prepared SSA-IR and ISA emission is a
  target-independent machine IR modeling only generic registers, explicit
  addresses, loads/stores, arithmetic, control flow, and explicit call/helper
  boundaries; it carries no context, hot-local, TOS-lane, frame, or spill
  concepts (`MachineReg`, `MachineAddr`).

- The IR splits into a code half (generic registers/addresses/widths, straight-line
  instructions, block/edge/terminator CFG, per-function/module containers plus a
  sidecar constant pool) and a non-code half — a shared runtime contract — with a
  structural validator over the code half (`MachineProgram::validate`).

- The runtime-side contract the code executes against — pinned input registers
  (runtime base, frame base, mem0_base, mem0_size), the per-function param/return
  frame regions, typed helper signatures, and runtime layout offsets for address
  derivation — is separate non-code metadata shared by every ISA backend.

- MachineIR registers, moves, selects, and block parameters carry a storage class
  (GpWord, GpI64, Fp32, Fp64, and V128) that distinguishes a pointer-width value
  from a true 64-bit GP value needing register-pair handling on 32-bit GP targets,
  and keeps FP/vector values in the right bank (`MachineStorageType`).

- Comparison operations are fused into branch conditions: the IR carries fused
  integer/float compare-and-branch terminators rather than a separate compare feeding
  a value branch (`MachineBranchCond`).

- v128 is held in the FP/vector register bank during computation while a frame slot
  stores it as a u64 raw handle, and most SIMD opcodes flow through a small set of
  generic opcode-carrying ops that each backend decodes late; the SIMD IR variants
  are compile-time-gated behind a SIMD cfg, absent entirely from a non-SIMD build
  (`sf_has_simd`).

- The runtime object native code reads is a repr(C) context with stable
  ABI-visible field offsets pinning mem0_base/mem0_size/stack_end and
  caching base+len storage views for memories, tables, and globals; compiled code
  reaches storage through these fixed offsets without calling into the store on the
  hot path (`NativeContext`).

- MachineIR is released once native code is emitted: the compiled module holds it
  only optionally, dropping it after emission, since only native emission
  consumes it.

## Facts

- 2026-03-18 (3778de1c) rationale: the storage type replaces an earlier optional
  float-width on block params and an implicit is-fp test on moves/selects — one
  storage class drives bank-consistency validation and is the input the 32-bit
  legalization reads to decide which ops split into hi/lo pairs, so an untyped
  register at a save-before-call point makes the legalizer unable to classify it
  (code).

- 2026-03-13 (c511b2ee) constraint: the native ABI and shared MachineIR assume a
  64-bit machine model — pointer-like fields and cached lengths are read through
  U64 accesses, so 32-bit targets need an explicit pointer-width abstraction rather
  than reusing this layout (code).

- 2026-04-22 (9ee2e65a) rationale: the SIMD instruction variants are
  compile-time-gated rather than always present with a runtime
  "emulator received SIMD without sf_has_simd" rejection branch, so a non-SIMD
  build elides them everywhere and the unreachable runtime error is removed,
  keeping the enum and its visitors free of dead SIMD arms (code).

- 2026-03-09 (c607440c) rationale: a deeper native stack (LIR -> Entry IR -> Role
  IR -> ISA) was explicitly weighed and rejected — the engine is intentionally
  small and embedded-friendly and the whole point of the TOS/local-cache model is
  to avoid a traditional register allocator and a large optimizer, so adding
  multiple native-only semantic layers would start to recreate a general
  optimizing JIT; the chosen shape is exactly one machine IR after LIR (all
  semantic restructuring at/before LIR, all real optimization in LIR->machine IR,
  ISA lowering intentionally dumb), and an embedder wanting a large optimizing JIT
  is expected to use something like Cranelift instead (sourced).

- 2026-03-11 (c4102007) statement: the native-backend design rules fix the central
  boundary — keep VM semantics above the machine IR and ISA details below it, with
  one ISA test gating every concept (a concept belongs below the IR only if ARM64
  and x86_64 would truly implement it differently rather than just encoding it
  differently); Wasm object-model knowledge (global N, memory 0, table/global
  layout, memory.grow's view-invalidation policy) is lowered above the IR into
  explicit address computation, checks, loads and stores (sourced).

## Moves

- 2026-03-11 (0282f727) replaced [[native-ir]]: the old NativeIR carried VM register kinds (Ctx/Fp/Hot/Tos/Tmp) and LIR planning-provenance storage (Frame/Spill slots) directly in its operands, leaking VM meaning and lowering history past the backend boundary; the new IR uses generic MachineReg(u16) plus explicit MachineAddr and moves all runtime layout, pinned-input meaning, and call-link contract into separate ABI/contract metadata, so the ISA backend sees a real machine IR with no context, hot-local, TOS-lane, frame, or spill concepts (code).

- 2026-04-09 (c329abab) replaced [[retain-machine-ir-at-runtime]]: MachineIR is only consumed during native code emission (and by the emulator backend, which executes it directly), so retaining it on the runtime module wastes memory after compilation; making it optional and dropping it once native code is emitted frees that footprint on memory-constrained targets (code).
