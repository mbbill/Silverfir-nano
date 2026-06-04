# JIT — emit native code

The current execution strategy: compile each Wasm function to native machine code
through a staged pipeline rather than interpreting it. This is the sole execution
backend; it replaced `fast-interpreter-with-fusion`.

The pipeline is **Wasm → Semantic IR (SIR) → SSA-IR → MachineIR → native**.
Portability becomes a per-arch backend behind the shared MachineIR (arm64,
x86_64, armv7a, riscv, thumb), and small footprint becomes a `no_std` JIT-only
binary still in the few-hundred-KB range, able to JIT on a 520 KB MCU.

How to *get* fusion on a JIT-capable target was its own fork
(`instruction-fusion-strategy/`): micro-JIT runtime fusion first, then a real
CFG+SSA pipeline. That sub-tree records the path from "the JIT *is* the fusion"
to the staged compiler that exists today.

## In practice

Must:
- Each Wasm function must be lowered through the four-stage pipeline
  Wasm → SIR → SSA-IR → MachineIR → native (entry: `vm/build.rs`
  `ensure_module_compiled()`).
- Every supported architecture must be a backend behind the shared MachineIR; a
  new ISA is a new backend, not a middle-end rewrite, and must lower every
  MachineIR op natively or via fallback.
- The JIT-only binary must stay `no_std` with zero runtime dependencies and small
  enough to run on a ~520 KB-SRAM MCU.
- The native backend must clear the spectest gate
  (root.all/correctness-validation.md) on each target before that target ships.

Must not:
- Must not retain the interpreter + fusion build system as a live hot path; the
  JIT is the only execution backend (`feature = "micro-jit"`).
- Must not require full general-purpose register allocation in the backend: only
  short-lived transients participate; canonical locals and deep stack / call
  payloads already have fixed frame-slot homes.

## Ground rules — instruction-fusion-strategy
Must:
- Produce, for a run of consecutive Wasm operations, native code that executes
  the run without re-entering per-opcode dispatch between the fused members.
- Reuse the established stack-machine value homes (the TOS register window plus
  the L0/L1/L2 hot-local cache) so fusion does not require a separate value model.
- Validate the chosen strategy against the spec testsuite and against the base
  interpreter as a differential oracle before it becomes a primary path.

Must not:
- Maintain more than one live fusion source for the primary execution path at
  the same time.
- Ship a fusion strategy whose handler set has not passed spectest.

## Ground rules — native-backend-structure
Must:
- Emit native code for the hot path; the base handler interpreter exists only as
  a fallback and differential oracle, not the primary execution path.
- Validate end-to-end against the spec testsuite on the structure's primary
  target before it is promoted.
- Lower every supported Wasm operation to native code through the backend's own
  value model — no value may require re-entry into per-opcode interpreter dispatch
  to execute on the hot path.

Must not:
- Keep a superseded backend structure wired into any build configuration once it
  has been replaced; a retired structure must be removed, not retained behind a
  flag.
