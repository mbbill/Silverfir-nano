- Each native backend is a self-contained monolithic compiler that drives the
  whole per-function emission itself: it walks blocks, emits each instruction
  and terminator, lays out edge parallel-move stubs, and emits its own
  return-ok / stack-overflow / error / deferred-trap tail regions, holding all
  label, fixup and patch bookkeeping in its own struct (`FunctionCompiler`).

- `compile_module` is an arch-specific entry that produces arch-typed compiled
  entries; the same per-function pipeline shape is copy-duplicated across the
  ARM64, x86_64 and ARMv7-A backends with no shared driver.

## Facts

- 2026-03-24 (215d0456) statement: the x86_64 backend carried its own
  per-function pipeline (block walk, instruction/terminator emission, edge
  stubs, return-ok/stack-overflow/error/deferred-trap tail), its own
  `compile_module`, and dispatched through a backend-specific `eval` rather
  than a shared one — collapsed into the shared CompilerCore form by the same
  re-decision (diff).

- 2026-03-28 (dfdac079) statement: the ARMv7-A backend additionally inlined a
  ~56-byte trap-handler sequence at every trapping site (rather than branching
  to a shared stub) and used a backend-private `Arm32TextEmitter` and helper
  module rather than the shared TextEmitter and scratch-pool infrastructure —
  collapsed into the shared form by the same re-decision (diff).

## Moves

- 2026-03-07 (bc6c91c8) replaced [[micro-jit-backend]]: the micro-JIT was
  embedded inside the handler-threaded preserve_none fast interpreter and its
  generated code retained interpreter-shaped overhead (loop-boundary dispatch,
  repeated memory-metadata loads, hybrid JIT/handler transitions), and its
  dependence on preserve_none could not port to RISC-V/ARM32/MCU targets; the
  native backend instead owns a self-defined VM ABI entered through a
  global-asm trampoline that threads native-entry addresses directly, so it no
  longer behaves as one more kind of fast-interpreter handler (diff).

- 2026-03-24 (b4808682) replaced by [[backends]]: each monolithic backend
  re-implemented the same per-function pipeline (prologue, block walk, edge
  parallel-move stubs, and the shared return-ok/stack-overflow/error/deferred-trap
  tail) so the orchestration was duplicated across every arch; factoring it
  into one CompilerCore plus a generic compile_function leaves only truly
  arch-specific behaviour (encoding, register mapping, prologue/epilogue,
  branch mechanics) on the ArchBackend trait (diff).
