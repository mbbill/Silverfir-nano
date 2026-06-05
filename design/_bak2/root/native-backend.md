- Functions compile whole to native code; execution enters the native ABI
  once per root invocation and stays native until final return or trap — no
  mixed-mode execution, no per-function fallback to any other executor.
- The pipeline is layered: Wasm decodes to a semantic IR; planning produces
  prepared LIR with explicit value residency; LIR lowers to a
  target-independent MachineIR; architecture backends translate MachineIR
  mechanically (registers, loads/stores, arithmetic, branches, ABI — nothing
  else).
- VM semantics (globals, memories, tables, runtime object layout) lower
  above MachineIR; below it only address computation, bounds checks, memory
  ops, control flow, and explicit runtime boundaries remain. A concept
  belongs below MachineIR only if it truly differs by ISA.
- Helper calls are explicit runtime boundaries that resume native execution
  directly; local calls and local indirect calls never leave the native ABI.
- Register caching of runtime state (memory views) is region-scoped with
  explicit invalidation contracts decided above the ISA layer — never
  assumed function-wide.
- The SSA in the pipeline is limited-residency and linear: every value is
  single-use, and the number of simultaneously-live values is bounded by
  construction.
- That form is what eliminates any register allocator: values live in
  bounded GP and FP transient banks plus canonical frame homes, with
  explicit slot traffic when they leave the banks — the SSA discipline is
  the allocation.
- Multiple architecture backends consume MachineIR: ARM64, ARMv7-A, x86_64,
  and the debug emulator.
- On 64-bit hosted targets, linear-memory bounds checking uses guard pages;
  explicit bounds checks serve targets without them.
- The OS boundary for executable memory is a four-function seam
  (allocate/free executable, begin/finish write): hosted targets implement
  it with mmap/guard pages, bare-metal targets with a static arena — porting
  means implementing these four functions.
