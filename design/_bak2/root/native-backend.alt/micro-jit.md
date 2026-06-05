- A micro-assembler, not a compiler: the TOS window and hot-local registers
  already fix the physical register assignment, and depth-variant selection
  determines each instruction's registers — so machine code is emitted
  directly (1–2 instructions per op) with no SSA, no register allocation,
  microsecond assembly time, and no external dependency.
- The micro-JIT is the fusion system, not a layer above it: it replaces the
  builder's final stage. Consecutive JIT-able ops group into a block whose
  emitted-code address becomes the handler pointer; unsupported ops keep
  their pre-compiled C handlers, and the dispatch chain interleaves both
  through the same `preserve_none` register contract.
- Static fusion's limits do not apply: no 3-immediate encoding budget, no
  finite pattern set, no workload-dependent discovery step.
- JIT codegen (Rust) and the C `SEM_*` macros are independent encodings of
  the same semantics; `SEM_*` remains the source of truth for static fusion
  on non-JIT platforms.
- Codegen never hardcodes registers, struct offsets, or encodings: platform
  details live behind register names, named offsets with `offset_of!`
  compile-time guards, and emit traits — a new target is a new backend, not
  new codegen.
- Memory ops emit inline bounds checks with an inlined trap path.
- micro-jit and static fusion are feature-gated alternatives sharing one
  builder pipeline; on JIT-capable platforms the JIT replaces static fusion
  entirely.
