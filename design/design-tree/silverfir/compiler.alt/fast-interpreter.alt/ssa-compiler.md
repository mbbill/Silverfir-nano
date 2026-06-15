- The default backend compiles each function through a multi-stage SSA
  pipeline — WebAssembly to SSA IR, SSA IR to a lowered register IR, then to an
  executable bytecode (XIR) — before interpreting that bytecode.

- The pipeline is split at the virtual-to-physical register boundary into two
  separate IRs: an SSA IR with unbounded virtual values for analysis, and an
  LIR with physical registers and explicit spill slots for execution.

- Register allocation happens once, in the middle stage; the backend that
  emits executable code is a thin translation with no allocation logic of its
  own.

- Compiled XIR is cached per function and built lazily on first call (or
  eagerly at instantiation when enabled); a function is compiled at most once.

## Facts

- 2025-09-17 (aed2ff42) rationale: the SSA backend is specified as a
  clean-slate, stand-alone interpreter that deliberately inherits no layout or
  constraints from the fast (stack-window) backend, integrating only with
  interpreter-agnostic runtime components (Store, ModuleInst, MemInst,
  TableInst, Globals); its performance thesis is to minimize the two
  interpreter costs at once — fewer dispatches by selecting fused
  superinstructions from SSA expression trees (tile covering of depth <=2
  chosen by Sethi-Ullman numbering, with store-root tiles computing their whole
  RHS in one dispatch), and less per-dispatch work via tiny direct-threaded
  handlers (diff).

- 2026-06-14 statement: xir and fast share the same handler-table dispatch (xir
  the later, more-refined form), so xir died on real bottlenecks rather than
  dispatch style — register permutation caps xir's usable registers while fast's
  fixed O(n) TOS-mapping uses every register with no allocator; xir's SSA-edge /
  ParCopy shuffles tax an interpreter (each is a dispatch) but are ~free on a JIT;
  and the full compiler pipeline is far harder to get correct than the trivial
  TOS + local-cache stack machine the microJIT inherited — full diagnosis in
  [[fast-interpreter.fact/why-xir-died]] (author).

## Moves

- 2026-02-14 replaced by [[fast-interpreter]]: the
  compiler-technology-for-an-interpreter line benchmarked poorly: XIR reached
  only ~90% of wasm3 while handler permutations exploded past 10k at 8 true
  registers, dispatch count — not memory access — remained the bottleneck, and
  the rotating TOS meant linear handler growth with no real register
  allocation; the author went back to the fast single-pass approach and
  carried it into -nano (author).
