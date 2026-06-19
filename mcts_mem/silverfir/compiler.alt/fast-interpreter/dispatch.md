- Each instruction is a fixed 4x64-bit (32-byte) word: one handler pointer plus
  three immediate slots; every handler tail-calls its successor through a
  `preserve_none` C trampoline, and each handler also preloads the
  handler-after-next; load-to-use latency of the dispatch pointer spreads
  across two invocations.

- The top stack slots are mapped to a window of hardware registers (four,
  passed as `preserve_none` arguments) that persist across the whole tail-call
  chain with zero memory traffic for in-window ops; Wasm stack height is
  statically known the compiler emits one handler variant per TOS depth, and
  ops crossing the window edge emit explicit spill/fill.

- The interpreter keeps the Wasm stack-machine IR rather than converting locals
  to SSA virtual registers; fused operand locations are then compile-time
  stack offsets; concatenating N handler bodies lets the C compiler eliminate
  intermediate loads/stores automatically.

- Each call frame is laid out as params, then locals, then three metadata slots
  (return_pc, saved_fp, saved_module), then the operand stack; the compiler
  tracks only stack height plus a spill-depth invariant (at most
  TOS_REGISTER_COUNT values live in registers) rather than per-slot types,
  emitting spill/fill at control-flow boundaries and calls.

## Facts

- 2026-02-22 (4bb1de83) rationale: staying stack-based (vs the wasm3/WAMR
  register-machine conversion) was the central thesis — virtual registers live
  in memory with no liveness-aware allocator, and register-machine operands are
  runtime indices that create aliasing barriers the compiler cannot see through,
  so register-machine fusion would need hand-written per-pattern handlers; the
  stack machine makes fusion mechanical — full argument and the godbolt proof
  in [[dispatch.fact/stay-stack-based]] (sourced).

- 2026-02-22 (4bb1de83) rationale: each instruction is fixed-wide (32 bytes)
  because fusion fills the slots — any candidate whose immediates exceed three
  slots is rejected during discovery, so every accepted fused pattern fits, and
  32 bytes aligns two instructions to one 64-byte cache line with no straddling
  (sourced).

- 2026-02-22 (4bb1de83) rationale: tail-call dispatch (each handler a separate
  function tail-calling the next) was chosen over switch and computed-goto
  because per-handler BTB entries give >95% indirect-branch prediction and
  independent functions let the C compiler optimize each handler in isolation,
  made workable by `musttail` + `preserve_none` + leaf handlers so every handler
  emits zero prologue/epilogue (sourced).

- 2026-02-22 (4bb1de83) rationale: after fusion/TOS-caching/hot-locals shrink
  handler bodies to ~1 instruction, the load-to-use latency of fetching the next
  handler pointer dominates, so each handler preloads the pointer for the
  instruction-after-next; a three-tier guard classification lets always-linear
  handlers prove the guard always-true and drop it entirely (sourced).

- 2026-02-22 (4bb1de83) measurement: CoreMark with fusion (189M dispatches) —
  TOS spill/fill overhead totals 3.10% of dispatches, confirming LLVM keeps hot
  values in a shallow stack region and the 4-slot window is rarely exceeded
  (sourced).

- 2026-02-22 (4bb1de83) pitfall: the dispatch tail call must be a hard
  guarantee, not best-effort TCO — an ordinary optimized tail call may compile to
  `call` rather than `jmp`, and the handler chain then overflows the native stack
  instantly; the chain requires `clang`'s `musttail` or GCC 15's `gnu::musttail`,
  which either emit a real tail call or fail at compile time, not plain TCO —
  from the design paper docs/INTERPRETER_DESIGN.md (deleted 78b1f6d6, content at
  4bb1de83) (sourced).

- 2026-03-01 (a05de669) rationale: the TOS window's contribution is replacing
  the runtime FSM of classic stack-caching (Ertl 1995, 1-2 TOS registers as a
  3-state FSM with a 2D dispatch table) with deterministic compile-time depth
  selection — no table lookup, no state update — scaled from 1-2 to four TOS
  registers, made feasible by `preserve_none` (sourced).

- 2026-03-01 (a05de669) rationale: mapping a register-machine's virtual
  registers to hardware registers (rather than staying stack-based) was rejected
  on two counts beyond the in-memory/aliasing argument: it would need a load-time
  register-allocation pass (liveness, interference graph, physical assignment)
  whose cost approaches a baseline JIT's allocator, and under the fixed
  tail-call/`preserve_none` ABI — where each handler receives VM state in a fixed
  argument-register order — per-function register assignments would force register
  permutation at every handler boundary, inflating code size; the stack machine
  sidesteps both because operand locations are compile-time constants needing no
  allocation infrastructure (sourced).

- 2026-03-03 (6fda0037) statement: the depth-variant scheme is four variants
  D1-D4 cycling modulo four as stack depth grows (depths 0 and 1 both map to D1,
  5 wraps back to D1), reusing one physical register set cyclically rather than
  one window state per absolute depth — corrected from an earlier D0-D4 framing
  (code).

- 2026-02-22 (4bb1de83) limitation: the TOS register window holds operands in
  64-bit integer GPRs, so float values are bit-cast through integer registers
  across the handler chain rather than kept FP-resident; the author flags this
  as possibly non-optimal on some platforms and notes float-heavy workloads were
  not extensively profiled — the later micro-JIT re-adds GPR-vs-FP slot-location
  tracking ([[micro-jit]]) precisely to keep floats FP-resident (sourced).

## Moves

- 2026-03-07 replaced by [[compiler]]: the interpreter's preserve_none
  handler-threaded model and its embedded micro-JIT retained interpreter-shaped
  overhead and could not port to RISC-V/ARM32/MCU targets, so a native
  code-generation backend owning its own VM ABI replaced the whole interpreter
  execution era (code).
