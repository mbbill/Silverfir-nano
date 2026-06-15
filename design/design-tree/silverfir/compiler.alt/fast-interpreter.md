- Every Wasm function executes through a threaded interpreter rather than
  compiled native code; there is no machine-code compiler tier, only handler
  dispatch (with an optional per-block micro-JIT that still threads its emitted
  code into the same dispatch chain).

- The interpreter body is a fixed 32-byte instruction (handler pointer plus
  three 64-bit immediate slots) and each handler tail-calls its successor's
  handler through a `preserve_none` C trampoline rather than returning to a
  central dispatch loop.

- The top stack slots live in a window of hardware registers passed as
  `preserve_none` arguments across the whole tail-call chain, and the hottest
  locals are cached in further dedicated registers; values crossing the window
  edge emit explicit spill/fill against the frame.

- Wasm lowers to a neutral, backend-agnostic IR (`Vec<IrOp>`) that resolves
  stack management once — TOS depth variants, spill/fill insertion, and
  hot-vs-frame local mapping — and the interpreter, static-fusion, and micro-JIT
  backends all consume that single IR, each falling back to 1:1 base resolution
  for ops it cannot optimize.

## Facts

- 2026-02-14 (b7c626df) measurement: CoreMark on Apple M4 scored 6,251 — the
  fastest interpreter measured, ahead of wasm3 -O3 (~1.58x), WAMR, and wasmi,
  reaching ~67% of the single-pass Wasmtime Winch JIT and ~41% of full Cranelift
  while staying a pure interpreter (diff).

- 2026-03-01 (a05de669) measurement: across CoreMark/SHA-256/bzip2/LZ4 on Apple
  M4, fastest interpreter by 1.7-2.5x over wasm3 (geomean 2.0x) and reaching
  27-62% of optimizing Cranelift (geomean 38%); core interpreter ~230 KB
  stripped, ~2.9 MB with the full 1,500-pattern fusion set plus WASI (growth
  dominated by ~1,500 patterns x 4 depth variants = ~6,000 generated C
  functions) — full table in [[fast-interpreter.fact/draft-paper-benchmarks]]
  (diff).

- 2026-02-19 (21c17390) measurement: WASI snapshot (500 discovered patterns,
  window=5) showed the interpreter roughly matching wasm3 and beating wasmi and
  WAMR-fast, but trailing Winch ~1.5-2x and Cranelift ~4-8x — the standing that
  later motivated the native backend (diff).

- 2026-02-22 (4bb1de83) rationale: the hot dispatch loop is C, not Rust,
  because `musttail` and `preserve_none` have no stable-Rust equivalent;
  a trampoline bridges Rust to C and non-C handlers pay a wrapper call,
  acceptable because the hot path is mostly simple arithmetic (author).

- 2026-02-14 (a8528504) rationale: in the fast-interpreter era the host-call
  boundary was a bare function pointer `fn(&[Value], &mut [Value]) -> Result<(),
  WasmError>` registered at instantiation as a (module, name, fn) list, chosen
  over sf-core's `dyn ExternalFunction` trait object because the plain pointer is
  zero-alloc and no_std-friendly and the caller-supplied result buffer carries
  multi-value returns without a Vec; this is the shape the later native
  runtime-call boundary re-cut (sf-nano.md design doc) (author).

- 2026-02-22 (4bb1de83) statement: the publishable design paper "Beating a JIT
  Compiler with an Interpreter" — the full human-facing writeup of this engine,
  with the stay-stack-based argument, `clang -O3` godbolt proofs, the
  instruction-distribution and TOS spill/fill measurement tables, and the
  L0/L1/L2/TOS/preloading techniques — is the source behind the dispatch, fusion,
  and hot-local rationale facts; recovered in full in
  [[fast-interpreter.fact/interpreter-design-paper]] (author).

- 2026-06-14 rationale: the interpreter's governing objective is to minimize
  the number of dispatches — each tail-call hop is the dominant
  per-instruction cost on modern CPUs — so every technique (fusion, the TOS
  register window, hot-local caching) is a means to fewer or cheaper
  dispatches, not an end in itself; fusion is the primary dispatch-count
  reducer but only one possible means, and a design that reduces dispatch
  another way need not use it (author).

## Moves

- 2026-02-14 replaced [[fast]]: the fast compiled-handler interpreter is the
  part of -rs that continued: ported into the fresh -nano codebase as its
  starting point — same design, new implementation substrate (author).

- 2026-02-14 replaced [[ssa-compiler]]: the
  compiler-technology-for-an-interpreter line benchmarked poorly: XIR reached
  only ~90% of wasm3 while handler permutations exploded past 10k at 8 true
  registers, dispatch count — not memory access — remained the bottleneck, and
  the rotating TOS meant linear handler growth with no real register
  allocation; the author went back to the fast single-pass approach and
  carried it into -nano (author).

- 2026-03-07 (bc6c91c8) replaced by [[compiler]]: the micro-JIT was embedded
  inside the handler-threaded preserve_none fast interpreter and its generated
  code retained interpreter-shaped overhead (loop-boundary dispatch, repeated
  memory-metadata loads, hybrid JIT/handler transitions), and its dependence on
  preserve_none could not port to RISC-V/ARM32/MCU targets; the native backend
  instead owns a self-defined VM ABI entered through a global-asm trampoline
  that threads native-entry addresses directly, so it no longer behaves as one
  more kind of fast-interpreter handler (diff)
