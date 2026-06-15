- The fast backend pre-builds each function into a fixed-size instruction
  stream once, then dispatches that stream via tail-chained handlers rather
  than re-decoding the original bytecode.

- A small set of top-of-stack values is kept in handler-local slots, with the
  handler variant chosen per opcode by which top-of-stack slots are occupied;
  the hottest operands stay out of the memory stack.

- Adjacent opcodes are fused into super-instructions at build time, with a
  fusion-disabled mode that emits one handler per opcode for profiling the raw
  stream.

- The dispatch handlers and their per-state variants are produced from a
  declarative specification by a build-time generator, not hand-written per
  permutation.

- Calls within a module are precompiled: direct and recursive call targets are
  encoded into the instruction, avoiding a runtime lookup on the hot call path.

- The handler tail-chain is implemented as a C trampoline that the per-opcode
  handler bodies inline into under cross-language LTO; the C component and its
  build/link flags are load-bearing to the design, not optional glue.

## Facts

- 2025-08-09 (65015cb5) rationale: the fast backend exists to maximize
  single-threaded interpreter speed on stable Rust without any JIT or
  native-code path; predecoding into a compact per-function FastIR removes the
  per-opcode LEB/immediate decoding the in-place interpreter repeats on every
  execution (stated in the accompanying FAST_INTERPRETER_PLAN.md) (diff).

- 2025-08-11 (4bab9be0) dependency: the threaded form pulls the first C code
  and the first build dependency (the cc crate) into sf-core: a build script
  compiles the C trampoline translation unit with -O3 -flto and sibling-call
  optimization, and link-time -flto is required so the Rust impl bodies inline
  into the C tail-jump wrappers — the register-residency win depends on that
  cross-language LTO, so the C component and its build/link flags are not
  optional glue but load-bearing to the design (diff).

- 2025-08-13 (4315907e) rationale: the fast backend's tail-chaining depends on
  cross-language (Rust+C) LTO so the Rust impl_* bodies inline into the C op_*
  wrappers and each wrapper collapses to a single tail jump; the build pins
  -Clinker-plugin-lto plus fat LTO / codegen-units=1 / panic=abort and treats
  disabling cross-language LTO as breaking the core performance assumption, not
  a tunable (diff).

- 2025-08-14 (53e8efd5) statement: function evaluation switched from running
  the in-place interpreter to running the fast interpreter unconditionally on
  the hot path, before any per-process backend switch existed; the in-place
  interpreter was retained but no longer reached (diff).

- 2025-08-09 (f8283349) statement: although the fast backend is the library
  default, the spec-test and WASI-test harnesses force the baseline interpreter
  on, evidence that the fast path was not yet trusted for full correctness at
  this stage; the fast path traps rather than silently miscomputing on any
  opcode it does not yet support (diff).

- 2025-08-15 (d3e9a056) rationale: a null funcref/externref is represented on the
  operand stack by the sentinel usize::MAX rather than 0, because a non-null
  reference is a raw function/table index whose value 0 is legitimate; ref.is_null
  tests against usize::MAX accordingly (diff).

## Moves

- 2026-02-14 replaced by [[fast-interpreter]]: the fast compiled-handler
  interpreter is the part of -rs that continued: ported into the fresh -nano
  codebase as its starting point — same design, new implementation substrate
  (author).
