- Normal return and trap share one always-valid-pointer path: every handler can
  return a single static terminal-instruction sentinel, no handler ever returns
  NULL (which is undefined behaviour under the musttail/preserve_none dispatch),
  and the trap cause is carried out-of-band in the context and converted to an
  error once after the trampoline returns.

- A C trap handler stays a leaf function: it writes a static message pointer into
  the context and returns the cached terminal-instruction pointer with no function
  calls, rather than calling a Rust helper that would force a prologue/epilogue on
  the whole handler; the deferred message is converted to the real error after the
  trampoline returns.

- The first trap wins: the trap slot is only written when still unset, and a
  secondary error raised while unwinding does not overwrite the root cause.

## Facts

- 2026-01-14 (60c4375a) measurement: an experiment replaced the Rust-helper trap
  call with an inline store of a C-side trap message plus a return of the global
  terminal sentinel, eliminating the prologue/epilogue (verified at the assembly
  level); on Apple Silicon CoreMark (5 runs each) the inlined version was ~1.5%
  SLOWER (avg 3494 vs 3545), so it was reverted — prologue/epilogue is cheap on
  out-of-order ARM cores, CoreMark never takes the trap path so the inlined trap
  code is dead icache weight, and the real hotspots are elsewhere (indirect-branch
  misprediction in if_/br_if/br_table). Lesson: do not inline the C trap path for
  performance on this microarchitecture (code).

- 2026-02-06 (ade98d4a) pitfall: zeroing a callee's locals with __builtin_memset
  let LLVM lower the loop to a bl _bzero library call, which forces a prologue/
  epilogue on the entire call handler — silently defeating the leaf-handler design
  the deferred-trap path exists to preserve; the fix replaces the memset with an
  explicit volatile-store loop, the lesson being that any construct the optimizer
  can lower to a libcall (memset/memcpy/bzero) reintroduces the prologue on a
  handler meant to stay leaf (code).

- 2025-08-14 (04146b8b) rationale: a trapping condition takes the instruction's
  otherwise-unused alt branch edge to an appended trap-terminal instruction rather
  than a dedicated trap return channel, so a trap is just a branch and the
  tail-chained handler loop needs no extra control path; callers unwind by checking
  the shared trap state after a callee returns (code).

- 2025-08-18 (0ec2eba9) pitfall: errors returned from external/host calls must be
  surfaced through the trap slot rather than collapsed to a bare status code,
  otherwise the real WasmError is constructed at the call boundary but never raised
  (code).

- 2026-06-14 rationale: the inline c_trap (static-message store, no function
  call) is adopted to keep every trapping handler a leaf, which is what lets the
  preserve_none + next-handler-preloading composition hold across the whole
  chain; the form is accepted as neutral on ARM, not adopted because it was
  re-measured faster (sourced).

## Moves

- 2026-02-06 (61c33372) replaced [[extern-helper-trap-path]]: a C handler trapped
  by calling the Rust fast_c_trap(ctx,msg) helper (and terminated by calling
  fast_term_inst()), and every such call forced a prologue/epilogue on the whole
  handler even though the trap path almost never runs; storing the trap path on the
  Context instead — c_trap writes a static message pointer into ctx.trap_message and
  returns a TERM_INST pointer cached in ctx.term_inst, with zero function calls —
  keeps trapping handlers as leaf functions, and the deferred message is converted
  to a WasmError once after run_trampoline returns (code).
