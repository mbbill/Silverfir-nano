- A C handler signals a trap by calling the extern Rust helper fast_c_trap(ctx,
  message), which constructs the WasmError immediately.

- A C handler obtains the terminal instruction by calling the extern Rust helper
  fast_term_inst().

- Call-depth exhaustion in the C call handler is reported by calling the extern
  Rust helper fast_call_depth_exceeded(ctx).

- The Context carries no trap-message or cached-terminal-instruction fields; trap
  construction and terminal lookup cross the FFI boundary into Rust.

## Moves

- 2025-12-07 (f10f96fa) replaced [[null-return-termination-separate-trap]]: the old
  scheme required impl_term to return NULL (checked on every term in the wrapper)
  and routed traps through a separate op_trap, but under the musttail/preserve_none
  dispatch a handler returning NULL is undefined behaviour; a single static
  TERM_INST that every handler can return — with the trap error stashed in
  ctx.error and inspected after the trampoline returns — unifies normal return and
  trap into one always-valid-pointer path, mirroring the XIR backend's TERM_INST
  design (code).

- 2026-02-06 (61c33372) replaced by [[trap-signal]]: a C handler trapped by calling
  the Rust fast_c_trap(ctx,msg) helper (and terminated by calling fast_term_inst()),
  and every such call forced a prologue/epilogue on the whole handler even though
  the trap path almost never runs; storing the trap path on the Context instead —
  c_trap writes a static message pointer into ctx.trap_message and returns a
  TERM_INST pointer cached in ctx.term_inst, with zero function calls — keeps
  trapping handlers as leaf functions, and the deferred message is converted to a
  WasmError once after run_trampoline returns (code).
