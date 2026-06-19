- Every trap funnels through one shared static terminal instruction: a handler
  that traps sets the error slot on the context and returns a pointer to that
  terminal instruction; the generic tail-chain wrapper always dereferences a
  valid next-pc and never needs a null check, and the terminal handler unwinds back to
  the entry trampoline where the pending error is observed.

- A C (pure-computation) handler cannot construct the Rust error value; a
  trapping C handler instead stores a static message pointer on the context and returns
  the cached terminal instruction inline, staying a leaf function, and the Rust side
  converts the deferred message to the error value after the trampoline exits.

## Facts

- 2025-10-11 (d9281a5a) statement: the trap target is a single static read-only
  instruction whose handler is the terminal handler; handlers set the error and
  return a pointer to it, so the tail-chain wrapper always dereferences a valid
  next-pc and never needs a null check (code).

- 2025-10-24 (f5a8bd12) rationale: pure-computation handlers are written in C and
  cannot construct a Rust error value; rather than mirror the error model in C,
  every C-side trap path delegated to a single Rust shim, so the C handlers only
  know a message string and the error construction stays on the Rust side (code).

- 2025-09-19 (0818e28c) statement: the backend's error discipline was established
  with the first handlers — a trap sets the error on the context and returns
  without tail-chaining, unwinding the trampoline frame rather than continuing the
  chain (code).

## Moves

- 2025-10-11 (d9281a5a) replaced [[dedicated-trap-opcode]]: a trapping handler
  returned a null next-pc that the generic tail-chain wrapper would have
  dereferenced; returning a shared terminal instruction lets every handler trap
  through one path with no null check in the C wrappers (code).

- 2026-02-12 (7f0e670c) replaced [[extern-rust-c-trap]]: the extern xir_c_trap
  forced every trapping C handler to make a cross-language call (constructing the
  WasmError in Rust), so it could not stay a leaf function; storing a static
  message pointer in ctx and returning a cached term_inst inline keeps C handlers
  leaf (no prologue/epilogue, no FFI call, no heap alloc on the hot path), with
  the Rust side converting the deferred message to a WasmError after the
  trampoline exits (code).
