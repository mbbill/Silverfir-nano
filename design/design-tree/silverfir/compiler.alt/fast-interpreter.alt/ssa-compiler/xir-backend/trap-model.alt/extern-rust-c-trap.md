- A C handler reporting a trap calls the exported Rust function `xir_c_trap`,
  passing a null-terminated message; that call constructs the error value, stores
  it in the context's error cell, and returns the terminator instruction to unwind
  the tail-call chain.

## Moves

- 2026-02-12 (7f0e670c) replaced by [[trap-model]]: the extern xir_c_trap forced
  every trapping C handler to make a cross-language call (constructing the
  WasmError in Rust), so it could not stay a leaf function; storing a static
  message pointer in ctx and returning a cached term_inst inline keeps C handlers
  leaf (no prologue/epilogue, no FFI call, no heap alloc on the hot path), with the
  Rust side converting the deferred message to a WasmError after the trampoline
  exits (diff).
