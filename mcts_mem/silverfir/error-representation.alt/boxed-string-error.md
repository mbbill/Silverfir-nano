- `WasmError` is a heap pointer wrapper: an 8-byte struct holding
  `Box<WasmErrorInner>` over a seven-variant inner enum, where every non-Exit
  variant carries an owned `String` message; a `Result` returned through the
  hot success path carries only one pointer's worth of error.

- Error messages are built by lazy-formatting closures (`wasm_error!` macro /
  `*_fmt` constructors) that run `alloc::format!` only when the cold error
  constructor is actually reached, letting a failing site interpolate the
  offending value (section id, version, index) into the message. Every error
  constructor is marked `#[cold] #[inline(never)]` and the macro defers
  `format!()` behind a `FnOnce` closure; message formatting runs only inside
  the out-of-line cold constructor and never enlarges the caller's stack frame
  on the success path.

- `WasmError` is not Copy: it implements Clone by deep-copying the boxed inner
  message, and its constructors are non-const runtime functions.

## Moves

- 2025-10-13 (04cd73c2) replaced [[flat-inline-error]]: the flat enum's message
  and backtrace inflated every Result-returning frame's stack footprint on the
  hot success path; boxing the payload shrinks WasmError to one pointer (code).

- 2026-04-10 (6d716c87) replaced by [[error-representation]]: the heap-boxed
  String-carrying error forced an allocation on every error path and could not
  be Copy or const-constructed, so error construction pulled in alloc::format
  and a Box per error even on the cold trap/validation paths; making WasmError a
  Copy enum of &'static str messages removes all allocation from the error path
  and lets errors be built in const fns, at the cost of dropping dynamic
  interpolation (offending values no longer appear in the message text) (code)
