- Termination is signalled by impl_term returning NULL: the op_term wrapper
  checks for NULL to end interpretation at the outermost frame and otherwise
  tail-calls the returned instruction.

- Traps are a distinct op_trap wrapper that calls a Rust impl to set the trap
  state and then returns, breaking the tail-call chain separately from the term
  path.

- The fast-path wrapper contract forbids handlers from returning NULL or a
  terminal-instruction pointer; terminal points are encoded by explicitly placing
  an op_term instruction in the stream.

- The trap cause is carried out-of-band as a shared Option<WasmError> the runtime
  reads after the trampoline returns.

## Facts

- 2025-08-13 (1ff2eb11) rationale: encoding termination as a null-fallthrough
  checked in every wrapper put a branch on the hot tail-chain; replacing it with a
  non-null op_term sentinel instruction whose handler simply returns lets the
  wrappers tail-call unconditionally, removing the per-instruction null check
  (code).

- 2025-08-16 (b3260998) rationale: the per-frame trap slot is a plain
  Option<WasmError> on the single-owner Context, not Rc<RefCell<Option<WasmError>>>;
  the Context is reached only through a raw *mut pointer inside handlers, so the
  interior mutability/refcount/allocation of Rc<RefCell> bought nothing and was
  removed for an allocation-free trap-signal path (code).

## Moves

- 2025-08-15 (9a585649) replaced [[boolean-trap-flag]]: a boolean flag could only
  say a trap occurred, forcing the runtime to fabricate a generic WasmError; a
  shared Option<WasmError> carries the actual trap cause from the handler that
  raised it out to the caller (code).

- 2025-12-07 (f10f96fa) replaced by [[extern-helper-trap-path]]: the old scheme
  required impl_term to return NULL (checked on every term in the wrapper) and
  routed traps through a separate op_trap, but under the musttail/preserve_none
  dispatch a handler returning NULL is undefined behaviour; a single static
  TERM_INST that every handler can return — with the trap error stashed in
  ctx.error and inspected after the trampoline returns — unifies normal return and
  trap into one always-valid-pointer path, mirroring the XIR backend's TERM_INST
  design (code).
