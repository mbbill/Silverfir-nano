- A trap is reported by setting a shared Rc<Cell<bool>> flag; the runtime, seeing
  it set, returns a generic WasmError::Trap with no specific cause.

## Moves

- 2025-08-15 (9a585649) replaced by [[null-return-termination-separate-trap]]: a
  boolean flag could only say a trap occurred, forcing the runtime to fabricate a
  generic WasmError; a shared Option<WasmError> carries the actual trap cause from
  the handler that raised it out to the caller (code).
