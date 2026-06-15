- A WASM call is executed by recursively re-entering the evaluator on the native
  call stack, returning the callee's result stack to the caller's handler
  (`eval_internal`).

- Each call frame is a fresh context holding that frame's own register file; call
  depth is a counter threaded through the recursive eval and checked against a
  fixed limit.

- A handler that performs an internal call blocks until the recursive eval
  returns, then copies the result words into the caller's result registers and
  advances to the next instruction.

## Facts

- 2025-10-12 (2cee86a4) rationale: a call handler executes its callee by
  recursively invoking the runtime's eval on the callee function instance rather
  than managing its own explicit interpreter frames; this leverages the host's
  native call stack and is simpler than building per-call frame management, with
  arguments and results marshalled through the caller's backing register file
  (diff).

## Moves

- 2025-10-16 (a3cc8801) replaced by [[call-return]]: recursive eval grew the
  native host stack one frame per WASM call; an explicit heap-allocated call stack
  in Ctx that the call handler pushes and the ret handler pops keeps WASM calls
  and returns off the native stack (diff).
