- A compiled function has two entry points: a public C-ABI entry at which the
  prologue runs, and an internal entry located past the prologue that direct
  local compiled-to-compiled (SF->SF) calls target by skipping the prologue
  (`internal_entry_label`).

- A function's success Return is lowered inline; its error path is a
  C-epilogue-free body-local tail that propagates the trap code in C_RET0; a
  callee entered at the internal entry (which never ran the C prologue) never
  runs a C epilogue it has no matching prologue for (`body_local_error_label`).

## Facts

- 2026-04-07 (b59caeff) statement: arm64 defers each direct call's patchable
  callee-address word into an end-of-body per-function literal pool (flushed
  after edge stubs, before the body-local error tail, within pc-relative load
  range of every call site) instead of emitting it inline after the BLR, which
  lets the direct-call lowering elide its trailing 'b continuation' when the
  next emitted block is the continuation (code).

- 2026-04-07 (b59caeff) measurement: measured call-boilerplate overhead is the
  kill-fact behind the entry split — full data in
  [[entry-and-tails.fact/call-boilerplate-overhead]] (code).

## Moves

- 2026-04-07 (b59caeff) replaced [[single-entry-shared-tails]]: direct
  compiled-to-compiled (SF-to-SF) local calls enter a callee at an internal
  entry point that does not run the C-ABI prologue, so the old function-wide
  return-error tail (which ran the C epilogue and assumed the prologue had
  executed) was wrong for locally-entered callees; the entry is split into a
  public C-ABI stub that bl's the internal entry and a body whose success
  Return is lowered inline and whose error path is a C-epilogue-free body-local
  tail that propagates the trap code in C_RET0 (code).
