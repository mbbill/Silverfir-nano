- A function has two entry points: a public C-ABI entry at which the prologue
  runs, and an internal entry located past the prologue that direct local
  compiled-to-compiled (SF->SF) calls target by skipping the prologue.

- The function tail binds a shared return_ok label (sets C_RET0=0 then runs the
  C epilogue) and a shared return_error label (runs the C epilogue); both
  success and error returns flow through these shared, function-wide tails that
  run the C epilogue regardless of which entry was used; a locally-entered
  callee returning or trapping through a shared tail runs an epilogue whose
  matching prologue it skipped.

- The runtime records a separate root_return code pointer per compiled function
  alongside its entry pointer.

## Moves

- 2026-04-07 (b59caeff) replaced by [[entry-and-tails]]: direct
  compiled-to-compiled (SF-to-SF) local calls enter a callee at an internal
  entry point that does not run the C-ABI prologue, so the old function-wide
  return-error tail (which ran the C epilogue and assumed the prologue had
  executed) was wrong for locally-entered callees; the entry is split into a
  public C-ABI stub that bl's the internal entry and a body whose success
  Return is lowered inline and whose error path is a C-epilogue-free body-local
  tail that propagates the trap code in C_RET0 (diff).
