- The indirect-call target index is left in a window register and passed to the
  call_indirect handler as a live operand; the window is deliberately not
  flushed before the call, keeping that value hot.

## Moves

- 2025-10-24 (010b53d5) replaced by [[lowering]]: a call is control flow that
  flushes the whole window like a direct call, so leaving the table-index operand
  live in a window register across that flush was a special case the window had to
  preserve; recording the index's vreg in the call metadata and reading it from the
  vreg file makes call_indirect a pure metadata-driven side-effect op (Sig_0_0)
  identical in shape to direct call (code).
