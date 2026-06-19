- The jt backend interprets the same in-place bytecode as the match-based dt
  backend but dispatches each opcode through a 256-entry table of per-opcode
  handler function pointers indexed by the opcode byte, each handler returning a
  Step that drives an explicit frame stack, instead of a single match over the
  opcode.

- jt shares the rest of the in-place interpreter design with dt: a single
  64-bit-word value stack, immediates decoded from the function body on every
  execution, and structured control flow resolved through the precomputed jump
  table.

## Facts

- 2025-08-08 (f9ef9d6a) pitfall: after a CALL pushes the callee frame, the inner
  per-frame dispatch loop must break back to the outer driver (which re-pops the
  frame and reinstalls func_inst/code/pc); the call handler used continue
  'inner, which kept running the caller's code and stale pc against the new
  frame — the frame-switch boundary requires break, not continue (code).

## Moves

- 2025-08-08 (d33f2413) removed: the handler-pointer-table dispatch was not a
  dead end — it has higher headroom than the match-based dt dispatch, and the
  idea was carried directly into the ssa/xir interpreter (the intended final
  goal), which dispatches exactly this way; it was abandoned only here on the
  in-place interpreter because optimizing the in-place method is not worthwhile —
  the in-place approach has an inherent performance ceiling, so a faster dispatch
  there buys nothing (sourced).
