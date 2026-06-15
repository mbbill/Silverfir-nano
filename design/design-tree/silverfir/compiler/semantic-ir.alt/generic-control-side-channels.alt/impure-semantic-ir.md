- Each semantic op carries a resolved D-variant (1-4) and the stack height
  before it executes (pre_height).

- Operand-cache management is expressed inline in semantic IR as explicit
  CacheSpill / CacheFill marker ops.

- Control flow is linear fallthrough plus an alt_target IR index.

## Moves

- 2026-03-09 (ab127bb7) replaced by [[generic-control-side-channels]]: the old
  semantic layer leaked stack-machine and TOS-cache state (variant, pre_height,
  spill/fill) forward into the backend, forcing backend codegen to deduce
  register behavior from stack metadata; purifying semantic IR pushes that
  policy down into a dedicated planning stage so each later pass reasons about
  loops/calls/locals without reconstructing them from low-level code (diff).
