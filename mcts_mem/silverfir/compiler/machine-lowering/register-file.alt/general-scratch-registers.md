- Fixed machine registers 2 and 3 are general-purpose scratch (SCRATCH0/SCRATCH1),
  handed out by index through a dedicated temp accessor.

- Lowering that needs a scratch register asks the regfile for temp 0/1 rather than
  borrowing an unowned transient lane.

## Moves

- 2026-03-13 (110b77f0) replaced by [[register-file]]: memory 0 is the hottest runtime view, so the two fixed scratch slots are repurposed to pin mem0_base/mem0_size for the whole function; ad hoc scratch is instead borrowed from transient lanes that no live value currently owns (code).
