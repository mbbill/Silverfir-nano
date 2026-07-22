- Every build configuration compiles local functions serially with exactly one
  function's compiler pipeline in flight.

## Moves

- 2026-07-22 (4454fa83) replaced by [[hosted-eager-parallelism]]: independent
  eager function compilation left multicore capacity idle; bounded workers
  preserve eager completion and per-worker streaming while reducing hosted wall
  time (code).
