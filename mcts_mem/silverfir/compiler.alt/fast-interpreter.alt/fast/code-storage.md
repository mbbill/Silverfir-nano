- A function's compiled fast IR (instruction array and side data) is stored as an
  owned value on the function spec, tying the compiled code's lifetime to the
  function spec rather than leaking it for the process lifetime.

## Moves

- 2025-08-14 (fec5a3aa) replaced [[leaked-raw-pointer-fast-ir]]: the raw-pointer
  triple leaked the IR boxes to obtain a process-lifetime pointer with no
  ownership; an owned FastCode ties the compiled code and blob to the function
  spec's lifetime instead (code).
