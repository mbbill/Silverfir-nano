- The instruction header stores an explicit fallthrough pointer to the next
  instruction (non-null for every non-terminal op) alongside the alt branch-target
  pointer; the IR builder threads each non-terminal op's fallthrough to the
  following instruction and routes return/unreachable to the terminal sentinel via
  fallthrough.

- Non-branch handlers advance control flow by reading the instruction's stored
  fallthrough pointer.

## Moves

- 2025-08-17 (ce4fd170) replaced by [[dispatch]]: each non-terminal op's
  fallthrough is the contiguously laid-out next instruction, so storing an explicit
  fallthrough pointer was redundant; fallthrough becomes pc+1 and only the alternate
  control-flow target (alt) stays in the header, dropping a per-instruction pointer
  field and its load on every non-branch op (code).
