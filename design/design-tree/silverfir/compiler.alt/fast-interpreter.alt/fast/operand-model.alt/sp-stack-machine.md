- The frame is laid out [params][locals][operand stack]; fp points at the
  params/locals base and a separate sp parameter starts at fp+params+locals as
  the operand-stack base, with locals frame-relative and operands pure
  stack-machine.

- Each handler reads and writes operands relative to the stack pointer (sp[-1],
  sp[-2]) and adjusts sp; there is no compile-time operand-slot encoding and no
  temp bookkeeping.

- The TOS registers, while being migrated in, shadow the sp path: each handler
  computes the result both ways under validation mode and asserts they agree.

## Moves

- 2025-12-14 (b9499733) replaced [[slot-tracking]]: the slot model encoded an
  absolute frame slot index for every operand of every instruction and tracked
  operand-stack positions plus temp allocation at compile time; switching to an
  implicit operand stack addressed via a stack pointer drops all operand-slot
  encoding and the compile-time slot/temp bookkeeping, leaving handlers to read
  sp[-1]/sp[-2] and adjust sp (diff).

- 2026-01-24 (d0c89f0b) replaced by [[operand-model]]: once the shadow-validated
  TOS path was trusted, removing the sp parameter and the parallel sp computation
  makes the register cache the sole source of truth, eliminating the
  per-instruction memory round-trip the cache existed to avoid; stack addresses
  still needed for spill/fill are derived from fp plus a fixed-offset frame
  layout instead of a tracked sp (diff).
