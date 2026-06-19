- The backend-lowered IR op carries resolved stack-machine state — a per-op
  D-variant (1-4) selecting which physical rotating-cache registers apply and the
  stack height before the op (pre_height) — and lowering commits hot-local
  placement, InitLocals, and explicit spill/fill before the backend sees it
  (`IrOp`).

- Backends select cold-helper entry points from stack-machine-shaped families
  (read_t0 / read_top2_d1 / write_t1) keyed on the op's pre_height-derived TOS
  register, and the same lowered IR is consumed by the fast, fusion, and native
  backends.

## Facts

- 2026-03-06 (7c8c1ebe) statement: operand-relative frame slots were first
  encoded as an offset plus a magic OPERAND_BASE=16384 placeholder sharing one
  u16 space with real absolute frame slots, resolved by a `fix_slot` pass applied
  to exactly the right fields, so absolute and operand-relative slots could not be
  told apart at the type level; a typed SlotRef with explicit Absolute(u16) vs
  OperandRelative(u16) variants replaced it so frame locals carry absolute slots
  directly and only operand-relative call deltas are rebased at finalization
  (code).

## Moves

- 2026-03-09 (ab127bb7) replaced by [[two-stage-planning]]: the lowered IR still
  leaked stack-machine state (pre_height, variant, window, helper-entry families
  like read_t0/write_t1) so the backend had to infer register validity and operand
  locations from stack-height metadata; the backend-facing IR must end the
  stack-machine abstraction and describe explicit register/memory behavior instead
  (code).
