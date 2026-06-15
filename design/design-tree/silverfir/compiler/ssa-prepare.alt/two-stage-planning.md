- The planning layer produces a separate planned-op IR (PlannedProgram/PlannedOp)
  via `build_planned_program` before any LIR exists, then lowers that planned-op
  IR to LIR — two IRs for one preparation job.

- Planning decides where the logical top of stack sits in a rotating TOS window
  and emits spill/fill artifacts against that window (`TosRotation`).

- Hot-local planning selects which locals live in a dedicated hot-local register
  class and produces a `HotLocalPlan` mapping; LIR encodes local accesses as
  hot-local register-class instructions carrying that register budget.

## Facts

- 2026-03-12 (f3fca0b4) statement: locals must keep canonical frame-slot identity
  in LIR because calls, returns, and frame layout rely on stable slot identity, so
  hot-local caching is execution policy decided below LIR (register-cached locals
  are mirrors of their canonical slot homes, not replacements); encoding local
  accesses as special hot-local storage kinds — and carrying the hot-local
  register budget — in LIR was wrong, and was replaced by ReadSlot/WriteSlot
  against canonical FrameSlot identity (diff).

## Moves

- 2026-03-09 (ab127bb7) replaced [[backend-lowered-ir]]: the lowered IR still
  leaked stack-machine state (pre_height, variant, window, helper-entry families
  like read_t0/write_t1) so the backend had to infer register validity and operand
  locations from stack-height metadata; the backend-facing IR must end the
  stack-machine abstraction and describe explicit register/memory behavior instead
  (diff).

- 2026-03-12 (2ea0bb68) replaced by [[ssa-prepare]]: the planning layer
  previously emitted its own intermediate planned-op IR with a rotating-TOS
  window and a hot-local register-class plan, then lowered that to LIR — two IRs
  for one preparation job; collapsing it so prepare_function produces prepared
  LIR directly removes the redundant planned-op IR and the rotating-TOS
  representation, and replaces the hot-local register-class plan with pure
  local-cache preference analysis (ranking hints, not storage kinds), since
  canonical local identity must stay slot-based and the cache swap is execution
  policy decided below LIR (diff)
