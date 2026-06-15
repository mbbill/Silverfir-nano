- On guard-page 64-bit targets the wasm value stack is allocated with an
  inaccessible guard range after its usable end (sized to the module's largest
  static frame), and stack-overflow protection is a single body-entry probe of the
  highest frame slot that faults into the guard on overflow; the signal handler
  classifies the fault as StackOverflow by whether the fault address lands in
  [stack_end, stack_guard_end) (`GuardPageStack`, `classify_trap_kind`).

- Explicit per-call-site frame-limit prechecks are retained only on non-guard
  targets (`use_explicit_stack_prechecks`).

## Facts

- 2026-05-14 (d3af717a) statement: the guard range after the usable stack is
  expanded by the module's largest static frame footprint (not just one wasm
  page) so a single overflowing callee-frame probe at the highest touched slot
  cannot jump past the protected reservation into mapped memory; the handler
  distinguishes a stack-overflow fault from a linear-memory OOB fault purely by
  whether the fault address lands in [stack_end, stack_guard_end); NativeContext
  carries stack_guard_end and the handler reads the siginfo fault address (diff).

## Moves

- 2026-05-14 (d3af717a) replaced [[per-call-precheck]]: computing the proposed
  callee frame base and comparing it against stack_end before every local call
  adds arithmetic and a branch to each call site; on guard-page 64-bit targets the
  wasm value stack can instead be allocated with an inaccessible guard range sized
  to the module's largest frame, so a single body-entry probe of the highest frame
  slot faults into the guard on overflow and the signal handler classifies it as
  StackOverflow by fault address, removing the per-call precheck from the hot
  local-call path (diff).
