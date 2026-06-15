- The shared value stack owns a usize sp_offset that is the index of the current
  logical top; push/pop/push_zeros/advance_used mutate it, and sp_offset() reads
  it.

- Entering a callee frame computes the locals base from the stack's current
  sp_offset minus the param count; on return, the result values are shifted down
  to the locals base and sp_offset is reset.

- A register-window flush to the shared stack advances the stack's sp_offset by
  the number of spilled lanes, keeping the two consistent.

## Moves

- 2025-08-16 (10c1c487) replaced by [[register-window]]: the stack kept its own
  sp_offset counter in parallel with the register window's Regs.sp pointer, so
  every register-window flush had to manually reconcile the two (flush bumped
  sp_offset by the spilled lane count); making Regs.sp the single source of truth
  for the live top and threading the logical frame size as an explicit stack_size
  parameter removes the dual-bookkeeping and its sync hazard (diff).
