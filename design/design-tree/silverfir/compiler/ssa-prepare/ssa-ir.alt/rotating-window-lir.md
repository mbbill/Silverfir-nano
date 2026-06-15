- The LIR op carries a rotating-window top (WindowRotation) instead of explicit
  T0..T3 operands; the backend derives the concrete T register from the current
  rotation plus the opcode stack effect.

- TOS is a 4-register rotating cache kept valid by planner-inserted Spill/Fill;
  control flow is a linear op stream with an optional alt target, not a
  basic-block graph.

## Moves

- 2026-03-10 (ce18df44) replaced by [[ssa-ir]]: by the time LIR exists TOS must
  no longer be a rotating window / variant / implicit top-register selection;
  representing TOS instead as SSA values, block parameters, and successor
  arguments lets LIR distinguish transient SSA values from durable hot-local
  state and frame/memory effects, and keeps the no-general-register-allocator
  model (TOS lanes are an entry/edge interface, hot locals are durable cached
  machine state, native lowering only places values into fixed VM locations)
  (diff).
