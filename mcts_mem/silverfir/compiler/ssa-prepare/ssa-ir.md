- The prepared middle-end IR is a linear SSA-IR whose slot ops are role-split:
  CellGet/CellSet are cell accesses (cells: named multi-use value slots, wasm
  locals being their origin, carrying `CellId` identity with frame homes in a
  published table) and Fill/Spill are operand-slot spill/fill addressed by
  frame geometry; cell access is distinguishable from deep-stack spill/fill
  (`SsaOp`).

- The IR is linear-SSA single-use: each transient value has exactly one use in
  the op stream, which later passes (constant absorption, sink planning) rely on.

## Facts

- 2026-03-17 (6612c624) pitfall: `local.tee` must not leave its value on the live
  stack while also storing it to the local slot — that gives the value two uses
  within the op stream and breaks the single-use invariant; it instead pops the
  value, stores it to the slot, and reloads a fresh single-use value from the slot
  for the continuation (code).

## Moves

- 2026-03-10 (ce18df44) replaced [[rotating-window-lir]]: by the time LIR exists
  TOS must no longer be a rotating window / variant / implicit top-register
  selection; representing TOS instead as SSA values, block parameters, and
  successor arguments lets LIR distinguish transient SSA values from durable
  hot-local state and frame/memory effects, and keeps the
  no-general-register-allocator model (TOS lanes are an entry/edge interface, hot
  locals are durable cached machine state, native lowering only places values
  into fixed VM locations) (code).

- 2026-03-26 (98de6d7b) replaced [[undifferentiated-slot-ops]]: a single
  LoadSlot/StoreSlot pair could not distinguish canonical-local access from
  deep-stack spill/fill nor carry the semantic local version that sink planning
  needs to prove the old version of a local is dead before redirecting a producer
  into the local's home (code).
