- LIR-to-XIR lowering is stateless: register allocation already happened in the
  middle stage, and the backend assigns no slots — physical registers R0..R7 map
  one-to-one onto the eight hot window slots v0..v7, and each handler's register
  permutation is read off the already-assigned operands (`lower_lir`).

- Eight abstract registers are carried hot in CPU registers across the tail-call
  chain; values that do not fit are moved to and from numbered spill slots by
  explicit spill instructions the backend emits, not by on-the-fly slot
  allocation.

- A spill instruction batches up to two register-slot pairs into one dispatch
  (registers carried in the handler's permutation signature, slots in the
  immediate fields), collapsing several spill moves into a single handler
  dispatch.

- LIR blocks are emitted in the register allocator's RPO order, not block-array
  order, matching where liveness, allocation, and merge-point reconciliation are all
  computed; fall-through branch elision uses the RPO-next block, and the
  emitted layout matches the order the allocator reconciled.

## Facts

- 2025-11-29 (005fae86) measurement: Apple-Silicon dispatch micro-benchmarks
  showed passing 8 virtual registers through the `preserve_none` convention costs
  ~0.6ns/op versus ~0.5ns for 3 (nearly free, since the values stay in CPU
  registers), while any dynamic register selection is the real cost — indexing an
  8-element register array reached 1.7-17ns and nested-ternary register selection
  3.4ns from branch misprediction; the lesson is to scale to 8 hot registers by
  baking the register choice into per-permutation handlers, not selecting
  dynamically — full data in [[register-model.fact/dispatch-microbench]] (code).

- 2025-11-09 (2d92cf03) statement: the window-management handlers (win_load /
  win_store and their batched and swap forms) were deleted wholesale and a spill
  handler pair moving values between numbered spill slots and the registers took
  their place when register allocation moved to the middle stage (code).

- 2025-11-30 (f88a98cd) rationale: spill handlers access the current frame's spill
  area through a raw pointer threaded in the handler signature and read/write slots
  by pointer arithmetic with no runtime bounds check, sound because spill-slot
  indices are fixed by the register allocator and validated at lowering, so an
  out-of-range index cannot reach runtime (code).

## Moves

- 2025-11-09 (d04cbd44) replaced [[window-manager]]: with register allocation
  moved into the middle stage, physical registers R0/R1/R2 already map
  one-to-one onto window slots v0/v1/v2, so the backend needs no WindowManager
  state and emits spills explicitly instead of allocating slots on the fly
  (code).

- 2025-11-29 (005fae86) replaced [[stateless-perm3]]: three hot registers forced
  heavy spill/load traffic, and dispatch benchmarks showed passing eight
  registers through the preserve_none convention is nearly free per instruction
  even though the per-permutation handler count grows ~10x (to ~15K), so the
  interpreter was widened to eight hot registers to keep more values resident
  (code).

- 2025-11-26 (dda758f9) replaced [[single-pair-spill]]: a single register-slot
  pair per instruction meant one handler dispatch per spilled value, and dispatch
  count is the interpreter's dominant cost; batching up to three pairs into one
  instruction (registers carried in the handler's permutation signature, slots in
  the immediate fields) collapses several spill moves into a single dispatch
  (code).
