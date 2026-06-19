- Control flow is encoded entirely in the threaded dispatch: br, br_if, and
  br_table handlers return the next instruction pointer to dispatch to — read from
  an immediate or selected from a jump table — rather than incrementing a loop
  variable; a branch is a tail to the target instruction.

- The handler for an operation is selected by the operand type, not the result
  type: comparisons and eqz return i32 but lower to the handler keyed on their
  operand type, taken from the input value's type.

- A br_table jump table holds one direct pointer per case target plus the
  default, each computed from its target block's start offset; no per-edge code
  runs between the table dispatch and the target block (phi elimination having
  moved upstream into the middle/LIR stage).

- A function's result slots are recovered from the Return instruction's named
  result values, not assumed to be the low slot indices; results may land in
  whatever slots register allocation chose.

- The handler permutation is selected to match where the operands already sit
  rather than shuffling operands into fixed slots; no runtime slot-shuffle
  instructions are emitted.

- An indirect call's table-index operand travels in the call's side-table
  metadata like every other call operand; a call is a uniform
  metadata-driven control-flow op rather than carrying a live operand across the
  pre-call window flush.

- Sub-width signed/unsigned loads and stores are realized by selecting a
  per-width, per-sign memory handler (load8_s, load16_u, store8, …) rather
  than by a size-and-sign field decoded at run time; the immediate carries
  only the offset (and the memory index for non-default memories), and the
  full-width access is its own handler.

## Facts

- 2025-10-12 (a5dc69b9) pitfall: comparison operators return i32 but must lower to
  the handler specialized on the operand type, taken from the lhs/rhs value type,
  not the i32 result type; selecting by the result type silently picks the i32
  comparison for i64/f32/f64 operands (code).

- 2025-11-12 (64d36bb5) pitfall: a unary op's type field is what keys its
  generated handler, so it must be the operand/input type, not the boolean result
  type; i64.eqz decoded with the i32 result type dispatched to the i32 handler
  that reads only 32 bits of the 64-bit operand (code).

- 2025-10-14 (87b2834b) pitfall: a lowered function can contain several blocks
  with a Return terminator (br_table return blocks plus the unified exit);
  lowering must pick a reachable one — preferring the entry block, else the block
  with the most predecessors — not the first Return block it finds, which may be
  unreachable (code).

- 2025-10-10 (e6646469) pitfall: the lowering emits a phi's predecessor copies
  only at an explicit branch terminator on the predecessor block; a block reaching
  a phi-bearing successor by implicit fall-through emitted no copies, so a
  loop-header phi was never initialized from the entry edge — every CFG edge into
  a phi-bearing block must be a materialized branch terminator (code).

- 2025-10-16 (a830c8a8) pitfall: br_table lowering emitted a jump-over-the-stubs
  instruction for layout symmetry, but br_table always branches, so that jump was
  dead; when the br_table block is the last block there is no valid after-stubs
  location, so the stub-skip jump is removed entirely (code).

- 2025-11-12 (0c84b408) pitfall: register allocation can land both arms of a
  select in the same physical register, leaving a degenerate select whose two
  inputs are identical; codegen detects that and emits a plain copy instead of a
  select, since the condition is then irrelevant (code).

- 2025-11-01 (2646713a) statement: each SSA instruction reports its used and
  defined values through uses()/defs(), with defs() returning multiple values for
  calls to model WebAssembly multi-value returns, the interface lowering and
  liveness consume (code).

- 2025-10-25 (bd23526f) rationale: br_if originally branched on a ZERO condition
  and the lowerer compensated by swapping then/else targets; this was abandoned for
  the native WebAssembly polarity (branch on NON-ZERO to then_target, unconditional
  br to else_target), producing equivalent control flow at the same instruction
  count without the target-swapping inversion (code).

- 2025-10-25 (bd23526f) statement: the original ZERO-condition br_if form was
  justified as 'easier code generation' (sourced).

## Moves

- 2025-10-11 (4b55a6c2) replaced [[op-only-dispatch]]: op-only dispatch could not
  select between the existing i32 and i64 handlers, so i64 binary ops lowered to
  the i32 handler; keying on (type, operation) routes each to its type-specialized
  handler (code).

- 2025-10-12 (2bee2d7a) replaced [[br-table-direct-targets]]: a single jump table
  pointing straight at target blocks cannot carry the per-edge phi assignments
  each br_table target needs, so each table entry now points at a small stub that
  performs that target's phi stores before jumping to the real block (code).

- 2025-10-16 (d1307ec8) replaced [[shuffle-to-fixed-slots]]: forcing operands into
  fixed slots emitted runtime shuffle-swap instructions on every misaligned
  operation; with the full permutation handler matrix the codegen instead picks
  the handler whose slots match the operands' current positions, so values stay
  put and no shuffle instructions are emitted (code).

- 2025-10-23 (a7a48d92) replaced [[low-index-result-slots]]: the low-index
  assumption could not express results landing in slots chosen by register
  allocation, so it mis-read results whenever a result vreg was not in
  0..num_results; deriving the slots from the Return instruction names the actual
  result vregs (code).

- 2025-10-24 (010b53d5) replaced [[indirect-target-in-window]]: a call is control
  flow that flushes the whole window like a direct call, so leaving the
  table-index operand live in a window register across that flush was a special
  case the window had to preserve; recording the index's vreg in the call metadata
  and reading it from the vreg file makes call_indirect a pure metadata-driven
  side-effect op (Sig_0_0) identical in shape to direct call (code).

- 2025-11-09 (d04cbd44) revived: the LIR migration moved phi elimination upstream
  into the middle stage (phi nodes become ParCopy on CFG edges before lowering),
  so a br_table target carries no per-edge phi stores and the jump table reverts
  to the [[br-table-direct-targets]] shape — one direct pointer per target plus
  the default — dropping the per-target stubs (code).
