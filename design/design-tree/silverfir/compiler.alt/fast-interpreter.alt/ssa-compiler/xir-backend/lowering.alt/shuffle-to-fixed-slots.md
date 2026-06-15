- Before an operation, operands are moved into the slots its single fixed-slot
  handler expects, emitting shuffle_swap window instructions when a value is in the
  wrong slot (any 3-slot permutation reachable in at most two swaps).

- The codegen emits only the v0_v1 permutation of each arithmetic handler; the
  handler-selection table maps every operation to its v0_v1 form and returns
  nothing for other slot pairs.

- Window-slot state is tracked: an operand already in its target slot needs no
  shuffle, but cross-slot use requires an emitted swap.

## Moves

- 2025-10-16 (d1307ec8) replaced by [[lowering]]: forcing operands into fixed slots
  emitted runtime shuffle-swap instructions on every misaligned operation; with the
  full permutation handler matrix the codegen instead picks the handler whose slots
  match the operands' current positions, so values stay put and no shuffle
  instructions are emitted (diff).
