- Binary operations are split into two permutation families by commutativity:
  commutative ops (add, mul, and, or, xor, eq, ne, min, max, ref_eq) use a
  3-arrangement family, exploiting that operand order is free; non-commutative
  ops use a 6-ordered-arrangement family.

- Neither family enumerates a same-register arrangement (both inputs in one
  register); every arrangement writes the output in place to the first input's
  register slot.

## Moves

- 2025-11-10 (11f72dc7) replaced by [[handler-generation]]: the commutative
  family kept only 3 of the operand arrangements by exploiting operand-order
  freedom and neither specialized family enumerated same-register arrangements, so
  the allocator could emit register combinations that had no handler; one Sig_2_1
  family enumerating all 9 register permutations (including the same-register
  V0V0V0/V1V1V1/V2V2V2 cases) covers every allocator output, at the cost of
  generating the full handler set for commutative ops too (code).
