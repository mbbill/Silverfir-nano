- One side table per function, built during validation; no control-flow
  discovery happens at runtime.
- Each entry holds the branch's pc, the target pc, a stack offset, an arity,
  and `next_idx` — the next jump-table slot to use once this branch is taken.
- The interpreter carries a current jump-table index alongside the pc; entries
  are reached positionally via `next_idx` — there is no pc→entry lookup at
  runtime.
- A taken branch is O(1): pc moves to the target, and the operand stack
  collapses by the stack offset while preserving the top `arity` values.
