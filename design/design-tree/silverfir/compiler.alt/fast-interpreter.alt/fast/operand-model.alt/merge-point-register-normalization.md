- At every structured control-flow op (LOOP, IF, ELSE, END, BR, BR_IF, BR_TABLE)
  the builder emits a fill, normalizing any memory-resident TOS operands back into
  the TOS registers; the branch target, back-edge, and fall-through all observe
  the same register-resident state.

- A loop body's initial entry and its back-edges share one normalized register
  state: the initial entry runs the same fill the back-edges run before branching.

- Branch handlers expect their merge operands in TOS registers, not in memory.

## Moves

- 2026-01-25 (b322a614) replaced by [[operand-model]]: register normalization
  could not reconcile a branch path that drops operands with the fall-through path
  that keeps them in different registers, so merge values are spilled to memory —
  the one location both edges agree on (diff).
