- A memory bounds check is lowered as a branch terminator: the comparison's true edge
  jumps to a dedicated trap block and the false edge to the access continuation block
  (split CFG).

- The leaf op returns `LeafLowering::Split`, naming the continuation block, the trap
  block, and the trap kind for the lowerer to wire up.

## Moves

- 2026-03-14 (5f7b0f37) replaced by [[memory-access]]: a straight-line out-of-bounds guard whose only cold behavior is to trap does not need a continuation split and a dedicated trap block; an inline TrapIf preserves explicit trap semantics while the backend lowers the guard to a shared cold trap stub (diff).
