- The CFG — each block's successors and predecessors — is derived on demand
  from its terminator (`successor_blocks`) rather than maintained as stored
  per-block edge lists; the terminators are the single source of truth, with no
  separate recompute pass populating edge state.

- Branch terminators carry the SSA values handed to their target block, and
  those carried values count as terminator uses for liveness; block
  parameters flow as ordinary values pinned live until the branch.

## Facts

- 2025-10-25 (553cd6f2) rationale: the conditional-branch terminator carries
  the block-argument values passed to its target and enumerates them among its
  uses so liveness keeps them live across the conditional edge; lowering itself
  ignores the values (phi-to-copy insertion is driven separately from the SSA
  phi nodes), the values exist on the terminator purely so the use-set is
  complete (diff).

## Moves

- 2025-11-05 (1c6d574d) replaced [[stored-edge-lists]]: control flow is encoded in block terminators, so stored predecessor/successor edges were redundant state the builder had to keep consistent at every branch; deriving the CFG from terminators makes terminators the single source of truth (diff).
