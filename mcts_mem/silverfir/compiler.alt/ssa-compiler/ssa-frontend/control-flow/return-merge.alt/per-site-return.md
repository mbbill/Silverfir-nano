- A RETURN opcode, and a branch whose target is the function frame,
  immediately materializes its result values and sets that block's
  Terminator::Return; each return site terminates its own block independently
  with no shared exit block.

## Moves

- 2025-10-12 (2bee2d7a) replaced by [[return-merge]]: setting a Return terminator immediately at each return site could not merge multiple return paths, so every return path now accumulates as a branch source on the function frame and on_end builds one exit block whose phi nodes merge them (code).
