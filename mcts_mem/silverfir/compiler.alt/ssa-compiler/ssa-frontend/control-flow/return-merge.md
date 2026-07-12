- A RETURN, and any branch whose target is the function frame, accumulates its
  result values as a branch source on the function frame rather than
  terminating its own block; at END the builder builds one exit block whose phi
  nodes merge all the recorded return sources.

## Moves

- 2025-10-12 (2bee2d7a) replaced [[per-site-return]]: setting a Return terminator immediately at each return site could not merge multiple return paths, so every return path now accumulates as a branch source on the function frame and on_end builds one exit block whose phi nodes merge them (code).
