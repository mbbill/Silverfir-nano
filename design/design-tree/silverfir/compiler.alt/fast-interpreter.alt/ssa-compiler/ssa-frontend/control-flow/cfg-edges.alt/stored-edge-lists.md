- Each SSA basic block stores its predecessor and successor block-id lists as
  fields alongside its terminator.

- The builder maintains these edges incrementally: every branch, if/else, loop,
  br_table, and merge adds the target as a predecessor as it sets the
  terminator.

- A finalize pass over the function recomputes successors from terminators and
  rebuilds all predecessor lists once construction finishes.

## Facts

- 2025-11-01 (9af2ddc5) rationale: predecessor edges were added incrementally
  during construction while successors were never populated on the production
  path, leaving the CFG potentially inconsistent with the terminators;
  recomputing both lists from the terminators once at the end of build
  guarantees the CFG always matches the terminators the middle stage consumes
  (diff).

## Moves

- 2025-11-05 (1c6d574d) replaced by [[cfg-edges]]: control flow is encoded in block terminators, so stored predecessor/successor edges were redundant state the builder had to keep consistent at every branch; deriving the CFG from terminators makes terminators the single source of truth (diff).
