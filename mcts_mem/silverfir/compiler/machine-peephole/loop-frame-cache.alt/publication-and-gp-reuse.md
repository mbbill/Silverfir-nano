- Successful loop-cache rewrites also coalesce temporary GP values across ordered full-frame stores inside the rewritten block. Single-read mutable frame words can be carried ahead of exit-only reuse.

## Moves

- 2026-09-06 (937ed3e0) replaced by [[compiler/machine-peephole/loop-frame-cache]]: Two native full20 draws reproduced severe sort regressions that overwhelm the independently confirmed roughly 0.54% CoreMark gain; retain the prior conservative cache and register policy (sourced).
