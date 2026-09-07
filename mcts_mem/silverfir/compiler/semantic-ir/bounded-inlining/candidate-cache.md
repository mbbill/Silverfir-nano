- A per-caller fixed table holds at most eight distinct callee keys, including negative eligibility results. Positive entries index a separate vector of decoded bodies; negative entries carry no body storage.

## Facts

- 2026-09-06 (fc26a6a9) measurement: the compact representation preserves every per-function dump across all20 plus CoreMark (1285 functions per architecture), and all 38 ERC20 functions. ARM ERC20 startup throughput improves 2.2896% against the tree-map cache, but remains 7.6814% below main. All 22 module heap peaks are unchanged; this is a compiler-work improvement, not a peak-memory reduction (sourced).

## Moves

- 2026-09-06 (fc26a6a9) replaced [[compiler/semantic-ir/bounded-inlining/candidate-cache.alt/tree-map-inline-bodies]]: A fixed key/index table separates negative results from body-sized map storage while preserving the same eight-candidate budget, resolution order and generated code (code).
