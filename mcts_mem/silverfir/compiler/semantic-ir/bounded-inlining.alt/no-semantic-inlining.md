- Semantic decoding preserves all original local call boundaries; there is no semantic-IR call expansion.

## Facts

- 2026-09-06 rationale: the earlier removal of inlining is recorded in the compiler's 2026-04-14 move and semantic IR's 2026-06-14 fact: per-function footprint and Algorithm4 local-cache pressure outweighed the earlier gains. Those constraints continue to apply to small-memory targets (sourced).

## Moves

- 2026-09-06 (049874d0) replaced by [[compiler/semantic-ir/bounded-inlining]]: Resource-bounded structured expansion recovers substantial recursive execution throughput while retaining original call boundaries for finite-memory and 32-bit GP targets; the user accepts the measured ERC20 startup cost (sourced).
