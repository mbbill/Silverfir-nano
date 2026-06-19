- The arm64 GP local-cache budget is a two-tier bank: callee-saved registers
  form a free tier-1 cache and caller-saved registers form a spilled tier-2
  cache, spanning far more registers than a single fixed set
  (`compile_backend_config`).

## Moves

- 2026-03-15 (45c0fcd3) replaced [[fixed-gp-cache]]: the same two-tier scheme
  that widened the FP cache is applied to the GP bank: callee-saved registers
  (X23-X28) form the free tier-1 cache, caller-saved registers (X9-X15) form the
  spilled tier-2 cache, and the transient bank moves to caller-saved X3-X8,
  growing the GP cache from 3 to 13 registers and transients from 4 to 6 (code)
