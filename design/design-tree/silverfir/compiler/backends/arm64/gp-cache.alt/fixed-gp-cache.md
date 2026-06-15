- The ARM64 GP policy preset spends only 3 cached-local registers and 4 transient
  registers of the backend's GP capacity; the dynamic register pool lists
  caller-saved registers (X9-X15) before callee-saved ones (X23-X28), and the
  small cache budget draws from the caller-saved end.

## Moves

- 2026-03-15 (45c0fcd3) replaced by [[gp-cache]]: the same two-tier scheme that
  widened the FP cache is applied to the GP bank: callee-saved registers
  (X23-X28) form the free tier-1 cache, caller-saved registers (X9-X15) form the
  spilled tier-2 cache, and the transient bank moves to caller-saved X3-X8,
  growing the GP cache from 3 to 13 registers and transients from 4 to 6 (diff)
