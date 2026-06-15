- The machine register file is statically partitioned [fixed | gp_cache | gp_trans
  | fp_trans | fp_cache]: each GP and FP bank is split at compile time into a
  fixed-size cached-local sub-bank and a transient sub-bank.

- A register's purpose is determined by its number alone; a register in the
  transient sub-bank can never hold a cached local and vice versa, even when the
  other sub-bank is idle.

- Lowering classifies a register into (partition index, is_cache) from its number
  rather than tracking a per-point semantic owner.

## Facts

- 2026-04-06 (a50a44d4) measurement: the old fixed transient sub-bank is a
  sliding window over the operand stack — a block needs transient registers only
  for the stack values it actually touches (pop N + push M near the top), not for
  all live stack values, so the bottom of a deep stack sits untouched in the
  frame; across 9 WASI benchmarks peak transient pressure rarely exceeded the
  static budget (a function with 66 locals still peaked at only 6 live
  transients), so the static 9-register split was almost never the bottleneck and
  spill/fill stayed small — the freed flexibility a unified budget buys lands on
  the cached-local side, not the transient side (diff).

## Moves

- 2026-04-04 (ea0cf447) replaced by [[register-file]]: a register fixed as cache-only or transient-only by its number could not be reassigned to whichever purpose a block needed; making each dynamic register's role explicit owner metadata lets one bank serve cached locals and transient stack values per program point (diff).
