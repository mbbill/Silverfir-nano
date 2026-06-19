- RV32 lowers each i64 variable shl/shr_s/shr_u/rotl/rotr by calling an extern
  "C" host helper that joins the lo/hi pair into a u64, performs the operation,
  and returns the 64-bit result.

## Moves

- 2026-04-25 (dc4e31a7) replaced by [[rv32-i64-shift]]: a host-helper call per
  i64 variable shift/rotate was costly; growing the RV32 GP scratch pool to six
  registers (keeping t2/gp/tp available) lets the lo/hi pair shift be emitted
  inline with no call (code).
