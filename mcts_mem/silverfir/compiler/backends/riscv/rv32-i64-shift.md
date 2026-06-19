- RV32 lowers each i64 variable shl/shr_s/shr_u/rotl/rotr register-only inline
  on the lo/hi pair (no call), enabled by growing the RV32 GP scratch pool to
  six registers while keeping t2/gp/tp available (`emit_i64_shl_var`).

## Moves

- 2026-04-25 (dc4e31a7) replaced [[rv32-i64-shift-helpers]]: a host-helper call
  per i64 variable shift/rotate was costly; growing the RV32 GP scratch pool to
  six registers (keeping t2/gp/tp available) lets the lo/hi pair shift be emitted
  inline with no call (code).
