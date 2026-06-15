- arm32 lowers i32 div_s/div_u/rem_s/rem_u with the hardware SDIV/UDIV
  instruction inline (with MLS to derive the remainder), assuming the target has
  the Integer-Divide extension (`enc::sdiv`, `enc::udiv`, `enc::mls`).

## Facts

- 2026-04-10 (272e06dd) statement: the i64 div/rem helpers (arm32_i64_div_s etc.)
  are left as software calls because there is no hardware 64-bit divide; the
  test runner switched its qemu cpu to cortex-a15 to model an IDIV-capable core
  (diff).

## Moves

- 2026-04-10 (272e06dd) replaced [[software-divrem-helpers]]: calling extern-C
  software-division helpers forced a caller-saved spill/restore and host-call
  round-trip around every i32 div/rem; emitting the hardware SDIV/UDIV
  instruction inline (with MLS to derive the remainder) removes the call and
  spill entirely (diff).
