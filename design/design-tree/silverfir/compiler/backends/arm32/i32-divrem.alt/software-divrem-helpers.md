- i32 div_s/div_u/rem_s/rem_u are lowered by spilling caller-saved GP regs and
  calling the extern-C arm32_sdiv / arm32_udiv software helpers, then restoring;
  there is no inline hardware divide instruction.

## Moves

- 2026-04-10 (272e06dd) replaced by [[i32-divrem]]: calling extern-C
  software-division helpers forced a caller-saved spill/restore and host-call
  round-trip around every i32 div/rem; emitting the hardware SDIV/UDIV
  instruction inline (with MLS to derive the remainder) removes the call and
  spill entirely (diff).
