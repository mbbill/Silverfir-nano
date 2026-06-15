- arm32 emits i64 clz/ctz/popcnt/shift and i64 add/sub as inline native
  sequences directly on the allocated host register pair (adds/adc, subs/sbc for
  add/sub) rather than calling out-of-line C helpers; the register-count
  rotl/rotr keep their helper fallback (rare, and the inline sequence is not a
  clear win once icache pressure is counted) alongside the i64 division/remainder
  and i64-to-float helpers (`compile_i64_pair_addsub`).

## Facts

- 2026-04-22 (c0f7ed6e) pitfall: inline i64 add/sub emits adds/adc (subs/sbc)
  across the host register pair, so the low-half step's destination must not be
  a register holding a high-half input the carry step still needs; the only
  harmful alias is dst_lo == a_hi or dst_lo == b_hi, broken by routing the low
  result through a scratch before the adc/sbc (diff).

## Moves

- 2026-04-22 (63f75d3d) replaced [[i64-via-c-helpers]]: calling an extern C
  helper for every i64 clz/ctz/popcnt/shift on the 32-bit ARM backend is slow;
  these now emit inline native sequences directly, while register-count rotl/rotr
  keep the helper fallback (rare in practice, and the inline sequence is not a
  clear win once icache pressure is counted) alongside the i64 division/remainder
  and i64-to-float helpers (diff).

- 2026-04-22 (c0f7ed6e) replaced [[i64-via-c-helpers]]: the old add/sub form
  wrapped its inline adds/adc (subs/sbc) in the helper-call ABI shell — spilling
  caller-saved GP regs, staging all four operand halves into R0-R3 via
  emit_quad_args_to_r0_r3, emitting on fixed registers, moving results back
  through R0:R1, then restoring — so every i64 add/sub round-tripped through
  fixed registers needlessly; add/sub now emit adds/adc directly on the allocated
  host register pair with no spill and no R0-R3 round-trip (guarded against the
  carry-chain alias where dst_lo overwrites a hi-half input) (diff).
