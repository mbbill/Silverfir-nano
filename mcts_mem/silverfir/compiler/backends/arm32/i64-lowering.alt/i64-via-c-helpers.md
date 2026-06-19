- i64 clz, ctz, popcnt, shl, shr_s, shr_u, and mul are lowered on arm32 as calls
  to out-of-line extern "C" helper functions that take the value as lo/hi u32
  halves and return a u64.

- i64 add/sub lower inside the helper-call ABI shell: spill the caller-saved GP
  regs, stage the four operand halves into R0-R3 (emit_quad_args_to_r0_r3), emit
  adds/adc (subs/sbc) on the fixed registers R0:R1 += R2:R3, move the result pair
  back out of R0:R1 into dst (emit_pair_results_from_r0_r1), then restore the
  caller-saved regs.

## Moves

- 2026-04-22 (63f75d3d) replaced by [[i64-lowering]]: calling an extern C helper
  for every i64 clz/ctz/popcnt/shift on the 32-bit ARM backend is slow; these now
  emit inline native sequences directly, while register-count rotl/rotr keep the
  helper fallback (rare in practice, and the inline sequence is not a clear win
  once icache pressure is counted) alongside the i64 division/remainder and
  i64-to-float helpers (code).

- 2026-04-22 (c0f7ed6e) replaced by [[i64-lowering]]: the old add/sub form
  wrapped its inline adds/adc (subs/sbc) in the helper-call ABI shell — spilling
  caller-saved GP regs, staging all four operand halves into R0-R3 via
  emit_quad_args_to_r0_r3, emitting on fixed registers, moving results back
  through R0:R1, then restoring — so every i64 add/sub round-tripped through
  fixed registers needlessly; add/sub now emit adds/adc directly on the allocated
  host register pair with no spill and no R0-R3 round-trip (guarded against the
  carry-chain alias where dst_lo overwrites a hi-half input) (code).
