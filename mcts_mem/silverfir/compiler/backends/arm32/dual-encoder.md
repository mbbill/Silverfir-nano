- The arm32 backend's `enc` module resolves at build time to either A32
  (`enc_a32.rs`) or Thumb-2 (`enc_t2.rs`) via a `#[path]` cfg swap on
  `sf_arm32_isa_thumb`; the Thumb-2 encoder mirrors every public function of the
  A32 encoder; the rest of the backend compiles unchanged against whichever is
  selected.

- On Thumb-2 builds every address used as a branch target (function entries,
  direct-call patch targets, continuation/local-ptr addresses, the JITted
  function pointer Rust enters via `blx reg`) has its LSB set to 1; BX/BLX
  switch the CPU into Thumb mode on entry; on A32 builds the helper is a no-op
  (`thumb_interworking_bit`).

## Facts

- 2026-04-14 (1f5c0da9) rationale: Thumb-2 mixes 16-bit and 32-bit instructions,
  but the backend's patch and fixup tables (MOVW/MOVT at fixed offsets,
  continuation/local-ptr offset arithmetic) assume a fixed instruction width, so
  the Thumb-2 encoder pads every 16-bit instruction to a 4-byte slot with a
  trailing 16-bit Thumb NOP (0xBF00) so every instruction occupies a uniform
  4-byte slot and the shared offset-based patcher works unchanged across both
  encoders (code).

- 2026-04-14 (1f5c0da9) rationale: A32 dispatches a br_table in O(1) with a
  PC-relative jump table (`ADD PC, PC, Rindex, LSL #2` plus a NOP pad); Thumb-2
  cannot express this (`ADD PC, PC, Rm, LSL #N` is UNPREDICTABLE when Rd=PC with
  a non-zero shift) so it emits an O(N) cmp+beq chain (code).

- 2026-04-14 (1f5c0da9) rationale: the Thumb-2 O(N) cmp+beq br_table chain is
  accepted because wasm br_tables are almost always small (sourced).

- 2026-04-14 (1f5c0da9) pitfall: Thumb-2 has no per-instruction condition field;
  conditional execution requires an `IT <cond>` prefix covering the next
  instruction. Callers that kept writing the direct A32 conditional form under
  Thumb-2 would emit only the unconditional op and silently drop the IT prefix;
  the fix routes all conditional DP/MOV emission through emit_*_cond_into helpers
  that, on Thumb-2, emit the IT prefix then the unconditional form plus a
  trailing NOP to stay 4-byte-slot-aligned (code).

- 2026-04-22 (05464604) pitfall: a Thumb-2 PUSH/POP register-list encoder must
  special-case a single-register list — the T2 STMDB/LDMIA form with one register
  is UNPREDICTABLE per the ARMv7 ARM, so the single-register path emits the T3
  form (STR Rt,[SP,#-4]! / LDR Rt,[SP],#4) instead; the bug was masked under
  lenient qemu but faults under a hardened decoder / real M33 silicon (code).
