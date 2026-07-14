- One arm32 backend family covers both 32-bit Arm profiles, selected per target:
  build.rs routes `thumbv*` targets to `sf_backend_thumbm` and other
  `target_arch = arm` targets to `sf_backend_armv7a`, giving the backend one
  encoding-neutral home with target-driven profile selection (`arch/arm32`).

- The GP dynamic bank is grouped by JIT-ABI class — four volatile lanes
  (caller-saved physicals), two preserved lanes on EABI callee-saved
  registers, and a two-lane internal-scratch tail; bodies save and restore
  the preserved lanes they clobber lazily in the body prelude and on every
  return path, with the sub-frame unwinding before the fixed link-save and
  call-record offsets (`lower_preserved_dynamic_body_save`).

## Facts

- 2026-03-27 (55b2ceea) pitfall: when an IntBinary's lhs is an Imm64 it was
  materialized straight into the destination register; if the rhs maps to the
  same hardware register as dst (e.g. both R3 after const folding freed the
  const's register), writing dst first clobbers rhs before the ALU op reads it
  (i32.add(const 1, g) became 1+1). The fix detects the rhs-aliases-dst case and
  materializes the immediate into SCRATCH0 instead — the same
  snapshot-rhs-before-materializing-lhs hazard that bites pair bitops on 32-bit
  backends (code).

- 2026-03-27 (3a7fa0f1) pitfall: i64 pair compare on arm32 must materialize the
  lo/hi comparison operands into out-of-bank scratch registers (SCRATCH0=R12,
  SCRATCH1=R14/LR), not stage them through R0/R1 via push/pop: staging the
  hi-word compare through R0/R1 clobbers a lo operand that happened to live in
  R0/R1, so the subsequent lo-word compare reads the hi value and returns the
  wrong result (code).

- 2026-04-09 (b8f8fca7) pitfall: an i64 pair And/Or/Xor on the migrated ARMv7-A
  backend materializes the lhs into the destination register pair before the ALU
  op; if a rhs half's physical register aliases either destination half (possible
  after dead-input register reuse), writing the lhs clobbers that rhs half before
  it is read. The fix snapshots both rhs halves into owned scratch first whenever
  a rhs half aliases a destination register, then performs the op — Lua's SWAR
  string-hash exposes this pattern via i64 And/Or/Xor, the same
  snapshot-rhs-before-materializing-lhs-into-dst hazard seen on the other 32-bit
  pair-bitop paths (code).

- 2026-07-13 (8be69337) rationale: the preserved lanes reuse registers the
  public entry already bulk-saves for the C ABI; the per-body lazy save
  exists for the internal body-to-body ABI, because internal calls enter
  past the public prologue and would otherwise clobber a caller's carried
  preserved-lane caches (code).

## Moves

- 2026-04-10 (1881a660) replaced [[armv7a-only-backend]]: a single armv7a module
  hardwired to A-profile could not host Thumb-only M-profile targets, and
  build.rs mapped every target_arch=arm to armv7a; renaming the module to arm32
  and splitting build.rs by target (thumbv* -> sf_arch_thumbm, else
  sf_arch_armv7a) gives the 32-bit Arm backend one encoding-neutral home and a
  target-driven profile selection (code).
