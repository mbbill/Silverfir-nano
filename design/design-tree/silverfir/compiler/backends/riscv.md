- The RISC-V register identities, base instruction encoder, and architectural
  register plan live in a shared `riscv/` module with two real consumers — RV64
  and RV32 each supply only their XLEN-sized frame math, literal width, and (for
  RV32) pair-valued i64 lowering as a thin ABI policy over it (`RiscvReg`).

## Facts

- 2026-04-26 (de04e532) statement: the RV32 backend is also selected for
  FPU-less RV32 triples (riscv32imac/imc/im/i, e.g. Hazard3 / RP2350 RV mode):
  build.rs leaves sf_fp_dp off for non-gc triples and the backend emits
  integer-only Wasm modules with no FP ops, mirroring the integer-only
  sf_backend_thumbm posture (diff).

- 2026-04-24 (17a3fe31) rationale: RISC-V loads/stores have only
  `[base + signed-12-bit-offset]` addressing and no base+index memory form, so
  the indexed global/memory lowering cannot honor the cross-backend stable-base
  indexed-addressing rule (keeping the original physical base as the load/store
  base operand, whose scratch-base violation was a measured 17% SHA-256
  regression on the register-rich backends); instead it computes
  `addr = base + extend(index)` into a GP scratch and uses the immediate offset
  only when the static offset fits 12 signed bits, and because the GP scratch
  pool has 2 slots an out-of-range offset must be folded into that address
  scratch before the source value is materialized, or the address, source, and
  large-offset scratches overlap live ranges and exhaust the pool (diff).
