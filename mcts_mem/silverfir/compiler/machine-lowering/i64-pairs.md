- On a backend whose GP register width is 4 bytes the lowerer is the single place
  that maps one i64 LirValue to a (lo, hi) machine-register pair, emitting pair-form
  MachineIR ops (pair binary/divrem/shift/compare and i64<->float conversions)
  directly; on 8-byte-GP backends the scalar path is used and no pair handling runs
  (`lower_i64_pair_leaf`).

- The high-half register of an i64-pair op is dropped when a backward
  per-instruction liveness analysis over MachineIR marks it not demanded by any later
  computation; the backend native emitter (arm32 / riscv32) then emits only the low half
  for selected pair add/sub/and, extend32_s, and small-count right shifts. The analysis
  is a shared MachineIR module feeding both 32-bit GP backends the same dead-high-half
  facts (`Low32DeadHiDefs`).

## Facts

- 2026-03-21 (959fc3c2) rationale: a recovered design document quantifies why the late
  legalizer was abandoned — ~1000 lines existed solely to recover information already
  available during lowering (~400 lines storage-flow analysis to rediscover register
  types, ~200 lines hi-half companion tracking, ~400 lines GP-bank compaction); the new
  design eliminates all three because the lowerer knows the LIR value types, pair
  instructions carry their operands explicitly, and the planner budgets 2 GP lanes per
  i64; everything above MachineIR stays Wasm-shaped and scalar — full document in
  [[i64-pairs.fact/legalization-doc]] (sourced).

- 2026-03-27 (3a9284d1) pitfall: when sinking an i64 value into a cached local on a
  32-bit GP target the pre-map must propagate the cached local's hi register and push
  both lo and hi lanes; pushing None for hi leaves the scalar with a mapping so a later
  Fill's pair allocation fails ("scalar already has mapping, cannot allocate pair")
  (code).

- 2026-04-27 (9b5e0e16) measurement: after sharing the dead-high-half liveness facts
  and adding ARM32 low-only lowering, ESP32-C6 Mandelbrot holds at 29 fps while Pico 2
  runs at 20 fps on the ARM (Cortex-M33) backend and 21 fps on the Hazard3 (RV32)
  backend (sourced).

## Moves

- 2026-03-21 (cf1c59ed) replaced [[late-i64-legalization-pass]]: the post-lowering pass had to rediscover by storage-flow analysis the i64/i32/float register types and re-pack an inflated GP bank that were already known during lowering; emitting pair MachineIR straight from the lowerer keeps that information in hand and removes the analysis, hi-half tracking, and bank-compaction infrastructure (code).

- 2026-03-20 (4316be53) replaced [[early-semantic-ir-legalization]]: splitting i64 at the semantic-IR level forces planning, LIR, locals, params, returns, and frame layout to all carry a 32-bit pair shape, duplicating arity bookkeeping the lowerer can do alone since it already knows each value's type, so keeping the split inside the lowerer wins by leaving everything above it Wasm-shaped and scalar (sourced)
