- A standalone post-lowering pass (`MachineModule::legalize`) rewrites true GP i64
  values into paired GP-word MachineIR ops, run only on 32-bit GP targets
  (gp_reg_width == 4) after MachineIR is already lowered and allocated.

- With LIR-level type information already lost at MachineIR, the pass first runs a
  fixed-point storage-flow dataflow analysis (`analyze_32bit_storage_flow`) to
  rediscover which machine registers hold i64 versus i32 versus float values.

- Each true i64 register gets a companion high-half register tracked post-hoc
  (`persistent_hi` / `current_hi`), plus a fixed pool of legalizer-private scratch
  registers, expanding the program's register count beyond what the lowerer allocated.

- After rewriting, a GP-bank compaction step packs the inflated register set
  (originals + hi-halves + scratch) back into the backend's physical register budget.

## Moves

- 2026-03-21 (cf1c59ed) replaced by [[i64-pairs]]: the post-lowering pass had to rediscover by storage-flow analysis the i64/i32/float register types and re-pack an inflated GP bank that were already known during lowering; emitting pair MachineIR straight from the lowerer keeps that information in hand and removes the analysis, hi-half tracking, and bank-compaction infrastructure (code).
