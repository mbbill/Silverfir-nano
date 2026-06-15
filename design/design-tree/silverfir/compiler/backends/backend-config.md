- Each backend declares its register-budget policy as a `BackendConfig` preset
  (GP/FP volatile and preserved lane counts plus backend-internal GP scratch,
  argument lanes, call scratch), kept per-arch and selected by the backend's own
  ABI code; the preset is policy, distinct from the hardware ABI facts (which
  physical registers back fixed roles, the callee-saved set, stack-frame layout)
  the backend describes separately.

## Facts

- 2026-03-15 (81cd64b0) rationale: the target-ABI and physical-register-mapping
  facts (which hardware registers back fixed machine roles, the GP/FP machine-reg
  file, scratch-vs-resident FP regs, the callee-saved set the shared
  prologue/epilogue preserves, and the derived stack-frame layout) were kept in
  `abi.rs` as the single source of truth, deliberately insulated from the
  tunable per-bank budget chosen in `config.rs`, so changing the physical
  mapping or save/restore layout could not silently change compiler policy and
  tuning the budget needed no ABI edit (diff).

- 2026-03-24 (413275ef) statement: the register-budget preset
  (`compile_backend_config`) was later moved out of the standalone `config.rs`
  module into the backend's own `abi.rs`, abandoning that deliberate
  separation; the preset stays per-arch and is still policy (GP/FP cached-local
  and transient lane counts), now selected from inside abi.rs (diff).

- 2026-03-27 (12ef375a) rationale: a backend's GP transient budget must be at
  least `(8/gp_unit_bytes)*2 + 1` GP units — the worst-case simultaneous
  pressure of `select(i64, i64, i32)`, since no wasm op takes more than three
  operands; on a 32-bit target this floor is 5 units, so armv7a moved R9 from
  the local-cache bank to the transient bank (4 transients were insufficient),
  with a debug_assert catching under-provisioned budgets at construction (diff).

- 2026-05-14 (f54f5dcf) measurement: arm64 widened its local-call GP
  argument-lane count from 4 to 9 — empirically 9 removes the remaining Lua
  frame-prefix call arguments that 8 leaves behind, while wider values grow
  code in that workload, and the count is kept below the full volatile-dynamic
  lane count so not every volatile lane is forced to be an argument lane (diff).

- 2026-03-15 (21ed5413) statement: the armv7a backend is the first non-arm64,
  non-emulator real consumer of the MachineIR/BackendConfig contract and the
  first 32-bit GP target — it maps the abstract fixed registers (CTX, FP,
  mem0_base, mem0_size) onto EABI registers and builds 32-bit immediates via
  MOVW/MOVT pairs — confirming the contract abstracts a real second ISA rather
  than encoding arm64 assumptions (diff).
