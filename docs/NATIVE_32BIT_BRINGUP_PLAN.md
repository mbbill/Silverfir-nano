# Native 32-Bit Bring-Up Plan

This document captures the staged plan for fixing the native pipeline so true
32-bit backends can be implemented cleanly without regressing the existing
64-bit backends.

The immediate target is `armv7a`, but the shared design work in phase 1 is for
all 32-bit native backends.

## Ground Rules

- Keep the canonical native frame contract: one Wasm value per 8-byte frame
  slot.
- Do not treat "stored in an 8-byte slot" as equivalent to "must live in a
  64-bit GP register".
- Above MachineIR, keep semantic Wasm types.
- At and below MachineIR, distinguish word-sized GP values from true 64-bit GP
  values.
- Land the work in small, testable slices. Validate after every slice.

## Validation Gate

Every phase 1 step must pass the same validation loop before moving on:

1. `cargo test -p sf-nano-core --features micro-jit --lib`
2. `cargo run --bin sf-nano-spectest -- --backend native`
3. `python3 benchmarks/wasi/run_tests.py --exec /abs/path/to/target/release/sf-nano-cli --cli-args "--backend native"`

Validation expectations:

- `arm64`: all commands pass, with no benchmark regression beyond normal noise
  versus [`benchmarks/wasi/RESULTS.md`](/Users/bytedance/Dev/Silverfir-nano/benchmarks/wasi/RESULTS.md)
- `x86_64`: spectest and `run_tests.py` pass under Rosetta when needed; score
  parity is not required for phase 1

## Phase 1

Goal: fix shared infrastructure above real ARMv7A code emission while keeping
`arm64` and `x86_64` correct.

### Step 0: Baseline

- Record current `cargo test`, native spectest, and `run_tests.py` results on
  the active `arm64` host.
- Record current `x86_64` native spectest and `run_tests.py` results under
  Rosetta.
- Treat the recorded outputs as the regression baseline for this phase.

### Step 1: Backend / Planner Width Contract

- Extend `BackendConfig` with explicit GP register width metadata.
- Thread that width into `PlanConfig`.
- Keep default behavior unchanged on current 64-bit hosts.
- Make architecture presets explicit about GP register width:
  - `arm64`: 8 bytes
  - `x86_64`: 8 bytes
  - `armv7a`: 4 bytes

This step should not change planning behavior yet.

### Step 2: Width-Aware GP Transient Budgeting

- Change transient pressure tracking from "number of non-float values" to GP
  register-units.
- Cost model on 32-bit:
  - `i32`: 1 GP unit
  - refs / pointer-width GP values: 1 GP unit
  - true `i64`: 2 GP units
- Preserve current behavior on 64-bit.

### Step 3: Width-Aware Cached-Local Budgeting

- Change cached-local selection from top-N-by-count to selection under a GP-unit
  budget.
- Keep FP cached-local selection unchanged.
- Make `i64` locals consume two GP units on 32-bit.

### Step 4: Preserve Semantic Types Above MachineIR

- Keep using existing `ValueType` in planning and LIR.
- Do not collapse refs into `i64`.
- Make any new shared helpers consume semantic `ValueType` rather than a reduced
  four-case enum.

### Step 5: Add Machine Storage Typing

- Extend the native handoff / MachineIR contract so GP values can be classified
  as:
  - word-sized GP
  - true 64-bit GP
  - `f32`
  - `f64`
- Ensure block params, edge args, and generic moves/selects carry enough typing
  information for later legalization.

### Step 6: Separate Pointer-Width Ops from True `i64`

- Audit lowering so pointer/ref/address operations use pointer-width semantics,
  not generic `I64` / `U64`.
- Keep true Wasm `i64` operations distinct from:
  - refs
  - indices
  - addresses
  - pointer-sized cached lengths and counts

### Step 7: Fence the MachineIR Optimizer

- Keep the current local peephole optimizations.
- Prevent the compare-branch fusion pass from creating 32-bit-hostile fused
  `I64` branch conditions before legalization.
- Legalization must see a representation it can lower mechanically.

### Phase 1 Exit Criteria

- `arm64`:
  - `cargo test -p sf-nano-core --features micro-jit --lib` passes
  - native spectest passes
  - `run_tests.py` passes
  - no meaningful regression versus
    [`benchmarks/wasi/RESULTS.md`](/Users/bytedance/Dev/Silverfir-nano/benchmarks/wasi/RESULTS.md)
- `x86_64`:
  - native spectest passes
  - `run_tests.py` passes
  - benchmark score does not matter in this phase
- MachineIR and lowering contracts are ready for 32-bit legalization without
  backend-specific type guessing

## Phase 2

Goal: implement correct 32-bit lowering from MachineIR to ARMv7A machine code.

### Step 1: 32-Bit Legalization

- Add a legalization pass for true 64-bit GP values.
- Split block params and edge args consistently.
- Rewrite true `i64` ops into forms legal for 32-bit backends.
- Consume GP register-units already budgeted in phase 1.

### Step 2: ARMv7A Lowering for Legalized MachineIR

- Implement pair-aware ARMv7A lowering for:
  - moves
  - load/store
  - add/sub with carry / borrow
  - compare / branch
  - shifts
  - multiply
  - conversions and reinterprets

### Step 3: Independent ARMv7A Backend Fixes

- Fix `select` aliasing / condition clobber separately from legalization.
- Fix likely unaligned `f64` load/store SIGBUS paths.
- Re-audit edge parallel moves and GP/FP transfer paths once machine storage
  typing lands.

### Step 4: ARMv7A Validation

- Run ARMv7A native spectest regularly during bring-up.
- Run `benchmarks/wasi/run_tests.py` regularly during bring-up.
- Inspect emitted LIR, MachineIR, and ARM32 assembly for representative
  `i64`, `f64`, memory, and `select` cases.

### Phase 2 Exit Criteria

- ARMv7A native spectest passes
- ARMv7A `run_tests.py` passes
- No known `SIGBUS` in the floating-point benchmarks
- Representative `i64`, `f64`, and `select` cases lower to correct ARM32 code

## Known Non-Issues

- `slot_offset_bytes(slot) = slot * 8` is not itself the architectural bug.
  The canonical frame is intentionally 8-byte slotted.
- The bug is conflating "8-byte frame storage" with "64-bit GP register
  semantics everywhere in the pipeline".
