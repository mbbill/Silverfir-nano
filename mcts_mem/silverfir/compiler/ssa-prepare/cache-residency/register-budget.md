- The backend budget the cache/transient planner optimizes against is expressed
  in GP and FP budget units with a per-backend GP unit size (`gp_unit_bytes`: 8 on
  64-bit, 4 on armv7a); a true `i64` is charged as two GP units on a 32-bit GP
  target while 64-bit behavior is unchanged.

- The fixed machine-register roles are shared MachineIR constants, not
  per-backend counts; `BackendConfig` carries a single per-bank dynamic budget
  (`gp_dynamic_budget` / `fp_dynamic_budget`) that transients and cached locals
  draw against jointly, each in its bank's budget units, rather than separate
  transient and cached-local budgets.

## Moves

- 2026-03-18 (3778de1c) replaced [[plain-counts]]: counting physical registers
  cannot charge a true i64 its real cost on a 32-bit GP target where it occupies
  a register pair; adding gp_unit_bytes (8 on 64-bit, 4 on armv7a) and reckoning
  the transient and cached-local budgets in GP budget units lets the planner
  charge an i64 as two GP units on 32-bit while leaving 64-bit behavior
  unchanged (code)
