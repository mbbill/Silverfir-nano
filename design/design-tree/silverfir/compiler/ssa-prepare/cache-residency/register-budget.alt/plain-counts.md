- The backend budget is a flat set of physical register/lane counts
  (`gp_local_cache_count`, `gp_lane_count`, `fp_local_cache_count`,
  `fp_lane_count`) with no notion of per-value width: every GP value costs one
  register; a true `i64` is charged the same as an `i32` on every target.

## Moves

- 2026-03-13 (4ae8509d) replaced [[per-backend-fixed-roles]]: every backend
  reserved the same fixed ABI roles, so a per-backend ctx/fp/tmp/tos count was
  redundant ceremony; the fixed roles (ctx, fp, two scratch) became shared
  MachineIR constants and BackendConfig shrank to the two budgets a backend
  actually varies — cached-local count and live-lane count (diff)

- 2026-03-18 (3778de1c) replaced by [[register-budget]]: counting physical
  registers cannot charge a true i64 its real cost on a 32-bit GP target where
  it occupies a register pair; adding gp_unit_bytes (8 on 64-bit, 4 on armv7a)
  and reckoning the transient and cached-local budgets in GP budget units lets
  the planner charge an i64 as two GP units on 32-bit while leaving 64-bit
  behavior unchanged (diff)
