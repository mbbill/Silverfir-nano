- `BackendConfig` carries five register-count fields (ctx, fp, tmp, hot-local,
  tos) that each backend populates.

- The lowering register file computes its fixed partition by laying out
  runtime-base, frame-base, local-cache, transient, and temp regions from those
  counts; even the runtime-context and frame-pointer register numbers are
  backend-supplied rather than fixed.

## Moves

- 2026-03-13 (4ae8509d) replaced by [[plain-counts]]: every backend reserved the
  same fixed ABI roles, so a per-backend ctx/fp/tmp/tos count was redundant
  ceremony; the fixed roles (ctx, fp, two scratch) became shared MachineIR
  constants and BackendConfig shrank to the two budgets a backend actually
  varies — cached-local count and live-lane count (code)
