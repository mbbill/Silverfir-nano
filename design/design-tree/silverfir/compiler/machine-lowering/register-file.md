- The lowering register file is a fixed partition derived from the backend's lane
  counts: one runtime-base register, one frame-base register, then the two pinned
  mem0 views, then ordered dynamic banks of local-cache and transient registers,
  addressed by index, never by physical name (`MachineRegFile`).

- The partition boundaries derive solely from the backend config carried on the
  machine module, with a const assertion tying the fixed-register count to the
  config; no second source of the layout exists to drift against.

- A dynamic register's role (cached local vs transient stack value) is explicit
  per-program-point owner metadata, not a property of its number; one bank serves
  either purpose wherever a block needs it (`DynamicOwnershipTracker`).

- A contiguous FP-only bank holds floating-point block-local SSA values, which never
  live in GP registers; a machine block parameter declares whether it carries a GP or
  an FP value of a given lane width, telling continuation-edge transfers each value's
  bank (`first_fp_reg`).

- Memory 0's base and size are pinned in two fixed machine registers for the whole
  function; ad hoc scratch is borrowed from transient lanes that no live value
  currently owns, never from a dedicated temp pool (`MACHINE_MEM0_BASE_REG`).

## Facts

- 2026-03-14 (3ad22658) rationale: the FP-only transient bank exists so float-heavy
  code is not forced to bounce every transient through GP registers; a tiny FP bank
  removes avoidable GP/FP churn while staying inside the fixed-budget lowering
  model — it is not a second local-cache system and not ABI-visible persistent
  state (author).

- 2026-03-15 (b9b02d80) statement: a fully-specified proposal to thread Wasm value
  types through decode/prepare/LIR/lowering so floats stay in FP transients (typed
  stack entries with per-bank GP/FP residency and separate per-bank budgets),
  motivated by the c-ray gap where spilled floats reload as untyped u64 and bounce
  through GP registers with fmov shuffles, was abandoned; the pipeline stayed
  single-bank and untyped and the c-ray FP pressure was instead addressed by widening
  the FP register banks — full proposal in [[register-file.fact/typed-residency-proposal]]
  (inferred → Q8).

- 2026-03-15 (5828c3c2) statement: correction to the 2026-03-15 (b9b02d80) entry
  above — the typed FP residency proposal was not abandoned; it was implemented
  the next day in 5828c3c2 ("fp is in", which threads exact Wasm value types
  through decode/prepare/LIR and gives separate GP/FP transient banks) and remains
  the live design (now in middle/rewrite/state.rs: live_types/type_stack +
  gp_live_budget/fp_live_budget; budget.rs `count_live_bank_budget_units` routes
  F32/F64/V128 to the fp bank). The doc (docs/NATIVE_TYPED_RESIDENCY_DESIGN.md,
  deleted b9b02d80) was removed because it had been realized in code, not shelved;
  widening the FP banks (f94559c "use all fp registers") was an additional tuning
  step on top of the implemented typed pipeline, not a substitute for it —
  resolves Q8 (author).

- 2026-03-14 (7b489b53) statement: the fixed machine roles (ctx, fp, and pinned
  views such as mem0_base/mem0_size) are ABI facts rather than tuning knobs; mem0
  is pinned because memory 0 is the hottest runtime view, the context stays the
  source of truth, local native calls keep the pinned view live, and only
  helper-backed boundaries reload it after return (diff).

## Moves

- 2026-03-13 (110b77f0) replaced [[general-scratch-registers]]: memory 0 is the hottest runtime view, so the two fixed scratch slots are repurposed to pin mem0_base/mem0_size for the whole function; ad hoc scratch is instead borrowed from transient lanes that no live value currently owns (diff).

- 2026-04-04 (ea0cf447) replaced [[static-partitioned-register-file]]: a register fixed as cache-only or transient-only by its number could not be reassigned to whichever purpose a block needed; making each dynamic register's role explicit owner metadata lets one bank serve cached locals and transient stack values per program point (diff).
