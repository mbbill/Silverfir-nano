- The live transient window tracks each value's exact Wasm type and budgets GP
  and FP banks separately; floats stay in FP registers end-to-end instead of
  losing their type at slot reloads and being recreated in GP transients
  (`count_live_bank_budget_units`).

## Facts

- 2026-03-15 (5828c3c2) statement: per-bank residency pressure is handled by
  spilling/reloading the deepest resident entry of the same bank — the live
  window stays a contiguous-suffix spill model (spill_depth, now bank-aware) —
  rather than by one combined lane count; the typed-residency proposal
  (docs/NATIVE_TYPED_RESIDENCY_DESIGN.md, deleted b9b02d80) named an
  oldest-by-last-use eviction order as its Open Design Question 3, while the
  recommendation and the shipped code kept the deepest/contiguous policy as the
  correct, deterministic shape that falls out of the existing prefix spill/fill
  machinery (sourced).

- 2026-04-08 (47daba23) pitfall: the prefix planner kept the entire fallthrough
  stack slice (stack_drop + arity + condition) live across a `br_if`, but `br_if`
  directly consumes only the condition; the taken edge needs only the branch
  payload bound and fallthrough-only prefix values can stay spilled and be
  refilled by the next prefix, so keeping the whole slice live overstated 32-bit
  register pressure on stack-heavy patterns and overflowed the gp32/emu32 budget —
  the keep-live count is reduced to arity + 1 (code).

## Moves

- 2026-03-15 (5828c3c2) replaced [[untyped-single-bank]]: the untyped
  single-bank live window could not keep float values in FP registers: floats
  lost their type at slot reloads and were recreated in GP transients; tracking
  each value's exact Wasm type and budgeting GP and FP banks separately lets
  floats stay FP end-to-end and removes representation-driven GP/FP churn (code)
