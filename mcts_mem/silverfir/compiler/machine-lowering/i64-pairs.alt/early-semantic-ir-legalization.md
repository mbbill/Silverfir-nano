- i64 values are split into (lo, hi) 32-bit pairs at the semantic-IR level,
  before planning and LIR lowering.

- A legalized op vocabulary at the semantic-IR and LIR layers carries the pair
  operations explicitly (LirLegalizedOp).

- Function parameters, returns, block signatures, and frame layout represent
  each i64 as two 32-bit units rather than one logical i64 value.

## Facts

- 2026-03-20 rationale: splitting at the semantic-IR level was meant to give the
  planner a true 32-bit GP shape to budget against from the start, instead of
  pair register pressure only becoming visible during lowering (sourced).

- 2026-03-20 (4316be53) statement: the approach was carried into a partial
  implementation on the abandoned_early_legalization branch — a typed semantic
  IR, a legalized LIR op vocabulary, and partial 32-bit lowering — before it was
  abandoned (sourced).

- 2026-03-20 statement: an initial form used explicit carry-out primitives
  (AddCarryOut / AddWithCarry) that needed compiler-private scratch locals to
  reorder the operand stack; these were redesigned into pair-ops consuming the
  natural stack order before the whole approach was abandoned (sourced).

- 2026-03-20 rationale: the recorded stall reason is register pressure in the
  i64-pair path, but no commit or doc pins the precise blocker (uncertain).

## Moves

- 2026-03-20 (4316be53) replaced by [[i64-pairs]]: splitting i64 at the
  semantic-IR level forces planning, LIR, locals, params, returns, and frame
  layout to all carry a 32-bit pair shape, duplicating arity bookkeeping the
  lowerer can do alone since it already knows each value's type, so keeping the
  split inside the lowerer wins by leaving everything above it Wasm-shaped and
  scalar (sourced)
