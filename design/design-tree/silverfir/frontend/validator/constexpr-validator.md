- Global, element, and data initializers are type-checked by walking the
  constant expression's opcodes against a small value stack: each init/offset
  expression must leave exactly one value, element and data offsets must be i32
  (i64 for a 64-bit memory/table), and passive-element expressions are
  restricted to ref.null, ref.func, the numeric constants (i32/i64/f32/f64),
  global.get, the 0xFB GC ops, and end.

- A constant expression may read any immutable global (imported or
  module-defined) and rejects mutable ones; the lone exception is a table
  initializer expression, which may read only imported globals.

- The validator type-checks a narrower 0xFB GC admit-set than the decode
  boundary accepts: struct.new, struct.new_default, ref.i31, and the
  anyref/externref conversions (any.convert_extern, extern.convert_any); any
  other 0xFB opcode reaching the validator — including the array.new* ops the
  decoder admits — is rejected as invalid.

- Per-site constant-expression validation carries a `ValidationContext` struct
  rather than a fixed parameter pair; it holds the validating global's own
  index and enforces that a global init expression references only globals
  defined earlier than itself.

## Facts

- 2025-06-22 (9e353093) conformance: ref.func in a constant expression is
  validated to reference a function declared in some element segment; the check
  is enforced only when element segments exist, and InitExprs-form element
  segments are treated leniently as declaring everything — a deliberately
  partial form pending a proper declared-functions context (diff).

- 2025-10-01 (6c1a2b67) rationale: the global-reference rule was relaxed for
  wasm 3.0 — module-defined immutable globals are admissible because they are
  validated after all globals are parsed; only mutable globals are rejected
  (table initializers keep the stricter imported-only rule) (diff).

- 2025-10-01 (cf8ff870) rationale: a table initializer's stricter
  imported-only-globals rule is threaded through the const-expr validator as an
  only-imported-globals context flag rather than a separate validation path
  (diff).

## Moves

- 2025-10-05 (439fd90d) replaced [[two-boolean-constexpr-validation]]: the
  two-boolean parameter pair (is_passive, only_imported_globals) could not carry
  the validating global's own index needed for the earlier-than-self rule (a
  global init expression may reference only globals defined earlier than itself;
  forward/self reference is 'unknown global'), which the context struct adds as
  validating_global_index: Option<usize> (diff).
