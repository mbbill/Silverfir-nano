- The call_indirect signature check compares the callee function-view's stored
  raw module type index against the call site's expected raw type index, emitted
  as an immediate.

## Moves

- 2026-03-13 (4369d7f6) replaced by [[indirect-type-check]]: raw type indices
  cannot express Wasm structural type equivalence across the module's type space,
  so the signature check must compare cached canonical equivalence-class ids: the
  context caches a per-type canonical table and each function view stores its
  signature's canonical id (code).
