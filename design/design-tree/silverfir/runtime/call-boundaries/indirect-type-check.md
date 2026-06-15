- A `call_indirect` signature check compares cached canonical
  equivalence-class type ids, not raw module type indices: each context caches a
  per-type canonical table and each function view stores its signature's
  canonical id, which the dispatch gate loads and compares against the call
  site's expected canonical id (`build_call_indirect_type_check_block`,
  `type_canon`).

## Moves

- 2026-03-13 (4369d7f6) replaced [[raw-type-index-check]]: raw type indices
  cannot express Wasm structural type equivalence across the module's type space,
  so the signature check must compare cached canonical equivalence-class ids: the
  context caches a per-type canonical table and each function view stores its
  signature's canonical id (diff).
