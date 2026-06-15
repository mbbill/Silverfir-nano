- Type equivalence recurses over individual (idx1, idx2) type-index pairs,
  guarding against cycles with a visiting set of pairs.

- Struct and array types are equivalent only when their two rec_group ids are
  equal; func types compare structurally regardless of group.

- Cross-module/cross-context concrete-type matching lives in a separate
  concrete_type_matches helper in gc_type_check.rs that calls a per-pair
  cross-module equivalence routine.

- Type equivalence does not consult declared supertypes or finality.

## Moves

- 2026-04-17 (35dbf09c) replaced by [[type-context]]: the old algorithm
  compared individual type-index pairs with a visiting-pair guard and decided
  struct/array equivalence by requiring identical rec_group ids, which cannot
  hold across modules whose groups carry different ids and does not correctly
  decide iso-recursive equivalence of mutually recursive GC rec-groups;
  equivalence is now decided per recursion group (start/len/offset must align)
  with in-group references compared by their position within the active group,
  matching Wasm GC iso-recursive typing, and extended to cross-context
  (cross-module) matching that walks the declared supertype chain (diff).
