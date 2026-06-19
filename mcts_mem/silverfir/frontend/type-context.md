- Module-level type identity is resolved through an explicit type context that
  supports recursive types, declared subtyping, and cross-module type
  equivalence (`TypeContext`).

- The module's type section is a unified vector of composite type definitions
  sharing one index space: each entry is a `DefType` wrapping a `CompositeType`
  (function, struct, or array) together with its supertype indices, a finality
  flag, and an optional recursive-group marker.

- Module-scoped type identity lives in a cheaply-clonable `TypeContext` that
  resolves every type index within the owning module and is passed to every
  subtyping check; subtyping over concrete (module-relative) indices never
  runs without its resolution scope.

- An explicit recursive type group (lead byte 0x4E) counts as one entry in the
  type section's vector length but expands to that many `DefType`s; the section
  loop consumes such groups inline, and each group is assigned a unique id at
  parse time.

- Before validating functions and expressions, a type-system phase checks every
  type definition's references (supertype indices and concrete heap-type
  indices in params/results, struct fields, array elements) are within the
  module's type count, reporting out-of-range references as invalid.

- Type equivalence is decided per recursion group: two groups match only when
  their length and the compared index's offset within the group align (the
  group start may differ across modules), and in-group references are compared
  by their position within the active group, matching Wasm GC iso-recursive
  typing.

- Cross-context (cross-module) matching of a concrete type walks the declared
  supertype chain rather than requiring identical rec-group ids.

## Facts

- 2025-10-05 (1e8bc150) rationale: a direct comptype with no 0x4F/0x50 subtype
  wrapper is parsed as final with no supertypes; the explicit-rec-group form is
  the only way to declare mutual type recursion (code).

- 2025-10-07 (b540047f) rationale: the recursive type-equivalence walk assumes
  a pair under comparison is equivalent in order to terminate on cyclic types
  (code).

- 2025-10-07 (b540047f) rationale: a concrete forward type reference (to a
  higher-numbered type) is valid only when the referencing and referenced type
  share the same explicit recursion group; a forward reference outside any rec
  group, or one crossing a rec-group boundary, is invalid — which is what makes
  recursion groups the unit of mutual type recursion (code).

- 2025-10-01 (cf8ff870) rationale: a table definition may carry an initializer
  constant-expression (the GC 0x40 0x00 tabletype-expr form) evaluated to fill
  the table; the table-section parser peeks the first byte to distinguish this
  form from a plain tabletype (code).

## Moves

- 2025-10-05 (5dc7bbc5) replaced [[composite-bare-vec]]: type-index resolution
  is module-local and is needed inside subtyping checks, but a bare owned Vec
  field could not be threaded by value and the optional-context subtyping path
  could silently return a wrong answer when no context was passed; a
  cheaply-clonable TypeContext makes the module scope a first-class value that
  subtyping now always receives (code).

- 2026-04-17 (35dbf09c) replaced [[per-type-pair-equivalence]]: the old
  algorithm compared individual type-index pairs with a visiting-pair guard and
  decided struct/array equivalence by requiring identical rec_group ids, which
  cannot hold across modules whose groups carry different ids and does not
  correctly decide iso-recursive equivalence of mutually recursive GC rec-groups;
  equivalence is now decided per recursion group (start/len/offset must align)
  with in-group references compared by their position within the active group,
  matching Wasm GC iso-recursive typing, and extended to cross-context
  (cross-module) matching that walks the declared supertype chain (code).
