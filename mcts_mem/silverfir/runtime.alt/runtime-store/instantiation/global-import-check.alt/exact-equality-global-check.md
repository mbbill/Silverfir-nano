- A global import's value types are checked by exact equality only
  (`is_compatible_with`: exported equals imported, or either is Unknown), with
  no dependence on mutability and no way to accept an exported type that is a
  proper subtype of the imported type.

- Mutability of an imported global must match the export exactly.

## Moves

- 2025-10-05 (b3ce11ba) replaced by [[global-import-check]]: the old
  exact-equality check (is_compatible_with: exported==imported or Unknown)
  could not express the immutable-global covariant rule, which requires the
  exported type to be a subtype of the imported type with concrete
  (module-relative) type indices resolved in the exporting module's namespace;
  verify_import's signature could carry no TypeContext to do that resolution, so
  valid immutable-global subtype imports were rejected (code).
