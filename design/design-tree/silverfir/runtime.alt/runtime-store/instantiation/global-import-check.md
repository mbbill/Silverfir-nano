- A global import's link-time type check branches on mutability: a mutable
  global requires the exported and imported value types to match exactly
  (invariant), an immutable global requires the exported type to be a subtype
  of the imported type (covariant) resolved against the exporting module's
  type context; mutability itself must match either way.

## Moves

- 2025-10-05 (b3ce11ba) replaced [[global-import-check.alt/exact-equality-global-check]]:
  the old exact-equality check (is_compatible_with: exported==imported or
  Unknown) could not express the immutable-global covariant rule, which requires
  the exported type to be a subtype of the imported type with concrete
  (module-relative) type indices resolved in the exporting module's namespace;
  verify_import's signature could carry no TypeContext to do that resolution, so
  valid immutable-global subtype imports were rejected (diff).
