- A concrete GC cast target matches the object type when a TypeContext reports the
  two type indices canonically equivalent, walking the object type's supertype
  chain.

- Type equivalence is decided by canonicalizing both type indices through a freshly
  built TypeContext.

## Moves

- 2025-10-14 (6f3e5855) replaced [[cast-by-index-identity]]: exact index identity
  matched only the literal type index and missed two structurally-equal concrete
  types declared at different indices; canonicalizing both sides through a
  TypeContext makes equal types match regardless of declaration index (diff).

- 2025-10-14 (f2489323) replaced by [[gc-ops]]: the cast match abandons TypeContext
  canonicalization for a hand-rolled structural-equivalence recursion that compares
  composite shape (struct fields, array element, supertypes) directly, guarded by a
  recursion-depth cap to avoid infinite descent on cyclic types (diff).
