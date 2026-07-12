- A reference is a bare usize alias (Ref = usize) with usize::MAX as the null
  sentinel.

- A function/table reference resolved from a module-local index is rebased to
  the module's range start, yielding a module-local-relative index rather than a
  global store index.

## Moves

- 2024-03-12 (35d0c137) replaced by [[plain-index-ref]]: a module-local
  reference index cannot be resolved against the flat global store without
  re-adding the module's range base, so references now carry the global store
  index directly and are wrapped in a newtype that cannot be confused with a raw
  integer (code).
