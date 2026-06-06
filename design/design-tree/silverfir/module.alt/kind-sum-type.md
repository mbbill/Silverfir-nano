- Each definition's import status is a sum type (`Kind<LocalType>`) with two
  variants: an imported variant carrying the import path, and a local variant
  carrying that kind's local payload (a function's locals and code, a global's
  init expression).

- Export status is a separate optional name field sitting alongside the sum
  type, so a definition reaches its local payload only through the local
  variant.

## Moves

- 2024-01-25 (49da4692) replaced by [[module]]: encoding import-vs-local as a
  sum type made imported-and-exported inexpressible, and parked export name on a
  field outside the variant; flat optional import/export fields express both
  states at once (diff).
