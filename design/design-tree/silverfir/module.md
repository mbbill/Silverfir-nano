- A parsed module holds its definitions in flat per-kind vectors — functions,
  memories, tables, globals, elements, data — plus the optional start-function
  index and binary version.

- Imported and local definitions of the same kind live in one vector. An item's
  import status and its export name are carried as two optional fields shared by
  all kinds (`Kind`), so a single definition can be both imported and exported.

- Kind-local payload is carried as optional fields directly on each definition
  (a function's locals, code, and code offset; a global's init expression) and
  is absent for imported definitions.

- The builder pre-sizes and then shrinks its vectors to fit before sealing the
  module, so the finished module holds no slack capacity.

## Facts

- 2024-01-25 (49da4692) rationale: import-vs-local was reshaped from a sum type
  (where a definition was *either* imported *or* a local-payload variant) into a
  flat record of optional import-path and export-path fields, because a Wasm
  definition can be imported and re-exported at once — a state the either/or sum
  type could not represent (diff).

## Moves

- 2024-01-25 (49da4692) replaced [[kind-sum-type]]: encoding import-vs-local as
  a sum type made imported-and-exported inexpressible, and parked export name on
  a field outside the variant; flat optional import/export fields express both
  states at once (diff).
