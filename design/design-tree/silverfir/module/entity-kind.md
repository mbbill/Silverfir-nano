- Whether an entity (function, table, memory, global) is imported or local is
  carried by a shared inline sub-struct holding an optional import path and an
  optional export name; import-ness is "import path present", export-ness is
  "export name present" (`Kind`).

- The local-only payload (a function's locals and code, a global's init
  expression) lives in optional fields directly on the entity struct, present
  only for local instances. Reading a local-only field on an imported entity
  is a programmer error and panics.

- The shared accessors (is-imported, import-path, set-export-path, …) are
  factored into a trait each entity implements by exposing its `Kind`
  (`KindTrait`), so the import/export logic is written once.

## Moves

- 2024-01-25 (49da4692) replaced [[kind-enum]]: a generic `Kind<Local>` enum
  forced every field read to pattern-match and destructure the variant, and
  the export name had to ride as a separate parallel field; flattening import
  and export into optional fields lets a shared trait serve all four entity
  kinds and turns field access into a plain Option check (diff).
