- Whether an entity is imported or local is a two-variant generic enum
  `Kind<Local>`: `Imported(ImportPath)` carries the import path, `Local(L)`
  carries a per-entity local payload struct (function locals/code, etc.)
  (`Kind`).

- Each entity kind instantiates the enum with its own local-payload type
  (`FunctionLocal`, `GlobalLocal`, and unit structs for table/memory).
  Reading a local field means matching `Local(ref local)` and reaching inside.

- Export state is not part of the enum; it rides alongside as a separate
  `exported: Option<String>` field on each entity struct.

## Moves

- 2024-01-25 (49da4692) replaced by [[entity-kind]]: a generic `Kind<Local>` enum
  forced every field read to pattern-match and destructure the variant, and
  the export name had to ride as a separate parallel field; flattening import
  and export into optional fields lets a shared trait serve all four entity
  kinds and turns field access into a plain Option check (diff).
