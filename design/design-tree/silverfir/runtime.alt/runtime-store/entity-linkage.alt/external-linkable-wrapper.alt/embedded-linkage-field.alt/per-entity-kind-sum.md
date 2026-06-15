- Each module entity's import-vs-local status is a generic sum type
  `Kind<LocalType>` (`Imported(ImportPath)` | `Local(LocalType)`); an imported
  entity cannot carry local data and a local entity cannot carry an import
  path — the two are mutually exclusive by construction.

- Per-entity local data lives inside the `Local` variant's payload:
  `FunctionLocal{locals, code, max_stack_height}` for functions,
  `GlobalLocal{init_expr}` for globals, and unit `TableLocal`/`MemoryLocal` for
  tables and memories.

- An entity's export name is a separate `Option<String>` field on the entity,
  independent of its `Kind`.

## Moves

- 2024-01-25 (49da4692) replaced by [[embedded-linkage-field]]: the per-entity
  Kind<LocalType> sum type made import-and-local mutually exclusive by
  construction so an imported entity could not hold local data and vice versa;
  flattening to a shared Kind{import_path,export_path} plus per-entity
  Option locals/code/init_expr makes that illegal state representable and
  trades the type-enforced exclusivity for
  import_path()/export_path()/locals()/code() accessors that panic on misuse
  (diff).
