- Each module entity vector stores `Linkable<T>`, an external enum wrapper that
  is either Imported (path, export name, import property) or Local (export name,
  property).

- Import-vs-local is queried by matching on the `Linkable` wrapper at the call
  site; the wrapped entity type is split into an import-property type and a local
  type via a `LinkableType` associated type.

## Moves

- 2024-02-15 (3906283c) replaced [[external-linkable-wrapper.alt/embedded-linkage-field]]:
  a single embedded linkage struct forced every entity to carry optional
  local-only fields (code, locals, init-expr) that imports never use; splitting
  import and local into a sum type with separate property types lets an imported
  function hold only its type and a local hold non-optional code, removing the
  Option/expect panics (diff).

- 2024-02-17 (613909d4) replaced by [[entity-linkage]]: wrapping every entity
  vector as Vec<Linkable<T>> pushed the import/local match to every call site;
  moving the import-or-local data inside each entity
  (Function/Table/Memory/Global) with macro-generated Importable/Exportable
  traits lets modules store plain Vec<Function> while each entity answers
  is-imported for itself (diff).
