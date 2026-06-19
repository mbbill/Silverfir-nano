- A module entity's import-vs-local status is carried inside the entity itself
  (`LinkableData<ImportSpec, Spec>`) via macro-generated linkable/exportable
  traits; a module stores plain per-kind entity vectors and each entity answers
  is-imported for itself.

## Moves

- 2024-02-17 (613909d4) replaced [[entity-linkage.alt/external-linkable-wrapper]]:
  wrapping every entity vector as Vec<Linkable<T>> pushed the import/local
  match to every call site; moving the import-or-local data inside each entity
  (Function/Table/Memory/Global) with macro-generated Importable/Exportable
  traits lets modules store plain Vec<Function> while each entity answers
  is-imported for itself (code).
