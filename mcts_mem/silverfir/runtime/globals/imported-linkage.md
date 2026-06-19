- Every global's value lives behind an `Rc<GlobalCell>` (`UnsafeCell<u64>`)
  reached through a `raw_ptr` indirection, rather than inline in the
  `GlobalInst`; this indirection is what lets two instances share one cell
  (`GlobalCell`).

- Imported globals link by one of two paths: by-value snapshots the exporter's
  current value into a fresh independent cell with no shared mutation; shared-cell
  clones the exporter's `Rc<GlobalCell>`, giving every instance importing the same
  exported global one cell to read and write (`ImportedGlobal`).

- Import-type compatibility is decided in the path's own context: a by-value
  import is type-checked within the importing module's single type context,
  while a shared global carries its exporter's own `TypeContext` and is checked
  cross-context — concrete heap-type equality and subtyping matched against each
  side's canonicalized definition rather than within the importer's context
  alone.

## Facts

- 2026-04-22 (219b5e56) statement: a shared global carried in from another
  instance brings its own exporter `TypeContext`, so import-type compatibility
  can no longer be decided in the importer's context alone — concrete heap-type
  equality (mutable globals, invariant) and subtyping (immutable globals,
  covariant initialization) are checked cross-context, matching each side's
  concrete type index against the other's canonicalized definition; only
  abstract-vs-abstract or scalar cases stay context-free (code).

## Moves

- 2026-04-22 (219b5e56) replaced [[by-value]]: importing a global by snapshotting
  its current value gave each importer an independent copy, so a write through one
  import was invisible through another alias of the same global; storing the value
  in a shared `Rc<GlobalCell>` (UnsafeCell<u64>) and importing it through a
  `raw_ptr` indirection makes every instance that imports the same exported global
  read and write one cell, which is the identity the spec's module-aliasing tests
  (instance.wast) require (code).
