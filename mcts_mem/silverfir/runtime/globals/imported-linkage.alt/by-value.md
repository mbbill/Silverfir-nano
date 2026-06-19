- A `GlobalInst` stored its value inline as `raw: u64` in the struct, with no
  shared backing cell; two instances importing the same exported global each
  materialized their own fresh `GlobalInst` and could never share one cell.

- `GlobalInst` exposed its value at the inline `RAW` field offset; generated code
  read and wrote the global directly at that offset.

## Moves

- 2026-04-22 (219b5e56) replaced by [[imported-linkage]]: importing a global by
  snapshotting its current value gave each importer an independent copy, so a
  write through one import was invisible through another alias of the same global;
  storing the value in a shared `Rc<GlobalCell>` (UnsafeCell<u64>) and importing
  it through a `raw_ptr` indirection makes every instance that imports the same
  exported global read and write one cell, which is the identity the spec's
  module-aliasing tests (instance.wast) require (code).
