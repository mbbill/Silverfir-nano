- On non-guard-page JIT builds a linear memory is created with an empty backing
  buffer at instance construction and its initial pages are materialized only
  after module compilation completes, before active data segments are applied
  (`MemInst::new_unallocated`, `ensure_allocated`).

## Moves

- 2026-04-27 (3911e481) replaced [[eager-backing]]: holding the full
  linear-memory allocation through compilation competes with the compiler for RAM
  on memory-constrained targets, so on non-guard-page JIT builds the memory
  backing is created unallocated and materialized only after
  ensure_module_compiled returns (diff).
