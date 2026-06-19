- A local or imported linear memory's backing buffer is allocated to its initial
  page count at instance construction, before module compilation runs.

## Moves

- 2026-04-27 (3911e481) replaced by [[lazy-backing]]: holding the full
  linear-memory allocation through compilation competes with the compiler for RAM
  on memory-constrained targets, so on non-guard-page JIT builds the memory
  backing is created unallocated and materialized only after
  ensure_module_compiled returns (code).
