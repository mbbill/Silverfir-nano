- When the emulator runs a 4-byte-GP-unit module on a 64-bit host it presents a
  synthetic 32-bit virtual address space with fixed base addresses and windows
  for the stack, runtime context, memory/table/function/global/type-canon
  views, and linear memory, translating between host pointers and target-32
  addresses; pointer-width register values stay representable in 32 bits
  during emulation (`Target32AddressSpace`).

## Facts

- 2026-03-19 (c86da6a9) pitfall: carving the 4 GB range into fixed windows
  imposes hard capacity ceilings the real backend does not have (at most 8
  memories and 16 tables, and each memory/table/view must fit its window);
  emu32 validates the module's runtime shape against these windows up front and
  rejects a module that would overflow one, so emu32 failures of this kind are
  address-space-model limits, not engine defects (code).

- 2026-04-22 (219b5e56) statement: because a 32-bit global is read through a
  `raw_ptr` indirection, the globals window splits into a metadata sub-window
  (the `GlobalInst` structs, whose only machine-visible field is `RAW_PTR`) and
  a separate raw-value sub-window (`GLOBAL_RAW_BASE_32`) holding the actual u64
  cells; a global load/store first reads the pointer from the struct, then
  lands in the raw-value window, where ref-typed globals are re-encoded through
  the machine ref representation and sub-word access is masked into the 8-byte
  cell (code).

- 2026-06-20 correction: the c86da6a9 Fact's "at most 8 memories" is stale —
  emu32 now enforces at most 32 memories (`MAX_MEMORY_COUNT_32`); tables stay at
  16 (`MAX_TABLE_COUNT_32`) (code).

- 2026-06-20 correction: the 219b5e56 Fact's separate `GlobalInst`-metadata
  sub-window no longer exists — it was collapsed into an inline globals-ptr tail
  appended to the context window (`globals_ptrs_inline_offset`); only the
  raw-value window remains a separate window, and its constant is spelled
  `GLOBALS_RAW_BASE_32` (plural), not `GLOBAL_RAW_BASE_32` (code).
