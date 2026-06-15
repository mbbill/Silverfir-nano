- JIT-emitted `global.get`/`global.set` reaches a global's backing cell through
  an inline `[*mut u64; globals_len]` array pinned at a compile-time-constant
  offset at the runtime-context tail: one indexed load of the per-global raw
  pointer followed by one dereference, with no view-pointer indirection
  (`globals_ptrs_tail`).

## Moves

- 2026-04-24 (9eee2860) replaced [[context-globals-view]]: reaching a global
  through the context's globals-view forced JIT global.get/global.set to chase a
  view pointer first — load the view base from the context, then an indexed load
  of the per-global raw_ptr out of the GlobalInst array (stride
  size_of::<GlobalInst>()), then dereference — three loads per access through a
  backend-indirect address; caching the per-global raw pointers in an inline
  [*mut u64; globals_len] array pinned at the context tail (constant offset
  size_of::<NativeContext>() from runtime_base) lets lowering reach the raw_ptr
  with a single indexed load at a compile-time-constant offset followed by one
  deref, two loads with no view-pointer indirection (diff).
