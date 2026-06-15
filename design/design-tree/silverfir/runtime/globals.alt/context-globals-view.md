- The runtime context holds a `globals_view` field — a `{ base: *mut GlobalInst,
  len }` pair pointing at the module's `GlobalInst` array — refreshed from the
  store on every `refresh_cached_views`.

- JIT-emitted `global.get`/`global.set` reaches a global's backing cell through
  three machine loads: load the `globals_view` base pointer from the context,
  then an indexed load of the per-global `raw_ptr` from the `GlobalInst` array
  (element stride `size_of::<GlobalInst>()`, field offset
  `global_offset::RAW_PTR`), then dereference the loaded `*mut u64`.

- The runtime context is a fixed-size `#[repr(C)]` struct allocated as a plain
  value (no trailing variable-length data); its ABI layout exposes
  `globals_view_offset` / `globals_view_base_offset` / `globals_view_len_offset`
  for the view triple.

## Moves

- 2026-04-24 (9eee2860) replaced by [[globals]]: reaching a global through the
  context's globals-view forced JIT global.get/global.set to chase a view pointer
  first — load the view base from the context, then an indexed load of the
  per-global raw_ptr out of the GlobalInst array (stride
  size_of::<GlobalInst>()), then dereference — three loads per access through a
  backend-indirect address; caching the per-global raw pointers in an inline
  [*mut u64; globals_len] array pinned at the context tail (constant offset
  size_of::<NativeContext>() from runtime_base) lets lowering reach the raw_ptr
  with a single indexed load at a compile-time-constant offset followed by one
  deref, two loads with no view-pointer indirection (diff).
