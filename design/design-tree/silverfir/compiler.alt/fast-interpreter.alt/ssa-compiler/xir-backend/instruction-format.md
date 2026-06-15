- A built XIR instruction is a handler pointer plus three uniform 64-bit
  immediate fields, a single fixed `#[repr(C)]` layout shared with the C
  trampoline; any datum — constants, branch targets, memory/table indices — is
  encoded into those interchangeable fields rather than into physically distinct
  typed fields.

- The packing and unpacking of each instruction's immediate fields is defined
  once per format as paired encode/decode functions in a single module, used by
  both lowering and the handlers, keeping the two from drifting (`format`).

- Operands too large for the three immediates (call argument/result lists with
  types, br_table jump tables, target heap types) live in pinned `#[repr(C)]`
  side-table structures owned by the lowered function for its lifetime and
  referenced by a raw pointer stored in an immediate.

- The lowered function holds its instruction stream and metadata pinned and
  immutable: the instruction stream references its own branch targets and the
  side-table metadata through raw self-referential pointers; the whole
  structure must never move after construction and is cached behind a refcount.

## Facts

- 2025-10-16 (62009f82) rationale: branch targets are resolved in two stages —
  lowering emits instructions carrying a signed offset to the target rather than a
  raw pointer, because the growing instruction Vec reallocates and any pointer
  taken into it before finalization would dangle; offsets are converted to
  absolute pointers only after the final array is boxed and pinned (diff).

- 2025-10-15 (0ffe0f96) statement: the lowered function is a self-referential
  struct (instruction immediates point into the metadata arrays; the br_table
  tables point back into the instruction stream), which Rust cannot express
  without a self-ref crate, so the design relies on Pin for no-move plus a boxed
  slice for no-remove, with the instruction stream declared before the metadata so
  drop order tears down the pointers' users first (diff).

- 2026-02-08 (da03882b) pitfall: the pin is load-bearing — the dispatch loop
  references the instructions through raw self-referential pointers, so any move
  or other invalidation of the pinned metadata is undefined behavior in the
  running dispatch loop; the compiled function must stay pinned (and cached behind
  a refcount) for the lifetime of any execution that can reach it (diff).

## Moves

- 2025-10-13 (10a69247) replaced [[leaked-bulk-op-side-table]]: the heap
  side-table allocated (and leaked) one Box per bulk-op instruction and added a
  pointer dereference per execution; the three register indices fit in the
  instruction's own b/c immediate fields (diff).

- 2025-10-15 (0ffe0f96) replaced [[growable-vec-metadata]]: the instruction stream
  holds raw pointers into the metadata (and br_table tables hold raw pointers back
  into the instruction stream), so a Vec whose pop/clear could remove entries and
  whose buffer could move leaves dangling pointers; Box<[Pin<Box<T>>]> makes the
  no-move, no-remove contract type-enforced (diff).

- 2025-10-21 (dc951039) replaced [[branch-offset-mixed-width-inst]]: the earlier
  instruction carried a special Option branch-offset field alongside split
  32/32/64-bit immediates, forcing each encoding to know which physical field its
  data lived in; collapsing to a handler pointer plus three interchangeable 64-bit
  immediates lets any datum (constants, branch targets, memory/table indices) be
  encoded uniformly and matches a single fixed repr(C) layout shared with the C
  trampoline (diff).
