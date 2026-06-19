- On guard-page-enabled builds a wasm linear memory is backed by an mmap
  reservation (8 GB + 64 KB per memory) whose committed (RW) prefix tracks the
  current size and whose remainder is mapped PROT_NONE, with the base pointer
  stable across grow (grow only re-mprotects, never reallocates)
  (`GuardPageMemory`).

- On guarded 64-bit configurations many linear-memory accesses skip explicit
  bounds checks and rely on guard pages to fault out-of-bounds accesses, with
  explicit checks retained only where guard pages cannot cover the case (such
  as multiword GP checks). Guard pages are a build-time-selected capability
  (`sf_has_guard_pages`).

- A per-memory page ceiling is enforced both at instantiation for the initial
  page count and at `memory.grow`, and is checked before compilation starts
  even though JIT memory backing is materialized after compilation.

## Facts

- 2026-03-15 (3b6d2f59) rationale: the 8 GB + 64 KB reservation is sized so the
  wasm32 maximum base address (2^32-1) plus the maximum static offset (2^32-1)
  plus the maximum access width (16 bytes) still lands inside the reservation,
  so no out-of-bounds access can escape the guard region and the JIT can drop the
  explicit bounds check entirely on this path (code).

- 2026-03-15 (3b6d2f59) pitfall: guard-page memory backing (a raw mmap region) is
  not cloneable, so `MemInst`'s Clone drops the guard backing and falls back to an
  empty Vec; reads/writes route through memory_ptr/memory_len so both the Vec path
  and the guard path share one accessor surface, but a clone of a guarded memory
  silently loses its guarded backing (code).

- 2026-04-21 (b206d2aa) rationale: at `memory.grow` the configured
  `wasm_memory_max_pages` ceiling is folded into effective_max =
  min(declared_max, runtime_cap) and on exceedance returns the standard
  memory.grow failure sentinel (u32::MAX / u64::MAX, -1) as a normal Ok value to
  the running program, not Unlinkable and not a trap — the page-cap enforcement
  point at instantiation is described on [[page-quota]] (code).
