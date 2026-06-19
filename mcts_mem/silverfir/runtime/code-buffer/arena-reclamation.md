- The hosted code-arena default capacity is a single cap for all hosted targets
  (16 MiB), not a per-pointer-width value (`CODE_DEFAULT`).

- Executable arena memory is reclaimed wholesale when an unreachable module is
  dropped (the whole arena is freed and its stale per-buffer state purged; the
  OS may safely reuse the same virtual address), rather than by releasing each
  module's unused tail after finalization.

## Facts

- 2026-04-26 (8dc01387) pitfall: once a code buffer is reset or dropped, its
  fault-to-error-tail trap-range mappings are stale even if the OS later reuses
  the same virtual address for another module; reset and drop must purge the
  trap ranges that fall inside the buffer's interval before the address can be
  safely reused (code).

## Moves

- 2026-04-26 (8dc01387) replaced [[single-cap-tail-release]]: shrinking each
  module's arena tail to bound 32-bit address-space use could not satisfy both
  ends of the size dilemma; instead the full arena is freed wholesale when an
  unreachable module is dropped, with stale per-buffer state purged so the OS may
  safely reuse the same virtual address (code).
