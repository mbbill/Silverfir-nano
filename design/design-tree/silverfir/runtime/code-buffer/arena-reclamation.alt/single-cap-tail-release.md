- After finalizing a module's native code, hosted POSIX builds munmap the unused
  tail of the reserved arena (page-aligned to the offset); a 32-bit target does
  not retain a full arena of virtual address space per compiled module.

## Moves

- 2026-04-25 (dc4e31a7) replaced [[dual-cap-no-release]]: a 32-bit-specific
  smaller cap was the wrong lever — 16 MiB is too big to retain per module on
  32-bit address space yet 12 MiB is too small for some benchmarks; one cap with
  whole-arena reclamation on module drop serves both (diff).

- 2026-04-26 (8dc01387) replaced by [[arena-reclamation]]: shrinking each
  module's arena tail to bound 32-bit address-space use could not satisfy both
  ends of the size dilemma; instead the full arena is freed wholesale when an
  unreachable module is dropped, with stale per-buffer state purged so the OS may
  safely reuse the same virtual address (diff).
