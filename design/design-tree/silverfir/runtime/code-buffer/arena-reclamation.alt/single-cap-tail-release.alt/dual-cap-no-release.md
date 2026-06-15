- The hosted code-arena default capacity is selected by pointer width: 12 MiB on
  32-bit targets and 16 MiB on 64-bit targets.

## Moves

- 2026-04-25 (dc4e31a7) replaced by [[single-cap-tail-release]]: a 32-bit-specific
  smaller cap was the wrong lever — 16 MiB is too big to retain per module on
  32-bit address space yet 12 MiB is too small for some benchmarks; one cap with
  whole-arena reclamation on module drop serves both (diff).
