- A safe-Rust driver loop executed the folded instruction stream one
  instruction at a time through the shared single-instruction executor —
  the same cells the native chain runs, with no emitted code.

- The loop was selectable per instance against the native chain and
  doubled as the correctness oracle (three-way differential tests: Rust
  loop vs native chain vs JIT, bit-exact) and as the dynamic profiler
  (per-op dispatch counts and the old-basis fold-ratio accounting).

## Facts

- 2026-07-23 measurement: CoreMark 837.9±8.2 (release, 5 runs) — 5.05×
  slower than the native chain on the identical folded stream, and ~15%
  slower again with the always-on profiling counters (code).

- 2026-07-23 measurement: the dynamic fold-ratio verification (0.416
  measured vs 0.489 predicted) was produced by this loop's profiler
  before removal; the numbers are recorded on [[interpreter]] (code).

## Moves

- 2026-07-23 replaced by [[dispatch]]: one high-performance interpreter,
  not two execution paths — the oracle role moved to the JIT differential
  tests and spectest, and the shared single-instruction executor remains
  as the native chain's slow path (sourced)
