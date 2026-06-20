- The arm64 backend emits both saturating and trapping float-to-int
  truncations inline as native `fcvtzs`/`fcvtzu`; trapping truncations add an
  inline NaN check plus bounds checks with a one-way trap exit and no register
  preservation, unlike arm32/x86_64 which call out-of-line helpers
  (`lower_trapping_trunc`).

## Moves

- 2026-03-22 (5eca447e) replaced [[helper-call-saturating-trunc]]: ARM64's
  fcvtzs/fcvtzu already match Wasm saturating-truncation semantics (NaN->0,
  overflow clamps to min/max) and saturating trunc can never trap, so the
  out-of-line helper call was pure overhead and the conversion can be a single
  native instruction (code).
