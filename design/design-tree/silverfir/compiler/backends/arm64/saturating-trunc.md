- The arm64 backend emits saturating float-to-int truncations inline as native
  `fcvtzs`/`fcvtzu`; only trapping truncations, which must detect out-of-range
  and NaN and raise a Wasm trap, stay an out-of-line helper call
  (`lower_trapping_trunc`).

## Moves

- 2026-03-22 (5eca447e) replaced [[helper-call-saturating-trunc]]: ARM64's
  fcvtzs/fcvtzu already match Wasm saturating-truncation semantics (NaN->0,
  overflow clamps to min/max) and saturating trunc can never trap, so the
  out-of-line helper call was pure overhead and the conversion can be a single
  native instruction (diff).
