- Each saturating truncation op lowers to a call to the `arm64_saturating_trunc`
  helper: the source bits go in X0, an op-code in X1, and the helper returns the
  saturated result in X0 (no error possible).

- The helper dispatches on the op-code to one of eight software
  saturating-conversion routines that hand-implement NaN->0 and min/max clamping.

## Moves

- 2026-03-22 (5eca447e) replaced by [[saturating-trunc]]: ARM64's fcvtzs/fcvtzu
  already match Wasm saturating-truncation semantics (NaN->0, overflow clamps to
  min/max) and saturating trunc can never trap, so the out-of-line helper call
  was pure overhead and the conversion can be a single native instruction (code).
