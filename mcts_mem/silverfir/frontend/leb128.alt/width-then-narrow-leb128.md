- LEB128 values are read by two functions, `read_leb128_unsigned` and
  `read_leb128_signed`, that accumulate into u64/i64 with no bound on the
  number of continuation bytes.

- Narrower widths are obtained by decoding the full-width value and numerically
  range-checking it down (e.g. `read_leb128_u32` rejects only when the u64
  result exceeds `u32::MAX`); an overlong encoding whose value still fits is
  accepted.

## Moves

- 2024-01-25 (9e801234) replaced [[leb128-crate]]: dropping the external leb128
  crate for an in-tree (value, bytes-consumed) reader folds into the engine's
  zero-runtime-dependency stance; low-stakes, since LEB128 speed matters only to
  the in-place interpreter, which need not be fast (sourced).

- 2024-03-08 (f9318ab7) replaced by [[leb128]]: decoding into u64/i64 and then
  numerically range-checking the narrowed result could not reject overlong
  encodings the spec deems malformed (a u32 spread across more continuation
  bytes than its width allows); per-width decoders that error once the shift
  reaches the target width's bit count catch the overlong case directly (code).
