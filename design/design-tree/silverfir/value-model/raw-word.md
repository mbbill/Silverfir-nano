- Internally every WebAssembly value is stored untyped as a single 64-bit word;
  numeric and reference types share that representation and are reinterpreted by
  the static type the validator already proved, not by a runtime tag.

- The static type is supplied at every access site by conversion functions that
  reinterpret the word for a given `ValueType` (`from_raw` / `to_raw`); an i32
  is zero-extended into the word and an f32 is stored in its low 32 bits.

## Facts

- 2025-10-26 (3f750d35) pitfall: converting an externally-facing Value::I32
  into the internal 64-bit RawValue word must zero-extend (cast through u32),
  not sign-extend: `*v as u64` sign-fills the high 32 bits, corrupting the word
  relative to how a statically-i32-typed consumer reads it; the fix is `*v as
  u32 as u64` at every host-result/param/struct-get/array-get boundary (diff).

## Moves

- 2025-06-22 (a061476a) replaced [[tagged-value-stack]]: the validator already
  proves every operand's static type, so a per-slot runtime type tag on the
  operand stack is redundant; storing each value as an untyped 64-bit word and
  reinterpreting it by the proven static type at each access drops the tag
  overhead and lets the per-value type check be removed from stack verification
  (diff).
