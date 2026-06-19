- The operand stack stores tagged Value enum elements (Vec<Value>), each
  carrying its own WebAssembly type at runtime.

- Stack verification checks both length and that each slot's runtime type tag
  equals the expected value type.

## Facts

- 2024-03-18 (6d712544) pitfall: converting an i32 wasm value to a usize index
  must zero-extend through u32 (val as u32 as usize), not sign-extend; a
  negative i32 cast straight to usize sign-extends to a huge address and
  corrupts table/memory indexing (code).

## Moves

- 2025-06-22 (a061476a) replaced by [[raw-word]]: the validator already proves
  every operand's static type, so a per-slot runtime type tag on the operand
  stack is redundant; storing each value as an untyped 64-bit word and
  reinterpreting it by the proven static type at each access drops the tag
  overhead and lets the per-value type check be removed from stack verification
  (code).
