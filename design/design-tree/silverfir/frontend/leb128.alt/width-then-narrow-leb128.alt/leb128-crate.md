- Unsigned and signed LEB128 are decoded with the external leb128 crate's
  reader driven by a `std::io::Cursor` over the byte slice.

## Moves

- 2024-01-25 (9e801234) replaced by [[width-then-narrow-leb128]]: dropping the
  external leb128 crate for an in-tree (value, bytes-consumed) reader folds into
  the engine's zero-runtime-dependency stance; low-stakes, since LEB128 speed
  matters only to the in-place interpreter, which need not be fast (author).
