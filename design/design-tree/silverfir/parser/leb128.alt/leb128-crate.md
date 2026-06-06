- LEB128 integer decoding is delegated to an external crate, declared as a
  dependency of the core library.

- Each read constructs an `io::Cursor` over the remaining bytes, decodes through
  the crate's reader interface, and then advances the parser by the cursor's
  reported position.

## Moves

- 2024-01-25 (9e801234) replaced by [[leb128]]: an external LEB128 crate drove
  decoding through an `io::Cursor`, an indirection unwanted on the hot parse
  path; a vendored decoder returning the consumed-byte count lets the reader
  advance directly (diff).
