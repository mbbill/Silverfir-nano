- LEB128 decoding is provided by the external `leb128` crate, declared as a
  dependency of the engine crate.

- Its reader consumes a `std::io::Read`, so the payload reader wraps each byte
  slice in a `std::io::Cursor`, decodes, then re-slices the payload past the
  cursor's reported position.

## Moves

- 2024-01-25 (9e801234) replaced by [[leb128]]: the external crate's reader
  is driven through `std::io::Cursor`, forcing the payload reader to wrap
  every slice in a Cursor and re-slice past the consumed bytes; a slice-in,
  (value, consumed)-out function decodes in place with no Cursor and leaves
  room for the planned unrolled fast path (diff).
