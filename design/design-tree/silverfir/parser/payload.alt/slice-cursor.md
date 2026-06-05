- A payload is a bare `&[u8]` slice. Each read splits the slice and replaces
  the stored slice with the remainder, so the cursor's position is implicit in
  how much slice is left.

- LEB128 reads go through `std::io::Cursor` wrapping the slice, using the
  external `leb128` crate's reader, then re-slice past the consumed bytes.

- Carving a sub-range splits the slice in two and returns both halves as
  borrowed slices; everything handed out is borrowed, never owned.

## Moves

- 2024-01-25 (9e801234) replaced by [[payload]]: a slice-shrinking cursor
  can only ever borrow, so an owned input could not produce owned sub-ranges;
  a Cow-plus-position cursor carries ownership through and keeps absolute
  offsets recoverable (diff).
