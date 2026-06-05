- A payload is the parser's byte-stream reader: it owns a `Cow` over the
  bytes plus a running position index, and offers typed reads (single byte,
  fixed-length byte run, LEB128 integers, length-prefixed UTF-8) that advance
  the index (`Payload`).

- Reads never shrink the underlying slice; the cursor moves the position index
  forward, so the original byte range and absolute offsets stay recoverable
  throughout the parse.

- Carving a sub-range yields a `Cow` that preserves the borrow/own nature of
  the source: a sub-range of borrowed bytes is borrowed, a sub-range of owned
  bytes is copied. A `Cow` (or a borrowed slice) converts into a payload
  directly.

## Moves

- 2024-01-25 (9e801234) replaced [[slice-cursor]]: a slice-shrinking cursor
  can only ever borrow, so an owned input could not produce owned sub-ranges;
  a Cow-plus-position cursor carries ownership through and keeps absolute
  offsets recoverable (diff).
