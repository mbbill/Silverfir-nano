- The byte reader (`Payload`) holds the input as a copy-on-write buffer plus a
  read position, so it can carry either borrowed module bytes or an owned copy
  without changing its type.

- Reads advance the position rather than reslicing; a split that hands out a
  sub-range returns borrowed bytes when the buffer is borrowed and owned bytes
  when it is owned, propagating the borrow/own choice outward.

- The reader decodes the wire primitives directly: LEB128 integers, fixed-width
  little-endian floats, single bytes, raw byte runs, and length-prefixed UTF-8.

## Facts

- 2024-01-25 (9e801234) rationale: the reader was reshaped from a borrowed
  slice into a copy-on-write buffer plus position so a module can be parsed from
  owned bytes (e.g. a file read into memory) without forcing a borrow; the
  split operation now propagates borrowed-vs-owned to every sub-range it hands
  out (diff).

## Moves

- 2024-01-25 (9e801234) replaced [[borrowed-slice-reader]]: a reader that was a
  bare borrowed slice could not own its bytes, so it could not back a module
  parsed from an in-memory copy; a copy-on-write buffer plus position can (diff).
