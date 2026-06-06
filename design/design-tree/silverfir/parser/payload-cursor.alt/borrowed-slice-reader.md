- The byte reader is a bare borrowed slice of the input; consuming bytes
  reassigns the slice to its own tail.

- A split returns two borrowed sub-slices, both tied to the original input's
  lifetime; the reader can never own its bytes.

## Moves

- 2024-01-25 (9e801234) replaced by [[payload-cursor]]: a reader that was a bare
  borrowed slice could not own its bytes, so it could not back a module parsed
  from an in-memory copy; a copy-on-write buffer plus position can (diff).
