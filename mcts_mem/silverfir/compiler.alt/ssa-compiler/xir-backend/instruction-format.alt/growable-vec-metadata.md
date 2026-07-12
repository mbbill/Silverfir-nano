- Side-table metadata is stored in growable Vec<Box<T>> fields on the lowered
  function; the no-move / no-remove discipline that keeps the instruction-stream
  pointers valid is maintained by code convention, not the type system.

## Moves

- 2025-10-15 (0ffe0f96) replaced by [[instruction-format]]: the instruction stream
  holds raw pointers into the metadata (and br_table tables hold raw pointers back
  into the instruction stream), so a Vec whose pop/clear could remove entries and
  whose buffer could move leaves dangling pointers; Box<[Pin<Box<T>>]> makes the
  no-move, no-remove contract type-enforced (code).
