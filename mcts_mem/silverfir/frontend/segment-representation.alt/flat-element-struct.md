- An element segment is a single struct holding a mode tag
  (Active/Passive/Declarative), a value type, a table index, and optional
  offset-expr, function-index, and init-expr fields.

- Accessing a mode-specific field (e.g. the offset expression) unwraps its
  Option unconditionally and panics when absent for the segment's mode.

## Moves

- 2024-02-16 (da94bda8) replaced by [[segment-representation]]: the flat struct
  carried a mode tag alongside optional offset/function-index/init-expr fields
  whose validity depended on the mode and were unwrapped unconditionally; an
  enum makes each mode carry exactly the fields it has so an offset cannot be
  read for a passive segment (code).
