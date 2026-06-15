- An element segment is an `Active`/`Passive`/`Declarative` sum type and a data
  segment is an `Active`/`Passive` sum type; each mode carries exactly the
  fields it has (an active segment's offset expression cannot be read for a
  passive segment).

## Facts

- 2025-06-20 (ae76a1b8) pitfall: an element segment's leading kind/flags field
  is LEB128-encoded (a varint), not a single raw byte; reading it with read_u8
  instead of read_leb128_u32 misdecodes any multi-byte encoding and selects the
  wrong element form, so the decode boundary must read it as a u32 varint
  (diff).

## Moves

- 2024-02-16 (da94bda8) replaced [[flat-element-struct]]: the flat struct
  carried a mode tag alongside optional offset/function-index/init-expr fields
  whose validity depended on the mode and were unwrapped unconditionally; an
  enum makes each mode carry exactly the fields it has so an offset cannot be
  read for a passive segment (diff).

- 2024-02-16 (01f2a6db) replaced [[flat-data-struct]]: the flat struct used an
  is_active bool with an optional offset expression that was unwrapped for
  active segments; an enum makes the active variant carry the offset expression
  and the passive variant omit it (diff).
