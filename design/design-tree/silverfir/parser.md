- The parser walks the binary once, section by section in encounter order, and
  drives a mutable accumulator (`ModuleBuilder`) that is consumed into an
  immutable `Module` at the end.

- Section order is enforced by a state machine: each non-custom section must
  have a larger id than the previous one, custom sections may appear anywhere,
  and the data-count section is the one exception that may precede code and
  data.

- Each section's parser reads directly from the single module-wide cursor and
  is bounded by the declared section length: after a section is parsed the
  cursor must sit exactly at the section's end, otherwise the binary is
  malformed.

- A function's code is recorded with its absolute byte offset from the start of
  the module (`code_offset`), not just its bytes.

## Facts

- 2024-01-28 (3a8b5fd6) rationale: carving each section into its own sub-cursor
  reset the byte position to zero, so a function's offset within the whole
  module could not be recovered; parsing from the single module-wide cursor and
  bounding by declared length keeps absolute offsets available (diff).

## Moves

- 2024-01-28 (3a8b5fd6) replaced [[per-section-cursor]]: a per-section cursor
  numbered bytes from the section start, so absolute in-module code offsets were
  unrecoverable; parsing from one module-wide cursor preserves them (diff).
