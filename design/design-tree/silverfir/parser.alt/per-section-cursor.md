- Each section's bytes are carved out of the module cursor into a fresh
  per-section cursor that numbers bytes from zero, and the section is parsed
  from that sub-cursor.

- A section is complete when its sub-cursor is empty; leftover bytes mean the
  binary is malformed.

## Moves

- 2024-01-28 (3a8b5fd6) replaced by [[parser]]: a per-section cursor numbered
  bytes from the section start, so absolute in-module code offsets were
  unrecoverable; parsing from one module-wide cursor preserves them (diff).
