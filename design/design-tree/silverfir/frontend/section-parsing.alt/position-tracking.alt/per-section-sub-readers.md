- Each section is parsed from its own sub-reader carved out of the module
  reader; the section's end is enforced by requiring that sub-reader to be
  empty afterward.

- Function code is the remaining bytes of its own sub-reader, with no record of
  where it sat in the module.

## Moves

- 2024-01-28 (3a8b5fd6) replaced by [[position-tracking]]: sub-section readers
  reset their position to zero and so cannot report a function body's absolute
  byte offset within the module; parsing from one reader and tracking positions
  preserves the code_offset needed to address disassembled instructions (diff).
