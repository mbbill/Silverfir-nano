- All binary reads (LEB128 ints, fixed-width floats, length-prefixed UTF-8, raw
  byte spans) go through one reader (`Payload`) that advances an explicit
  position cursor over the input and reports short reads as malformed.

- The reader does not consume input by rewriting a borrowed slice; carrying a
  position cursor lets a section be split off as a bounded sub-region while the
  whole reader keeps reporting absolute byte offsets within the module.

## Moves

- 2024-01-25 (9e801234) replaced [[borrow-only-slice-reader]]: a borrow-only
  &[u8] reader cannot own caller-supplied input; holding the bytes as Cow with
  a position cursor lets the same reader carry both borrowed and owned
  (Cow::Owned) module data (code).
