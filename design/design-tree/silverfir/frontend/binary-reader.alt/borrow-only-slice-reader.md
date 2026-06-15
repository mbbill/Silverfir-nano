- The reader holds a borrowed &[u8] and consumes input by replacing that slice
  with the unread remainder; it can only read from borrowed input.

- Splitting off a sub-region returns two fresh borrowed readers over the
  original slice.

## Moves

- 2024-01-25 (9e801234) replaced by [[binary-reader]]: a borrow-only &[u8]
  reader cannot own caller-supplied input; holding the bytes as Cow with a
  position cursor lets the same reader carry both borrowed and owned
  (Cow::Owned) module data (diff).
