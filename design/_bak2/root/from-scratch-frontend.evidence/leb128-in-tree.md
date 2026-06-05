---
commit: 9e801234
---
Day 1 (e4a20f95) used the external leb128 crate for LEB decoding; three
days later it was replaced by an in-tree implementation. The file carries
the why in its own comments: the reference implementation is an unrolled,
goto-chained C decoder (__stream_read_vu64_unchecked) from the author's
previous C interpreter — speed-honed technique carried across languages —
with the test suite credited to Mohanson's leb128 project (MIT). The
from-scratch choice is both control and performance.
