- Each call frame owns its spill slots as a heap-allocated Vec sized to the
  function's slot count, allocated on entry and dropped on return (`CallFrame`).

- A frame also carries its result-slot indices as a heap-allocated Vec, and the
  live frames are held in a stack of such frame structures.

## Moves

- 2025-11-30 (0ebc0961) replaced by [[call-return]]: each call allocated a fresh
  Vec for the frame's spill slots (and another for result-slot indices), putting
  heap allocation on every call and return; a single pre-allocated buffer with
  bump-pointer push and instant pop makes call/return allocation-free (code).
