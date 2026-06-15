- A built XIR instruction carries a handler pointer, an optional branch-target
  offset field (relative instruction index), and split immediate fields of 32,
  32, and 64 bits.

- Branch targets are stored separately from the data immediates rather than
  encoded in them; each encoding must know which physical field its datum lives
  in.

## Moves

- 2025-10-21 (dc951039) replaced by [[instruction-format]]: the earlier
  instruction carried a special Option branch-offset field alongside split
  32/32/64-bit immediates, forcing each encoding to know which physical field its
  data lived in; collapsing to a handler pointer plus three interchangeable 64-bit
  immediates lets any datum (constants, branch targets, memory/table indices) be
  encoded uniformly and matches a single fixed repr(C) layout shared with the C
  trampoline (diff).
