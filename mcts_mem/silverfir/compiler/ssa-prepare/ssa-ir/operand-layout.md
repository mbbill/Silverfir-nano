- A primitive SSA op carries two inline operands in the `SsaInst` plus, when it
  has more, an overflow span in the block's `extra_args` vector whose start index
  is carried in the op's meta; operand-heavy primitives (GC `array.copy` 5,
  `array.fill`/`array.init_*` 4, wide `struct.new`/`array.new_fixed`) are
  representable (`pack_primitive_args`, `extra_args`).

## Moves

- 2026-04-16 (ccceb9a8) replaced [[inline-plus-single-extra]]: the flat SsaInst
  layout held at most two inline operands plus a single overflow operand and
  hard-errored on any primitive with more than three args; GC array ops need up
  to five operands (array.copy 5, array.fill/array.init_* 4,
  array.new_fixed/wide struct.new variable), so the third-and-beyond operands
  are spilled into a contiguous extra_args span whose start index is carried in
  meta (code)
