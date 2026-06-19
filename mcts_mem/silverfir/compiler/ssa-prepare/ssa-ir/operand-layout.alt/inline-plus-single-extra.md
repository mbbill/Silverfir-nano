- A primitive SSA op carries at most three operands: two inline in the `SsaInst`
  plus one in the block's `extra_args` vector; a primitive with more than three
  args is rejected as unsupported in the flat `SsaInst` layout (`pack_primitive_args`).

## Moves

- 2026-04-16 (ccceb9a8) replaced by [[operand-layout]]: the flat SsaInst layout
  held at most two inline operands plus a single overflow operand and
  hard-errored on any primitive with more than three args; GC array ops need up
  to five operands (array.copy 5, array.fill/array.init_* 4,
  array.new_fixed/wide struct.new variable), so the third-and-beyond operands
  are spilled into a contiguous extra_args span whose start index is carried in
  meta (code)
