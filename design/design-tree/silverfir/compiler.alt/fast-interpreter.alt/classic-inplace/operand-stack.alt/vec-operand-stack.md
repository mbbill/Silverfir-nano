- The operand stack is a Vec<RawValue> used directly as a stack, pushing and
  popping via Vec::push/Vec::pop.

- Popping multiple operands returns a freshly allocated owned Vec<RawValue>.

- Locals are addressed by a per-frame base offset into the same Vec, and result
  shifting rotates the tail and truncates.

- The backing Vec is created with a fixed heuristic initial capacity.

## Moves

- 2025-06-22 (3bedfb76) replaced by [[operand-stack]]: a Vec used directly as a
  stack reallocates as it grows and its multi-pop allocated a fresh owned Vec on
  every call, so the operand stack is reworked into a pre-allocated buffer
  addressed by an explicit stack pointer whose multi-pop returns a borrowed
  slice, eliminating per-operation heap traffic in the interpreter hot loop
  (diff).
