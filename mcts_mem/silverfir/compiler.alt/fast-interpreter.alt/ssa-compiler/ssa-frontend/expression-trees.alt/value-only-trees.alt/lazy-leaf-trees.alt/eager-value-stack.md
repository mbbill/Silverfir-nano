- The builder holds a value stack of SSA values; each Wasm operation pops its
  operands and immediately pushes one SSA instruction (Const, Binary) into the
  current block; the SSA instruction sequence mirrors the Wasm operation
  order one-to-one.

- No fusion or tree inspection occurs: every arithmetic opcode becomes its own
  linear SSA instruction at decode time.

## Moves

- 2025-09-30 (7accd393) replaced by [[lazy-leaf-trees]]: emitting each Wasm operation immediately as a linear SSA instruction off an operand stack fixed the computation shape before it could be inspected, leaving no tree to match superinstruction patterns against; accumulating pure operations as expression trees on an expression stack and materializing them only at barriers exposes the tree shape to fusion (code).
