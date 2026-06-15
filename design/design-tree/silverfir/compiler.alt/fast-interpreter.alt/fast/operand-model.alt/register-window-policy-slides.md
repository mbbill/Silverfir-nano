- Hot operand-stack values live in a fixed set of registers; a compile-time
  policy pass walks the bytecode, records stack height per instruction, and
  chooses an optimal window state at each merge point, minimizing slides inside
  loops.

- Window overflow and underflow are handled by slide_up (spill the oldest
  register to memory) and slide_down (reload from memory) instructions inserted
  by the policy pass, including proactive slides placed before loops and blocks
  rather than only on overflow.

- At a branch, the policy pass inserts slides that make the source window match
  the target block's expected window state; consistency at merge points is achieved
  by explicit normalization, not by a depth-derived mapping.

## Moves

- 2026-01-20 (8136fd44) replaced by [[operand-model]]: WASM validation fixes the
  stack depth at every merge point, so making register assignment a pure function
  of depth makes merge-point state automatically consistent — removing the
  register-window design's per-merge-point policy pass and explicit slide
  instructions (diff).
