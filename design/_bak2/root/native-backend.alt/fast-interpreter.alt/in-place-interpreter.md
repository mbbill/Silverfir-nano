- Executes raw Wasm bytecode directly; there is no internal IR.
- Immediates are LEB128-decoded at runtime — the accepted cost of zero
  load-time transformation.
- Structured control flow resolves in O(1) through a precomputed side jump
  table (see `jump-table`).
- Operand-stack slots are untagged 64-bit words (see `untagged-slots`).
- One contiguous value stack serves all frames: a caller's argument values
  become the callee's locals in place — no copying at the call boundary.
- Calls never recurse in Rust: an explicit frame stack drives a single
  dispatch loop.
