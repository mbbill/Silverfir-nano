---
status: abandoned
---
# Convert to a register machine

The wasm3-style approach: rewrite the stack-machine bytecode into virtual-register
instructions whose operands are immediates indexing copy-on-write virtual
registers (a local is shared on read, given a fresh slot on write). `local.get`
effectively disappears and `local.set` becomes a register-to-register copy.

## In practice

Must:
- Assign virtual-register indices to locals and stack values and encode each
  instruction's operands as immediates indexing those registers.

Must not:
- Be relied on for automatic fusion: register operands are loaded from the
  instruction stream, so the compiler must assume aliasing and cannot prove one
  fused instruction's output feeds the next, which blocks cross-instruction
  optimization (see facts/stack-fusion-godbolt-5-vs-15-instructions.md).
- Attempt automated superinstruction generation: per-instruction operand patterns
  explode combinatorially with fusion length, which is why register interpreters
  use a few hand-selected fused patterns rather than automatic discovery.
