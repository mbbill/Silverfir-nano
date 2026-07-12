- The IR builder annotates each instruction with a Stack Top Offset (STO) — the
  operand-stack height at that instruction — and handlers address operands at
  fixed offsets below STO (frame_base + sto - n) rather than tracking where each
  value actually lives.

- local.get and const emit instructions that copy the value onto the operand
  stack; with no compile-time tracking of which slot already holds a value,
  redundant local/constant copies are not elided.

- STO is the only per-instruction location information; operand input and output
  positions are derived from STO and the instruction's fixed stack signature.

## Facts

- 2025-12-04 (e5ae6403) rationale: recorded in no_stack.md (a recovered design
  document) — because WebAssembly is a structured stack machine, a valid
  function's stack height at every instruction is statically determinable and
  identical on all control-flow paths reaching it (guaranteed by validation), so
  each instruction's operand positions are compile-time constants; the builder
  computes each instruction's STO during construction and the operand stack
  becomes a fixed array of virtual registers addressed by precomputed slot
  indices, removing per-instruction sp increment/decrement. The document argues
  the stack-height tracking is self-contained in the fast backend's builder
  rather than reusing the validator's JumpTable, because the JumpTable only has
  entries for branch instructions and reusing it would couple the two subsystems
  and duplicate the control-flow logic (sourced).

## Moves

- 2025-12-05 (741d70e1) replaced [[sto-in-imm0-low16]]: the 16-bit STO field in
  imm0 capped the stack-top offset and crowded immediate data into imm1/imm2,
  while the wide imm2 was needed to later hold a three-slot (dest/src_a/src_b)
  encoding; moving STO to imm2 frees imm0 and imm1 as full 32-bit data fields and
  reserves imm2 for the planned per-instruction slot encoding (code).

- 2025-12-04 (9a490383) replaced [[pure-sp-memory-stack]]: the stack-based model
  threaded a runtime stack pointer through every handler and executed an sp
  increment/decrement per push/pop; since a valid wasm function's stack height at
  each instruction is statically known, the IR builder precomputes each
  instruction's stack-top offset (STO) and handlers address operands by absolute
  frame slot index, eliminating the runtime stack pointer and its per-instruction
  maintenance entirely (code).

- 2025-12-06 (b7b5dc6a) replaced by [[slot-tracking]]: STO addressing still
  copied locals and constants through the operand stack and re-derived operand
  positions from one stack-top offset; tracking each value's actual slot
  (Local/Operand/Temp) at compile time lets local.get and const emit no
  instruction at all (the consumer reads the source slot directly) and lets every
  operation carry explicit dest/src slot indices, eliminating the redundant copies
  the offset-only encoding could not avoid (code).
