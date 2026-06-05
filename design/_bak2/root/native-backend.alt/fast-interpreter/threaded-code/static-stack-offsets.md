- No stack pointer exists at runtime: WebAssembly's validation rules make
  the stack height at every instruction a static fact, so each instruction
  carries its stack-top offset (STO) as a compile-time immediate, and
  operand slots are fp-relative constants derived from STO plus the op's
  stack signature.
- The fast IR builder tracks stack heights itself during IR construction —
  self-contained, deliberately not reusing the validator's jump table
  (incomplete coverage, fragile coupling).
- Block parameters and results never move: a block is a re-view of the same
  slots, with the result landing where the parameters began.
- Hot state still travels as `preserve_none` register arguments between
  handlers — ctx, pc, and the frame pointer — with no sp among them.
