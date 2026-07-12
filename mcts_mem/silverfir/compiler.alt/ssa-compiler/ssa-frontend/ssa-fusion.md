- Instruction fusion runs as an SSA-to-SSA pass over the completed SSA
  function, building a value-to-defining-instruction map and rewriting a
  matched root instruction into one fused instruction.

- The fused shapes are a small set of hardware-backed patterns: shladd
  (scaled-index addressing), madd/msub, add_imm, shladd_load and shladd_store
  (fused scaled-index memory access).

- Fusion rewrites only the root instruction and leaves the now-unused operand
  instructions in place; their removal is left to the separate dead-code
  elimination pass that runs last.

## Facts

- 2025-11-28 (3f4bdb19) pitfall: msub fusion must match sub(mul(a,b), c) to
  fold into the runtime's madd(a,b,c,is_add=false) = (a*b)-c; an earlier draft
  matched sub(c, mul(a,b)), which computes c-(a*b), the wrong sign, so the
  subtraction's operand order has to align with the fused instruction's fixed
  evaluation order (code).

## Moves

- 2025-11-28 (1fae526c) replaced [[tree-time-fusion]]: matching fusion patterns on the expression tree during materialization could only see a single unmaterialized tree and could not fuse across barriers such as local.tee; running fusion as a pass over completed SSA lets it see through those barriers and match patterns whose operands span barrier boundaries (code).
