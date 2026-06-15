- Each SSA-IR instruction is an `SsaInstKind` enum value (Value/Fill/Spill/Local*/Call);
  a Value op holds its operands and results as heap-allocated Vecs and its leaf
  opcode as an `SsaLeafOp`.

- An operand is an `SsaOperand` enum that is either a transient SSA value
  reference or an inline 64-bit constant absorbed by constant folding.

## Moves

- 2026-04-13 (782a6dfb) replaced by [[encoding]]: the per-variant enum carried
  heap-allocated arg/result vecs and inline 64-bit constants on every op,
  bloating the block op stream; a flat fixed-size record with payloads interned
  into program-level pools (primitive ops, constants, call ops, extra args) and
  packed 4-byte operands keeps SsaBlock.ops cache-dense and shrinks SSA-IR
  memory (diff)
