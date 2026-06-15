- VIR carries a distinct instruction variant per (operation, type): separate
  BinaryI32/BinaryI64/BinaryF32/BinaryF64 for binary ops and
  ConstI32/ConstI64/ConstF32/ConstF64 for constants, with the type encoded in the
  variant name rather than a field.

- Unary operations carry no type field; their type is recovered from instruction
  metadata.

## Moves

- 2025-10-22 (1494e762) replaced by [[vir-two-stage-middle]]: one instruction per
  operation carrying an inline ValueType tag mirrors the SSA IR's own variants,
  giving a 1:1 SSA-to-VIR lowering instead of a fan-out of per-type variants while
  still keeping types per-instruction for backend dispatch (diff).
