- Each instruction's Stack Top Offset is packed into the low 16 bits of imm0
  (imm0[0:15]); overflow data such as memidx sits in imm0[16:31]; 32-bit
  instruction data (indices, i32/f32 constants) goes in imm1 and 64-bit data
  (i64/f64 constants) in imm2.

- RefType (ref.cast/ref.test) packs its full reference type into imm2 via
  RefType::encode_to_u64, and BrOnCast packs its label index into imm1; STO still
  occupies imm0[0:15] for these instructions like every other.

## Facts

- 2025-12-05 (741d70e1) statement: imm2 is reserved for the Stack Top Offset
  specifically as a stepping stone to slot-based addressing — the 64-bit imm2 is
  sized to later hold three packed slot indices (dest, operand A, operand B), STO
  is parked there now and imm0/imm1 are freed for primary instruction data,
  avoiding a second immediate-layout churn when slot encoding lands (diff).

## Moves

- 2025-12-05 (741d70e1) replaced by [[sto-no-stack]]: the 16-bit STO field in
  imm0 capped the stack-top offset and crowded immediate data into imm1/imm2,
  while the wide imm2 was needed to later hold a three-slot (dest/src_a/src_b)
  encoding; moving STO to imm2 frees imm0 and imm1 as full 32-bit data fields and
  reserves imm2 for the planned per-instruction slot encoding (diff).
