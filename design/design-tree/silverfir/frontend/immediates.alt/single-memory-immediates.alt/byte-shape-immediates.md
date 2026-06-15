- Decoded immediates are tagged only by their byte shape (e.g. U8, U32,
  U8_U32, U32_U32); two operands with the same encoding but different roles
  are indistinguishable to the handler.

- Block-type and br_table immediates are carried as the same generic shape
  tags.

## Moves

- 2024-01-29 (a719e961) replaced by [[single-memory-immediates]]: raw byte-shape
  variants conflated operands of different meaning behind one tag (the single
  U32 tag served funcidx, localidx, globalidx and tableidx indistinguishably),
  so the decoder now names each immediate by its operand role and the handler
  reads meaning without re-deriving it from the opcode (diff).
