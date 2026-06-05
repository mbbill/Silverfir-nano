commit: a719e961

The immediate union first tagged variants by raw byte shape — `U8`, `U32`,
`U8_U32`, `U32_U32`, `U8_U8` — so a memarg and a call-indirect pair were the
same `U32_U32`, indistinguishable without knowing the opcode. Commit a719e961
re-tagged every variant by its Wasm-level role (`LabelIndex`, `FunctionIndex`,
`MemArg { align, offset }`, `CallIndirectArgs { typeidx, tableidx }`,
`SelectTypes`, the table/memory arg structs, …) and gave the union a `Display`
impl. The shape-tagged form held the same bytes but pushed the meaning back onto
the consumer; the role-tagged form lets the validator pattern-match the operand
it expects (and error if the decoder handed it the wrong kind) and lets the
printer render a readable operand without an opcode lookup. The driving need is
the real validator that lands days later: it matches `Immediate::LabelIndex(..)`,
`Immediate::MemArg{..}`, etc. directly, which the raw-shape variants could not
support cleanly.
