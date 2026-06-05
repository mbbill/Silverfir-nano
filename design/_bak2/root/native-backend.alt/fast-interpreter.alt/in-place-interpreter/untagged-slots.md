- Operand-stack slots are untagged raw 64-bit words (`RawValue`); every value
  type is bit-cast into the same 8-byte slot.
- Value types are static facts established by validation; no type is tagged,
  stored, or checked at runtime.
