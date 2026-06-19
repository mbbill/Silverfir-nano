- The number of top-of-stack registers is fixed at four, baked into the design
  rather than derived from a single constant.

- Every dispatch site selects its handler from a literal four-element array
  [op_X_D1, op_X_D2, op_X_D3, op_X_D4] written inline across the dispatch builder,
  one such array per opcode.

- The spill/fill handler is chosen by an explicit match over (count, variant)
  whose arms enumerate exactly variants 0..3 (D1..D4) for counts 1..4.

- Variant indices are computed with literal % 4 expressions and the variant range
  is asserted as variant <= 3.

- The cyclic TOS register-assignment macro is a power-of-two bitmask,
  TOS_REGISTER(height,position) = ((height)-(position)) & 3, which presumes the
  register count is a power of two.

## Moves

- 2026-01-28 (1dcc1655) replaced by [[operand-model]]: the register count 4 was
  baked into the design as literal D1..D4 handler-lookup arrays at every dispatch
  site, explicit (count,variant) match arms enumerating only variants 0..3, % 4
  variant formulas, and a power-of-two & 3 register mask, so the count could not be
  retuned without hand-editing every generator and handler and could never take a
  non-power-of-two value; making it the single build-time constant
  TOS_REGISTER_COUNT that every Rust and C generator derives from — variant names
  D1..DN, register names t0..t{N-1}, the C ABI register params, generated
  per-handler handler_lookup arrays, the variant_index formula ((depth-1) % N)+1,
  and the cyclic register-assignment reg = (height-position) % N with the mask
  falling back to modulo when N is not a power of two — lets the whole handler set
  be regenerated to any register count by changing one constant (code).
