- A standalone register-based interpreter: compute reads and writes named
  registers, with no implicit operand stack (the shared stack serves only
  calls/returns and rare fallbacks).
- The frontend builds expression trees during Wasm decode and materializes
  them into SSA at barriers; trees are a frontend-only concept, never passed
  to the middle layer. Locals and params are captured eagerly at access
  time.
- Functions compile through SSA: locals and temps convert to SSA with φ at
  joins.
- Hot patterns fuse by SSA-level pattern matching (madd, shladd,
  shladd_load, add-immediate), with a matching fusion lowering in the
  backend — patterns are structural, not opcode sequences.
- The middle layer (see `lir`) eliminates φ in place on SSA IR and
  allocates registers, producing LIR for the backends.
- Instructions are fixed 32-byte records {handler, alt, a, b, c}; wide or
  variable-arity payloads live in side blobs.
- Dispatch reuses the proven tail-chaining trampoline: `preserve_none` +
  `musttail`, with hot state pinned in registers as arguments.
- The C/Rust handler boundary is machine-generated and machine-checked:
  bindgen produces the FFI bindings and signature validation is automated —
  handler ABI consistency is never maintained by hand.
- SSA-level passes run before lowering: DCE, dead-φ and trivial-φ
  elimination, and a validator checking SSA well-formedness.
