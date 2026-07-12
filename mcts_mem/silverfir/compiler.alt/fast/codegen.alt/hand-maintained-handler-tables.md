- The fast backend's handler set is declared by hand in three parallel places
  that must be kept in sync manually: a macro listing every op_ extern symbol, a
  DEFINE_OP_WRAPPER list in the C trampoline, and a map_handler match in the IR
  builder translating each wasm opcode to its handler pointer.

- Adding or removing an instruction requires editing all three lists plus the
  handler implementation; a missed edit silently desynchronizes the extern
  declarations, the C wrappers, and the opcode map.

## Moves

- 2025-12-03 (e76be08e) replaced by [[codegen]]: adding one fast-interpreter
  instruction previously required editing four places kept in sync by hand — the
  extern-declaration macro in handlers.rs, the DEFINE_OP_WRAPPER list in
  vm_trampoline.c, the map_handler match arm in ir_builder.rs, and the impl_
  function — with no single source of truth; a declarative handlers.def from
  which the C wrappers, Rust extern declarations, and the wasm-op-to-handler map
  are all build-generated removes the three-way manual synchronization (code).
