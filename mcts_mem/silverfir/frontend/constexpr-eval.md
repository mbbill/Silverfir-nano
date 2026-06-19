- Constant init expressions (global, element offset, data offset, element init,
  table init) are evaluated at instantiation by a small dedicated stack machine
  over the retained bytecode that handles only the constant-expression opcodes
  (const, ref.null, ref.func, global.get, and the wasm 3.0 GC allocation ops),
  not the full interpreter.

## Facts

- 2025-10-05 (b3ce11ba) pitfall: ref.null in a constant expression takes a heap
  type as its immediate (wasm 3.0), which is either an abstract heap-type byte
  or a concrete type index encoded as a signed LEB128 and so may be multi-byte;
  decoding it as a whole ValueType (the prior approach) misreads
  concrete-type-index heap types, so it must go through the dedicated heap-type
  parser and build a nullable ref from the parsed heap type (code).
