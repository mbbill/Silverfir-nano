- The frontend turns a raw `.wasm` byte slice into a parsed `Module` of typed
  entities (types, functions, tables, memories, globals, tags, elements, data)
  before any compilation begins.

- Loading a module runs decode then validate as two distinct passes over the
  binary, producing a fully-typed `Module` before any instantiation or
  compilation.

- The decoder owns only binary-format and structural correctness (section
  order, decodable opcodes, in-range indices) and reports those failures as
  malformed; it performs no type or semantic checking.

- The validator owns all semantic and type checking (subtyping, stack typing,
  constant-expression and initializer rules, export-name uniqueness) and
  reports those failures as invalid.

- The function decoder tracks a parallel stack of operand value types alongside
  its height and unreachable bookkeeping (`type_stack`); typed reference and
  GC operations resolve their refined result and branch types during decode
  rather than reconstructing them later.

## Facts

- 2025-10-26 (06e9f3bd) statement: the decoder accepts only core WebAssembly
  modules, which all carry binary version 1 (MVP through 3.0; new features are
  signalled by opcodes and section contents, never by the version field), and
  rejects Component Model binaries (version 0x0d / WASI Preview 3) and any
  other non-1 version as malformed — fixing the engine's scope to core modules
  (code).

- 2024-03-15 (306f8802) rationale: the global mutability flag byte admits only
  0x00 (const) and 0x01 (var); any other byte is malformed at the decode
  boundary, not invalid at the validator — keeping the binary-shape error where
  the bytes are read (code).

- 2025-06-20 (db6185a5) rationale: a datacount section declaring a nonzero
  count with no data section is rejected as malformed at decode time; the
  datacount/data segment-count agreement is a binary-shape check owned by the
  decoder, not the validator (code).

- 2025-10-06 (7c2ec193) pitfall: the 0xFB type-cast family must be numbered
  ref.test 0x14, ref.test-null 0x15, ref.cast 0x16, ref.cast-null 0x17,
  br_on_cast 0x18, br_on_cast_fail 0x19, any.convert_extern 0x1a,
  extern.convert_any 0x1b; an earlier table omitted the null-typed test/cast
  variants and the convert ops, assigning ref.cast/br_on_cast the wrong
  opcodes (0x15/0x16) and mis-decoding every module using them (code).
