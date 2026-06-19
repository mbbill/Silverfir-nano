- Decoded immediates are tagged by their operand role (`LabelIndex`,
  `FunctionIndex`, `MemArg{align,offset}`, `CallIndirectArgs`, ...); a
  handler reads an operand's meaning directly instead of re-deriving it from the
  opcode.

- Memory-accessing operands carry a real memory index (`MemoryIndex`,
  `MemoryInitArgs`, `MemoryCopyArgs`); a module operating on more than one
  memory is representable.

## Facts

- 2025-10-05 (8a6e5f01) pitfall: array.get/array.set/array.get_s/array.get_u
  carry only a type-index immediate — the element index is a stack operand, not
  an immediate; decoding them with a second LEB field (the field-index shape
  used by struct.get/struct.set) over-consumes the byte stream and
  desynchronizes the next instruction's decode, so the array accessors must be
  split out from the struct accessors in the immediate decoder (code).

## Moves

- 2025-10-01 (b7d4218c) replaced [[single-memory-immediates]]: the reserved-byte
  encoding could address only memory 0 (it rejected any non-zero value), so it
  could not express WebAssembly 3.0 modules that declare and operate on more
  than one memory (code).
