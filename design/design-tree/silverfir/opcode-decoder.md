- A function body is decoded by a single pass over its bytecode that splits
  each instruction into its opcode and a decoded immediate, and pushes the pair
  to a handler trait (`OpcodeHandler`) rather than building an instruction
  list; consumers (validation, the opcode printer) implement the handler.

- The handler also receives each instruction's byte offset within the body, so
  a consumer can report positions without re-tracking the cursor.

- Opcode immediates are modelled as one enum (`Immediate`) covering every
  operand shape a Wasm instruction can carry (none, single integers/floats,
  pairs, the branch table's index vector, select's type vector).

- The opcode set is generated from a single table that pairs each mnemonic with
  its numeric value and a display string; the table covers the base opcodes and
  the two multi-byte prefixes (`0xFC` saturating/bulk-memory, `0xFD` vector),
  each prefix decoded as its own sub-opcode.
