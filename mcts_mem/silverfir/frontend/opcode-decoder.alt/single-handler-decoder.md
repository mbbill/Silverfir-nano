- A function body is decoded by a free decode_function that drives exactly one
  OpcodeHandler, calling on_opcode per decoded opcode.

- The disassembly printer is owned as a field of the validator and invoked
  inline from the validator's on_opcode only when info-level logging is
  enabled.

## Moves

- 2024-02-01 (5bb02079) replaced by [[push-broadcast-decoder]]: a single decode
  pass drove only one handler, so the disassembly printer had to be embedded
  inside the validator and gated on the log level; broadcasting each opcode to a
  list of registered handlers decouples the printer from the validator and lets
  either run independently over one decode of the body (code).
