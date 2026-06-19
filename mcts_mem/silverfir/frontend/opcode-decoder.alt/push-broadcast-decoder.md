- The decoder drives the decode loop and pushes each decoded opcode to every
  registered handler through an on_opcode(op, offsets, immediate) callback.

- Handlers implement on_decode_begin / on_opcode / on_decode_end and cannot
  request more than the single opcode currently being delivered.

## Moves

- 2024-02-01 (5bb02079) replaced [[single-handler-decoder]]: a single decode
  pass drove only one handler, so the disassembly printer had to be embedded
  inside the validator and gated on the log level; broadcasting each opcode to
  a list of registered handlers decouples the printer from the validator and
  lets either run independently over one decode of the body (code).

- 2025-08-13 (c7ae92e5) replaced by [[opcode-decoder]]: the push
  callback handed handlers one opcode at a time with no way to look ahead; a
  pull stream lazily decodes on demand and exposes a multi-op lookahead window
  (code).
