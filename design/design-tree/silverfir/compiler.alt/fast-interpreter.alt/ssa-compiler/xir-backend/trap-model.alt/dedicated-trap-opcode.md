- A runtime trap inside a handler sets the error and returns a null
  next-instruction pointer; the lowering emits a dedicated trap opcode for
  `unreachable`, whose C wrapper sets the error and unwinds without tail-chaining.

- The handler-name table distinguishes the terminal and the trap opcodes as two
  separate special handlers.

## Moves

- 2025-10-11 (d9281a5a) replaced by [[trap-model]]: a trapping handler returned a
  null next-pc that the generic tail-chain wrapper would have dereferenced;
  returning a shared terminal instruction lets every handler trap through one path
  with no null check in the C wrappers (diff).
