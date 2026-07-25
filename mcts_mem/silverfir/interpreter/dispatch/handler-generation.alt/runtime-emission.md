- The dispatch handler set was emitted into executable memory at instance
  setup by a micro-encoder private to the interpreter, deliberately not
  the JIT's encoder, with only the executable-memory substrate shared.

- The encoder produced machine-code words directly, and every branch inside
  a handler carried a hand-computed instruction delta.

- Handler variants were addressed through a dense op-by-variant matrix
  allocated per instance.

- Only one target had an encoder, and the interpreter required the JIT
  subsystem to be compiled in for the executable-memory substrate.

## Facts

- 2026-07-25 measurement: the emitted engine was 334,828 bytes and the
  buffer carried a hard 512 KB assert, so every added operand class was
  priced against a fixed allocation ceiling rather than against binary
  size (code).

- 2026-07-25 pitfall: hand-counted branch deltas and bit-packing
  collisions were this emitter's two recurring failure shapes, each caught
  by tests rather than by inspection (code).

- 2026-07-23 rationale: handlers were emitted at instance setup rather
  than generated at build time because it was the fastest correct
  bring-up; build-time generation was always the stated path to targets
  without runtime executable memory (sourced).

## Moves

- 2026-07-25 replaced by [[handler-generation]]: emitting handlers into
  executable memory at instance setup cannot run where code generation is
  forbidden or impossible, which is the tier's own reason for existing,
  and it tied the interpreter to the JIT's executable-memory substrate
  (code)
