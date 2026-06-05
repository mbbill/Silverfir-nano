commit: 4336bd63

The handler-based decoder shipped before any real consumer existed. Its first
two `OpcodeHandler` implementations — the function validator and a disassembly
printer — do no semantic work at this point: the validator only adjusts an
indent counter and the printer only logs `offset: opcode immediate`. So the
streaming-handler shape was chosen on its own merits (no IR to allocate,
multiple passes share one decode walk), not retrofitted to fit a validator
that already worked. The validator and printer being interchangeable handlers
over the same walk is the evidence the abstraction was the point.
