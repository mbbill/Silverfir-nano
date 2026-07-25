- The dispatch handler set is produced at BUILD time, per target, and
  linked into the binary's own text segment; nothing is emitted, mapped,
  or made executable at run time (`interp_gen/`).

- The generator emits assembler SOURCE rather than machine code, and the
  platform assembler resolves every branch label and instruction encoding.

- The generator is compiled by the build script only, never by the crate.
  It shares exactly one module with the crate: the operand-class layout,
  which the generator enumerates and the linker classifies cells against.

- Handler slots are packed per variant family rather than as a dense
  op-by-variant matrix, since most ops vary only one or two of the three
  operand positions.

- The engine's entry points and the packed table of handler offsets reach
  the runtime as two linker symbols, with offsets measured from the blob's
  base rather than as absolute addresses.

- Every backend implements one interface and the driver owns what must be
  identical across them: which class combinations exist, the order they
  are emitted in, and the slot each is recorded at. A backend declares
  which operand classes and which ops it covers; anything it declines has
  no handler and takes the shared slow path.

- The interpreter no longer depends on the JIT subsystem being compiled
  in: having a generated engine for the target is the only condition.

## Facts

- 2026-07-25 rationale: emitting assembler text rather than machine code
  is what removes hand-counted branch deltas, which the emitter's own
  record names as its most frequent bring-up defect; the assembler also
  removes the need for a separate binary encoder per ISA, which is the
  cost that kept the engine single-target (code).

- 2026-07-25 measurement: the arm64 engine came through the port at parity
  — CoreMark 8139 (median of 5, quiet machine) against the 8143 baseline,
  and 334,820 emitted bytes against 334,828 — so the assembler text
  reproduces the tuned encoder instruction for instruction (code).

- 2026-07-25 measurement: linking one padding cell after each function's
  instruction stream, so the last handler's next-cell prefetch cannot read
  past the allocation, is free: interleaved x4 on CoreMark it measured
  7348/7031/6983/7091 with the pad against 7300/7046/7377/7177 without
  (code).

- 2026-07-25 measurement: the packed per-family slot table holds ~10.5 k
  live handlers in ~42 KB; the dense op-by-variant matrix the runtime
  emitter used would need ~160 KB for the same set (code).

- 2026-07-25 measurement: emitted engine size per backend — arm64 327 KB,
  x86-64 370 KB, RV64 362 KB, arm32 118 KB (A32) and 87 KB (Thumb-2), RV32
  93 KB. The 32-bit figures are the reduced class set: dropping the second
  pinned local takes a three-operand op from 100 variants to 48 (code).

- 2026-07-25 pitfall: `global_asm!` parses its argument as a format
  template, so an ARM register list (`push {r4-r12, lr}`) is a syntax
  error until its braces are doubled. Escaping in the text sink rather
  than in the one backend that needs it keeps the next emitter from
  rediscovering it (code).

- 2026-07-25 pitfall: `global_asm!` assembles against the target's BASE
  ISA, not the triple's full feature string, so a RISC-V backend must
  request `m`, `f` and `d` explicitly or `mul` and every float
  instruction is rejected on rv64gc (code).

- 2026-07-25 pitfall: the mirror case also bites — a feature that is an
  extension on one profile is BASELINE on another, and the assembler
  rejects being asked for it there (`.arch_extension idiv` is required on
  ARMv7-A and an error on ARMv8-M). "The target has this instruction" and
  "the assembler must be told about it" are separate questions, and a
  backend flag that conflates them builds on one profile only (code).

- 2026-07-25 pitfall: `cargo check` stops at metadata and never runs the
  assembler, so a handler set that does not assemble passes it clean (code).

- 2026-07-25 statement: the smoke row for a backend with no spectest
  harness has to be a build (code).

- 2026-07-25 rationale: a handler may bail to the slow path only where the
  bail leads to a trap, because the slow path is accumulator-oblivious and
  writes the frame slot, so a producer that bailed and then SUCCEEDED would
  leave its consumer reading a stale register (code).

- 2026-07-25 rationale: an op a backend cannot do fully is declined
  outright rather than bailed from at run time (code).

- 2026-07-25 rationale: a decline works because it lets the linker strip the
  pair's accumulator hints, which a runtime bail cannot (code).

## Moves

- 2026-07-25 replaced [[runtime-emission]]: emitting handlers into
  executable memory at instance setup cannot run where code generation is
  forbidden or impossible, which is the tier's own reason for existing,
  and it tied the interpreter to the JIT's executable-memory substrate
  (code)
